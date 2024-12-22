import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class DitModelArgs:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embed: Optional[int] = 384 # (B, T, C) "C", Must make n_embed // n_heads == head_dim
    n_heads: Optional[int] = 6
    head_dim: Optional[int] = 64
    bias: Optional[bool] = False
    dropout: Optional[float] = 0.2
    patch_size: Optional[int] = 16
    in_channels: Optional[int] = 3 # RGB channels
    n_labels: Optional[int] = 10
    img_w: Optional[int] = 16
    img_h: Optional[int] = 16
    patch_w = img_w // patch_size
    patch_h = img_h // patch_size
    n_patch = patch_w * patch_h
    n_layers: Optional[int] = 12
    
    base_scale: Optional[float] = 1.0 / (n_embed ** 0.5)
    time_step: Optional[int] = 1000
    rope_theta: Optional[float] = 10000.0
    eps: Optional[float] = 1e-5

# use Euclidean norm make atten score only focus on high dimension vector direction
def Euclidean_norm(x: torch.Tensor):
    return x / x.norm(p=2, dim=-1, keepdim=True)

# function init the freqs_cis
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs) # Use outer product
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

# reshape the freqs_cis match the image size
def reshape_for_broadcast(
        freqs_cis: torch.Tensor,
        x: torch.Tensor
) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"x.shape[1]:{x.shape[1]}, x.shape[-1]: {x.shape[-1]}, freqs_cis.shape: {freqs_cis.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

# function init the Rotary embedding
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # e.g. (B, T, n_heads, head_dim) -> (B, T, n_heads, head_dim // 2) represented the complex
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # (1, T, 1, head_dim // 2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # (B, T, n_heads, head_dim)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3) # 3 mean flatten the head_dim dimension
    return xq_out.type_as(xq), xk_out.type_as(xk) 

# noisy is relative with the time step
class Noise(nn.Module):
    def __init__(self, time_step: int) -> None:
        super().__init__()
        _, _, self.alphas_cumprod = self._create_noise_parameters(time_step)

    def _create_noise_parameters(self, time_step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        betas = torch.linspace(0.0001, 0.02, time_step) # (time_step, )
        alphas = 1 - betas # (time_step, )
        alphas_cumprod = torch.cumprod(alphas, dim=-1) # [a1,a2,a3,....] ->  [a1, a1*a2, a1*a2*a3, ..., a1*a2*...*a{t}]
        alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[: -1]), dim=-1) # alpha_t-1 product (time_step,),  [1, a1, a1*a2, a1*a2*a3, ..., a1*a2*...*a{t-1}]
        variance = (1 - alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod) # the denoise variance
        return alphas, variance, alphas_cumprod

    # use forward add noise
    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x) # (batch_size, in_channels, img_h, img_w)
        batch_alphas_cumprod = self.alphas_cumprod[time_step].view(x.size(0),1,1,1)
        x = torch.sqrt(batch_alphas_cumprod) * x + torch.sqrt(1-batch_alphas_cumprod)*noise # create added noise image
        return x,noise # (batch_size, in_channels, img_h, img_w)

class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.half_n_embed = n_embed // 2
        # use math.log(10000) scales the exponent
        half_embed = torch.exp(torch.arange(self.half_n_embed) * (-1 * math.log(10000) / (self.half_n_embed - 1)))
        self.register_buffer('half_embed', half_embed)

    # time_step: (batch_size, )
    def forward(self, time_step):
        time_step = time_step.view(time_step.size(0), 1) # (time_step, 1)
        half_embed = self.half_embed.unsqueeze(0).expand(time_step.size(0), self.half_n_embed) # (time_step, n_embed // 2)
        half_time_embed = half_embed * time_step # (time_step, n_embed // 2)
        time_embed = torch.cat((half_time_embed.sin(), half_time_embed.cos()), dim=-1) # (time_step, n_embed)
        return time_embed

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(self.dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.dim, f"x.size: {x.size}, n_embed: {self.dim}"
        out = self._norm(x.float()).type_as(x)
        return out * self.weight

class MultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_embed = config.n_embed

        self.key = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.query = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.value = nn.Linear(config.n_embed, config.n_embed, bias=False)

        # sqrt_qk hyperphere norm
        self.sqk_init_val = 1.0
        self.sqk_init_scaling = config.base_scale
        self.sqk = torch.nn.Parameter(self.sqk_init_scaling * torch.ones(self.n_embed, dtype=torch.float32)) # use for scale norm key and query.e.g.  

    # x: (batch_size, seq_len, n_embed)
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.n_embed, f"x.shape: {x.shape}, n_embed: {self.n_embed}"
        assert self.n_embed == self.n_heads * self.head_dim, f"n_embed: {self.n_embed}, n_heads: {self.n_heads}, head_dim: {self.head_dim}"

        q, k, v = self.query(x), self.key(x), self.value(x)
        q = q.view(x.size(0), x.size(1), self.n_heads, self.head_dim) # (batch_size, n_patch, n_heads, head_dim)
        k = k.view(x.size(0), x.size(1), self.n_heads, self.head_dim)
        v = v.view(x.size(0), x.size(1), self.n_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis) # use Rotary position embedding

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # use the hyperphere norm
        sqk = (self.sqk * (self.sqk_init_val / self.sqk_init_scaling)).view(1, self.n_heads, 1, self.n_embed // self.n_heads)
        q = sqk * Euclidean_norm(q) # normalize size: 1
        k = sqk * Euclidean_norm(k)

        atten = torch.matmul(q, k.transpose(-1, -2) * self.head_dim ** -0.5)
        atten = F.softmax(atten, dim=-1)
        atten = torch.matmul(atten, v)
        atten = atten.permute(0,2,1,3).contiguous().view(x.size(0), x.size(1), x.size(2)) # (batch_size, seq_len, n_embed)
        return atten

class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed * 4),
            nn.GELU(),                                                                  
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embed * 4, config.n_embed)
        )
    def forward(self, x):
        return self.net(x)

class DitBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads

        self.gamma1 = nn.Linear(config.n_embed, config.n_embed)
        self.gamma2 = nn.Linear(config.n_embed, config.n_embed)
        self.beta1 = nn.Linear(config.n_embed, config.n_embed)
        self.beta2 = nn.Linear(config.n_embed, config.n_embed)
        self.alpha1 = nn.Linear(config.n_embed, config.n_embed)
        self.alpha2 = nn.Linear(config.n_embed, config.n_embed)

        # self.atten_norm = RMSNorm(config.n_embed, eps=config.eps)
        self.multi_atten = MultiHeadAttention(config)
        # self.ffn_norm = RMSNorm(config.n_embed, eps=config.eps)
        self.feed_forward = MLP(config)

        # mlp hyperphere norm
        self.mlp_alpha_init_val = 0.05
        self.mlp_alpha_init_scaling = config.base_scale # 1 / head_dim ** 0.5
        self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling * torch.ones(config.n_embed, dtype=torch.float32)) # torch.Tensor[head_dim ** -0.5, ...], torch.Size([n_embed])

        # muti-attention hyperphere norm
        self.atten_alpha_init_val = 0.1
        self.atten_alpha_init_scaling = config.base_scale
        self.atten_alpha = torch.nn.Parameter(self.atten_alpha_init_scaling * torch.ones(config.n_embed, dtype=torch.float32))

    # x: (batch_size, seq_len, n_embed), condition: (batch_size, n_embed)
    def forward(self, x: torch.Tensor, condition: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        gammal1 = self.gamma1(condition) # (batch_size, n_embed)
        beta1 = self.beta1(condition)
        alpha1 = self.alpha1(condition)

        gammal2 = self.gamma2(condition)
        beta2 = self.beta2(condition)
        alpha2 = self.alpha2(condition)

        scale_atten = self.atten_alpha * (self.atten_alpha_init_val / self.atten_alpha_init_scaling) # scale score is 0.05
        scale_atten = torch.abs(scale_atten)

        scale_mlp = self.mlp_alpha * (self.mlp_alpha_init_val / self.mlp_alpha_init_scaling) # scale score is 0.05
        scale_mlp = torch.abs(scale_mlp)

        # y = self.atten_norm(x)
        y = (1 + gammal1.unsqueeze(1)) * x + beta1.unsqueeze(1) # (batch_size, 1, n_embed) -> (batch_size, seq_len, n_embed)
        y = self.multi_atten(y, freqs_cis)
        y = y * alpha1.unsqueeze(1)
        y = Euclidean_norm(y) * scale_atten + Euclidean_norm(x) * (1.0 - scale_atten)
        
        # z = self.ffn_norm(y)
        z = (1 + gammal2.unsqueeze(1)) * y + beta2.unsqueeze(1)
        #z = self.feed_forward(z)
        z = z * alpha2.unsqueeze(1)
        return (1.0 - scale_mlp) * Euclidean_norm(y) + scale_mlp * Euclidean_norm(z) # (batch_size, seq_len, n_embed)

class Dit(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.patch_size = config.patch_size
        self.time_step = config.time_step
        self.n_embed = config.n_embed
        self.n_patch = config.n_patch
        self.label_emb = nn.Embedding(config.n_labels, config.n_embed)
        self.time_step_emb = nn.Sequential(
            TimeEmbedding(config.n_embed),
            nn.Linear(config.n_embed, config.n_embed),
            nn.SiLU(),
            nn.Linear(config.n_embed, config.n_embed)
        )
        self.conv_layer = nn.Conv2d(in_channels=self.in_channels, out_channels= self.in_channels * self.patch_size ** 2, kernel_size=self.patch_size, padding=0, stride=self.patch_size) # recover to patch
        self.patch_emb = nn.Linear(self.in_channels * self.patch_size ** 2, config.n_embed)
        self.dit_block = nn.ModuleList([DitBlock(config) for _ in range(config.n_layers)])
        self.layer_norm = RMSNorm(config.n_embed, eps=config.eps)
        self.linear_layer = nn.Linear(config.n_embed, config.in_channels * config.patch_size ** 2)
        
        self.freqs_cis = precompute_freqs_cis(
            config.n_embed // config.n_heads,
            config.n_patch, # since use Rotary position embedding, the n_patch = patch_h * patch_w must known and can't changed
            config.rope_theta,
        )

        self.get_noise = Noise(self.time_step)
        self.loss_fn = nn.L1Loss()
    
    # init model weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # x: (batch_size, in_channels, img_h, img_w), label: (batch_size, )
    def forward(self, x: torch.Tensor, label: torch.Tensor):
        # make img [0, 1] -> [-1, 1]
        x = x * 2 - 1

        # add the rand time step
        t = torch.randint(0, self.time_step, (x.size(0),))

        # get the forward noise img and added noise img
        x, noise = self.get_noise(x, t)

        # embed
        label = self.label_emb(label) # (batch_size, n_embed)
        time_step = self.time_step_emb(t) # (batch_size, n_embed)
        condition = label + time_step

        # patchify
        assert x.size(1) == self.in_channels, f"img_channels: {x.size(1)} can match in_channels: {self.in_channels}"

        x = self.conv_layer(x) # (batch_size, in_channels * patch_size ** 2, patch_h, patch_w)
        _, _, patch_h, patch_w = x.shape

        assert self.n_patch == patch_h * patch_w, f"n_patch size: {self.n_patch} can match patch_w: {patch_w}, patch_h: {patch_w}"
        
        x = x.permute(0,2,3,1) # (batch_size, patch_h, patch_w, in_channels * patch_size ** 2)
        x = x.view(x.size(0), patch_h * patch_w, x.size(3)) # (batch_size, n_patch, in_channels * patch_size ** 2)
        x = self.patch_emb(x) # (batch_size, n_patch, n_embed)

        # dit block
        for layer in self.dit_block:
            x = layer(x, condition, self.freqs_cis)

        # layer norm
        x = self.layer_norm(x)

        # linear
        x = self.linear_layer(x) # (batch_size, n_patch, in_channels * patch_size ** 2)

        # reshape
        x = x.view(x.size(0), patch_h, patch_w, self.in_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 2, 4, 5) # (batch_size, in_channels, patch_h, patch_w, patch_size, patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5) # (batch_size, in_channels, patch_h, patch_size, patch_w, patch_size)
        x = x.reshape(x.size(0), self.in_channels, patch_h * self.patch_size, patch_w * self.patch_size)   # (batch, in_channels, img_h, img_w)

        # compute loss
        loss = self.loss_fn(x, noise)

        return x, loss
    
    # function `generate`: input sample noise img, and custom time_step generate the denoise img
    @torch.no_grad()
    def generate(self, x: torch.Tensor, time_step: torch.Tensor, label: torch.Tensor):

        x = x * 2 - 1 # convert [1, 0] to [-1, 1]

        # label emb
        label = self.label_emb(label) #   (batch_size, n_embed)

        # time emb
        time_embed =self.time_step_emb(time_step)  #   (batch_size, n_embed)
        
        # condition emb
        condition = label + time_embed
        
        # patchify emb
        #assert x.size(1) == self.in_channels, f"img_channels: {x.size(1)} can match in_channels: {self.in_channels}"
        x = self.conv_layer(x)  # (batch_size, in_channels * patch_size ** 2, patch_h, patch_w)
        _, _, patch_h, patch_w = x.shape
        x = x.permute(0,2,3,1)    # (batch_size, patch_h, patch_w, in_channels * patch_size ** 2)
        x = x.view(x.size(0), patch_h * patch_w, x.size(3)) # (batch_size, n_patch, patch_size ** 2)
        x = self.patch_emb(x) # (batch_size, n_patch, n_embed)
        # dit blocks
        for layer in self.dit_block:
            x = layer(x, condition, self.freqs_cis)
        
        # layer norm
        x = self.layer_norm(x)
        
        # linear
        x = self.linear_layer(x) # (batch_size, n_patch, in_channels * patch_size ** 2)
        
        # reshape
        x = x.view(x.size(0), patch_h, patch_w, self.in_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 2, 4, 5) # (batch_size, in_channels, patch_h, patch_w, patch_size, patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5) # (batch_size, in_channels, patch_h, patch_size, patch_w, patch_size)
        x = x.reshape(x.size(0), self.in_channels, patch_h * self.patch_size, patch_w * self.patch_size)   # (batch, in_channels, img_h, img_w)
        
        return x