import os
import torch
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from src.dit.model import Dit, DitModelArgs

# if pass model will generate error image? TODO: what make it?
"""
# load model
def load_model(model: object, model_path: str) -> object:
    global MODEL_LOADED
    if not MODEL_LOADED:
        model.eval()
        try:
            model_path = load_file(model_path)
            model.load_state_dict(model_path)
            MODEL_LOADED = True
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Model file not found: {model_path}")
            return None
        except RuntimeError as e: # Catch potential errors during state dict loading
            print(f"Error loading state dict: {e}")
            return None
    else:
        pass
    return model
"""

# get the denoise parameter
def backward_denoise(model: object, model_path: str, img: torch.Tensor, classes: torch.Tensor, time_step: int) -> list[torch.Tensor]:
    # load model
    model = model.eval()
    model_path = load_file(model_path)
    model.load_state_dict(model_path)
    # get the origin noise img
    denoised_images = [img.clone(), ] 
    # get the denoise parameters
    alphas, variance, alphas_cumprod = model.get_noise._create_noise_parameters(time_step)

    # from time_step to 0.
    for time in range(time_step - 1, -1, -1):
        t = torch.full((img.size(0),), time) # (time * batch_size)
        noise = model.generate(img, t, classes) # get the denoise img
        shape = (img.size(0), 1, 1, 1)
        mean = 1 / torch.sqrt(alphas[t].view(*shape)) * \
            (img - (1 - alphas[t].view(*shape))/torch.sqrt(1 - alphas_cumprod[t].view(*shape)) * noise)
        if time != 0:
            img = mean + torch.randn_like(img) * \
                torch.sqrt(variance[t].view(*shape))
        else:
            img = mean
        img = torch.clamp(img, -1.0, 1.0).detach() # make sure the img value in the range(-1.0, 1.0) and dont inpput compute the model grdient update
        denoised_images.append(img) # add current denoised image
    return denoised_images

# save denoise model
def save_inference_image(image_paths: list[str], denoise_img: torch.Tensor, time_step: int) -> None:
    # get current work path
    WORKING_DIR = os.getcwd()
    WORKING_DIR = WORKING_DIR + "/outputs/"
    # save inference process
    # set denoise process image size
    img_rows_num = 5
    img_cols_num = 6
    num_images = img_cols_num * img_rows_num # 30 images
    step = time_step // num_images
    fig, axes = plt.subplots(img_rows_num, img_cols_num, figsize=(4,4))
    axes = axes.flatten()

    for i in range(num_images):
        t = time_step - 1 - (i * step)
        img_show = denoise_img[time_step - 1 - t].squeeze(0) # from the origin noise image begin
        img_show = (img_show + 1) / 2 # convert [-1, 1] to [0, 1]
        if img_show.ndim == 4:
            img_show = img_show.squeeze(0)
        img_show = img_show.permute(1,2,0).cpu().numpy() # convert Tensor to numpy to plot
        axes[i].imshow(img_show)
        axes[i].axis('off')
    plt.tight_layout()
    fig.savefig(WORKING_DIR + "inference.png", bbox_inches='tight')
    plt.close(fig)  # close the figure to avoid memory issues
    image_paths.append(WORKING_DIR + 'inference.png')

    # save last denoise image
    plt.imshow(((denoise_img[-1] + 1) / 2).squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(WORKING_DIR + 'result.png', bbox_inches='tight')
    plt.close()
    image_paths.append(WORKING_DIR + 'result.png')
    
    return

# inference image
def model_inference(image_paths: list[str], model_path: str, input_value: int, time_step: int) -> None:
    # load model config
    try:
        config = DitModelArgs(n_labels=10, in_channels=1)
        model = Dit(config=config)
        model_params = sum(p.numel() for p in model.parameters())
        model_params_size = model_params * 4 / (1024 ** 2)
        print("Model parameter:", model_params)
        print(f'Model size: {model_params_size:.2f} MB')
    except Exception as e:  # Catch any exception during initialization
        print(f"Error during model initialization: {e}")
        return None
    # set the random noise
    origin_noise = torch.rand(size=(1, 1, 16, 16))
    classes = torch.tensor([input_value], dtype=torch.long)
    
    denoise_img = backward_denoise(model, model_path, origin_noise, classes, time_step=time_step)
    save_inference_image(image_paths, denoise_img, time_step)
    return