import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from safetensors.torch import save_file, save_model, load_model, load_file
import warnings
warnings.filterwarnings("ignore", message="Using a target size")

# Custom the train model iterations and save model conditions 
"""
    -Req:
        model should return logits and loss
    -Args:
        save_model_dir: save_model_dir
        epochs: train iterations
        min_loss: if loss < min_loss save model file
        save_model_epochs: if epoch/save_model_epochs == 0 save model file
        model: model
        dataloader: dataloader
        optimizer: optimizer
"""
def train_safetensors_save_file( # save_file only store the model weight dictionary
        save_model_dir: str,
        epochs: int,
        min_loss: float,
        save_model_epochs: int,
        model: nn.Module, 
        dataloader: torch.Tensor, 
        optimizer: optim,
        ) -> None:
    assert os.path.isdir(save_model_dir), f"save_model_dir not exist"
    assert save_model_epochs > 0, f"error save_model_epochs: {save_model_epochs}"
    num = 0
    for epoch in range(epochs):
        model.train() # set the train mode that you can use dropout and batch normalization
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True) as pbar:
            for _, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                _, loss = model(inputs, targets)
                
                if loss != None:
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item()) # load the progress bar
                    pbar.update(1) # update the progress bar by one batch
                if loss < min_loss:
                    dir_path = os.path.join(save_model_dir, f"model_weights_num_{num+1}.safetensors")
                    save_file(model.state_dict(), dir_path) # use safetensors.torch save model weights dictionary
                    num += 1

        if epoch != 0 and epoch % save_model_epochs == 0:
            dir_path = os.path.join(save_model_dir, f"model_weights_epoch_{epoch+1}.safetensors")
            save_file(model.state_dict(), dir_path)

# Custom the load pre-train model and fine tuned
"""
    -Req:
        model should return logits and loss
    -Args:
        save_model_dir: save_model_dir
        pre_train_model_path: pre_train_model_path
        epochs: train iterations
        min_loss: if loss < min_loss save model file
        save_model_epochs: if epoch/save_model_epochs == 0 save model file
        model: model
        dataloader: dataloader
        optimizer: optimizer
"""
def fine_tuned_safetensors_load_file_model(
        save_model_dir: str,
        pre_train_model_path: str,
        epochs: int,
        min_loss: float,
        save_model_epochs: int,
        model: nn.Module, 
        dataloader: torch.Tensor, 
        optimizer: optim
        ) -> None:
    assert os.path.isdir(save_model_dir), f"save_model_dir not exist"
    assert os.path.exists(pre_train_model_path), f"pre_train_model_path not exist"
    assert save_model_epochs > 0, f"error save_model_epochs: {save_model_epochs}"

    pre_train_model_weights = load_file(pre_train_model_path) # use safetensor.load_file only load the model weights
    model.load_state_dict(pre_train_model_weights)

    num = 0
    for epoch in range(epochs):
        model.train() # set the model to training mode
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True) as pbar:
            for _, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad() # clear gradients from the previous step
                _, loss = model(inputs, targets)
                
                if loss != None:
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item()) # load the progress bar
                    pbar.update(1) # update the progress bar by one batch
                if loss < min_loss:
                    dir_path = os.path.join(save_model_dir, f"fineTuned_weights_num_{num+1}.safetensors")
                    save_file(model.state_dict(), dir_path) # use safetensors.torch save model weights dictionary
                    num += 1

        if epoch != 0 and epoch % save_model_epochs == 0:
            dir_path = os.path.join(save_model_dir, f"fineTuned_weights_epoch_{epoch}.safetensors")
            save_file(model.state_dict(), dir_path)

def train_safetensors_save_model(
        save_model_dir: str,
        epochs: int,
        min_loss: float,
        save_model_epochs: int,
        model: nn.Module, 
        dataloader: torch.Tensor, 
        optimizer: optim,
    ) -> None:
    assert os.path.isdir(save_model_dir), f"save_model_dir not exist"
    assert save_model_epochs > 0, f"error save_model_epochs: {save_model_epochs}"
    num = 0
    for epoch in range(epochs):
        model.train() # set the train mode that you can use dropout and batch normalization
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True) as pbar:
            for _, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                _, loss = model(inputs, targets)
                
                if loss != None:
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item()) # load the progress bar
                    pbar.update(1) # update the progress bar by one batch
                if loss < min_loss:
                    dir_path = os.path.join(save_model_dir, f"model_num_{num+1}.safetensors")
                    save_model(model, dir_path) # use safetensors.torch save model weights dictionary
                    num += 1

        if epoch != 0 and epoch % save_model_epochs == 0:
            dir_path = os.path.join(save_model_dir, f"model_epoch_{epoch+1}.safetensors")
            save_model(model, dir_path)

def fine_tune_safetensors_save_model(
        save_model_dir: str,
        pre_train_model_path: str,
        epochs: int,
        min_loss: float,
        save_model_epochs: int,
        dataloader: torch.Tensor, 
        optimizer: optim
        ) -> None:
    assert os.path.isdir(save_model_dir), f"save_model_dir not exist"
    assert os.path.exists(pre_train_model_path), f"pre_train_model_path not exist"
    assert save_model_epochs > 0, f"error save_model_epochs: {save_model_epochs}"

    model = load_model(pre_train_model_path)
    num = 0
    for epoch in range(epochs):
        model.train() # set the model to training mode
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True) as pbar:
            for _, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad() # clear gradients from the previous step
                _, loss = model(inputs, targets)
                
                if loss != None:
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item()) # load the progress bar
                    pbar.update(1) # update the progress bar by one batch
                if loss < min_loss:
                    dir_path = os.path.join(save_model_dir, f"fineTuned_model_num_{num+1}.safetensors")
                    save_model(model, dir_path) # use safetensors.torch save model weights dictionary
                    num += 1

        if epoch != 0 and epoch % save_model_epochs == 0:
            dir_path = os.path.join(save_model_dir, f"fineTuned_model_epoch_{epoch}.safetensors")
            save_model(model, dir_path)