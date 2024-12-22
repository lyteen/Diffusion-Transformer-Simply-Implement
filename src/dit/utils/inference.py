import os
import torch
from typing import Optional, Tuple
from safetensors.torch import load_file, load_model

def safetensors_load_file_inference(
        model_path: str,
        model: object, 
        dataloader: object
    ) -> Tuple[list, list]:
    assert os.path.exists(model_path), f"model_path not exist"

    model.eval()
    model_path = load_file(model_path)
    model.load_state_dict(model_path)
    eval_pred = [] # store model predicted results to list
    tar_eval_pred = [] # store targets to list
    loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            tar_eval_pred.extend(target.item() for target in targets) # append the true targets to the list
            logits, _ = model(inputs)
            for log_idx, logit in enumerate(logits):
                predicted_class = logit.argmax(-1).item() # along the last dimension return predicted result
                target_class = tar_eval_pred[batch_idx * dataloader.batch_size + log_idx]
                if predicted_class != target_class:
                    loss += 1.0
                    eval_pred.append(predicted_class)
    print(f"accuracy: {1 - (loss / len(tar_eval_pred))}") # compute the accuracy
    return eval_pred, tar_eval_pred

def safetensors_load_model_inference(
        model_path: str,
        dataloader: object
) -> Tuple[list, list]:
    assert os.path.exists(model_path), f"model_path not exist"
    
    model = load_model(model_path)
    model.eval()
    eval_pred = [] # store model predicted results to list
    tar_eval_pred = [] # store targets to list
    loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            tar_eval_pred.extend(target.item() for target in targets) # append the true targets to the list
            logits, _ = model(inputs)
            for log_idx, logit in enumerate(logits):
                predicted_class = logit.argmax(-1).item() # along the last dimension return predicted result
                target_class = tar_eval_pred[batch_idx * dataloader.batch_size + log_idx]
                if predicted_class != target_class:
                    loss += 1.0
                    eval_pred.append(predicted_class)
    print(f"accuracy: {1 - (loss / len(tar_eval_pred))}") # compute the accuracy
    return eval_pred, tar_eval_pred