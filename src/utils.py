import torch
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import json
from src.model import get_model

def save_model(model, path, model_name, image_size, batch_size, epochs, learning_rate):
    filename = f"{model_name}_img{image_size[0]}x{image_size[1]}_batch{batch_size}_epochs{epochs}_lr{learning_rate:.5f}.pth"
    save_path = os.path.join(path, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

def load_model(model_name, image_size, batch_size, epochs, learning_rate, save_path):
    filename = f"{model_name}_img{image_size[0]}x{image_size[1]}_batch{batch_size}_epochs{epochs}_lr{learning_rate:.5f}.pth"
    model_path = os.path.join(save_path, filename)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"The model file at {model_path} does not exist.")

    model = get_model(model_name, num_classes, weights=None)
    state_dict = torch.load(model_path)

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

def get_class_weights(labels):
    labels = np.array(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float)

def load_label_names(save_path):
    with open(os.path.join(save_path, 'label_mapping.json'), 'r') as f:
        label_mapping = json.load(f)
    return label_mapping['labels']