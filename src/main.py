import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.train import train_model
from src.evaluate import evaluate_model

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    config_path = "config/config.yaml"
    config = load_config(config_path)
    
    # Flatten the config dictionary for easier access
    config_flat = {
        **config['data'],
        **config['model'],
        **config['training'],
        **config['evaluation']
    }
    
    train_model(config_flat)
    evaluate_model(config_flat)