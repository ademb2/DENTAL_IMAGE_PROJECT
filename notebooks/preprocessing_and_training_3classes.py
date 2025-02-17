#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data:
image_dir = "../data/raw/all_images"
label_file = "../data/raw/dental_images.csv"
image_size = (224, 224)
batch_size = 32
validation_split = 0.5
test_split = 0.3
image_type = "photo"
# model:
model_name = "VGG16"
weights = "IMAGENET1K_V1"
num_classes = 3

# training:
epochs = 40
learning_rate = 0.0001
save_path = "../outputs/models/vgg16_new_aug"

# evaluation:
metrics = ["accuracy", "precision", "recall", "f1_score"]


# # data loading and preprocessing
# 

# In[2]:



import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image
import json

# class ImageDataset(Dataset):
#     def __init__(self, image_dir, labels_df, base_transform=None, augment_transform=None):
#         """
#         Args:
#             image_dir (str): Path to the directory with images.
#             labels_df (pd.DataFrame): DataFrame with filtered image labels.
#             base_transform (callable, optional): A function/transform to apply to the original images.
#             augment_transform (callable, optional): A function/transform to apply for augmented images.
#         """
#         self.image_dir = image_dir
#         self.base_transform = base_transform
#         self.augment_transform = augment_transform

#         # If augment_transform is provided, duplicate the labels to create a dataset with both original and augmented images
#         if augment_transform:
#             self.labels = pd.concat([labels_df.assign(aug_flag=False), labels_df.assign(aug_flag=True)], ignore_index=True)
#         else:
#             self.labels = labels_df.assign(aug_flag=False)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.image_dir, self.labels.iloc[idx]['file_name'])
#         try:
#             image = Image.open(img_name).convert("RGB")
#         except FileNotFoundError:
#             raise FileNotFoundError(f"Image {img_name} not found.")

#         label = self.labels.iloc[idx]['label']
#         aug_flag = self.labels.iloc[idx]['aug_flag']

#         # Apply the appropriate transform based on the augmentation flag
#         if aug_flag and self.augment_transform:
#             image = self.augment_transform(image)  # Only apply augment_transform to the duplicated images
#         elif self.base_transform:
#             image = self.base_transform(image)  # Apply base_transform to original images

#         return image, torch.tensor(label, dtype=torch.long)

class ImageDataset(Dataset):
    def __init__(self, image_dir, labels_df, base_transform=None, augment_transform=None):
        """
        Args:
            image_dir (str): Path to the directory with images.
            labels_df (pd.DataFrame): DataFrame with filtered image labels.
            base_transform (callable, optional): A function/transform to apply to the original images.
            augment_transform (callable, optional): A function/transform to apply for augmented images.
        """
        self.image_dir = image_dir
        self.base_transform = base_transform
        self.augment_transform = augment_transform

        # If augment_transform is provided, duplicate the labels to create a dataset with both original and augmented images
        if augment_transform:
            self.labels = pd.concat([labels_df.assign(aug_flag=False), labels_df.assign(aug_flag=True)], ignore_index=True)
        else:
            self.labels = labels_df.assign(aug_flag=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.labels.iloc[idx]['file_name'])
        try:
            image = Image.open(img_name).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image {img_name} not found.")

        label = self.labels.iloc[idx]['label']
        aug_flag = self.labels.iloc[idx]['aug_flag']

        # Convert image to numpy array
        image = np.array(image)

        # Apply the appropriate transform based on the augmentation flag
        if aug_flag and self.augment_transform:
            image = self.augment_transform(image=image)['image']  # Only apply augment_transform to the duplicated images
        elif self.base_transform:
            image = self.base_transform(image=image)['image']  # Apply base_transform to original images

        return image, torch.tensor(label, dtype=torch.long)


def create_transforms(image_size):
    base_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2()
    ])

    augment_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.OneOf([
            A.RandomBrightnessContrast( p=1.0, brightness_limit=(0.5), contrast_limit=(0)),
            A.RandomBrightnessContrast( p=1.0, brightness_limit=(0), contrast_limit=(0.5)),
            A.RandomGamma(p=1.0, gamma_limit=(30, 150))],p=0.5),
        A.Rotate(limit=50, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Blur(p=0.1, blur_limit=(3, 3)),
        A.GaussNoise(p=0.3, var_limit=(10.0, 60.0)),
        A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.7, 1.0), p=0.3),
        A.ColorJitter(p=0.3, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.08, 0.07)),
        A.Sharpen(p=0.3),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2()
    ])

    return base_transform, augment_transform



def load_labels(label_file, image_type):
    labels = pd.read_csv(label_file)
    if 'image_type' not in labels.columns or 'category' not in labels.columns:
        raise KeyError("Label file must contain 'image_type' and 'category' columns.")
    labels = labels[labels['image_type'] == image_type]
    #replace values in category column from environmnet_cohort to control_cohort
    labels['category'] = labels['category'].replace('Environment_cohort', 'Control_cohort')
    labels['label'] = labels['category'].astype('category').cat.codes
    # Create a mapping of label codes to names
    category_mapping = labels['category'].astype('category').cat.categories
    label_mapping = {str(code): name for code, name in enumerate(category_mapping)}
    
    
    # Save the mapping to JSON
    with open(os.path.join(save_path, 'label_mapping.json'), 'w') as f:
        json.dump({"labels": label_mapping}, f)
        
    return labels

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def create_datasets(image_dir, labels, base_transform, augment_transform, test_split, validation_split):
    indices = list(range(len(labels)))
    train_idx, temp_idx  = train_test_split(
        indices, test_size=test_split, stratify=labels['label'], random_state=42)

    test_idx, val_idx = train_test_split(
        temp_idx, test_size=validation_split, stratify=labels.iloc[temp_idx]['label'], random_state=42)

    train_dataset = ImageDataset(image_dir, labels.iloc[train_idx], base_transform=augment_transform)
    val_dataset = ImageDataset(image_dir, labels.iloc[val_idx], base_transform=base_transform)
    test_dataset = ImageDataset(image_dir, labels.iloc[test_idx], base_transform=base_transform)

    return train_dataset, val_dataset, test_dataset

def load_data():
    labels = load_labels(label_file, image_type)
    base_transform, augment_transform = create_transforms(image_size)
    train_dataset, val_dataset, test_dataset = create_datasets(
        image_dir, labels, base_transform, augment_transform, test_split, validation_split)
    return create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)





# In[4]:


# #count totla labels in train_loader
# total_labels = 0   
# for i, (images, labels) in enumerate(train_loader):
#     total_labels += len(labels) 
    
# print(f"Total labels in train_loader: {total_labels}")
# total_labels = 0   
# for i, (images, labels) in enumerate(test_loader):
#     total_labels += len(labels) 
    
# print(f"Total labels in test_loader: {total_labels}")
# total_labels = 0
# for i, (images, labels) in enumerate(val_loader):
#     total_labels += len(labels) 
    
# print(f"Total labels in val_loader: {total_labels}")


# # Model creation

# In[5]:


import torch.nn as nn
import torchvision.models as models

def get_model(model_name, num_classes, weights=None):
    """
    Initialize and return a model with the specified architecture and weights.

    Args:
        model_name (str): Name of the model architecture to use.
        num_classes (int): Number of output classes.
        weights (str, optional): Pre-trained weights to load.

    Returns:
        nn.Module: The initialized model with the custom classifier.
    """
    if model_name == "VGG16":
        model = models.vgg16(weights=weights)
        # for param in model.features.parameters():
        #     #param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        # model.classifier = nn.Sequential(
        # nn.Linear(model.classifier[0].in_features, 2048),  # Reduce the size of the first fully connected layer
        # nn.ReLU(inplace=True),
        # nn.Dropout(p=0.6),  # Increase dropout rate
        # nn.Linear(2048, 1024),  # Reduce the size of the second fully connected layer
        # nn.ReLU(inplace=True),
        # nn.Dropout(p=0.6),  # Increase dropout rate
        # nn.Linear(1024, num_classes)  # Output layer
    
    # elif model_name == "ResNet18":
    #     model = models.resnet18(weights=weights)
    #     model.fc = nn.Linear(model.fc.in_features, num_classes)
    # elif model_name == "DenseNet121":
    #     model = models.densenet121(weights=weights)
    #     model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    # # Add other models as needed
    # else:
    #     raise ValueError(f"Model {model_name} not supported")
    # Wrap the model with DataParallel
    
    return model


# # training 

# In[6]:


import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
# from src.model import get_model
# from src.data_loading import load_data
# from src.utils import save_model
# from src.metrics import calculate_metrics

def train_model():
    
    
     # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    accuracies = []
    
    #load data
    train_loader, val_loader, _ = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(model_name, num_classes, weights)
    # Move model to GPU and wrap with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
     # Calculate class weights based on training labels
    train_labels = [label for _, label in train_loader.dataset]
    class_weights = get_class_weights(train_labels)
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    
    best_val_loss = float('inf')
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            #labels = labels.long()
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # Free GPU memory
            torch.cuda.empty_cache()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
        
        # # Validation
        # val_metrics = calculate_metrics(model, val_loader, device, criterion)
        # print(f"Validation Metrics: {val_metrics}")
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        # Validation
        model.eval()
        val_metrics = calculate_metrics(model, val_loader, device, criterion)
        print(f"Validation Metrics: {val_metrics}")
        epoch_val_loss = val_metrics.get('loss', float('inf'))  # Set default value if loss is not returned
        val_losses.append(epoch_val_loss)
        accuracies.append(val_metrics['accuracy'])
        
        # # Save checkpoint
        # checkpoint = {
        #     'epoch': epoch + 1,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': running_loss / len(train_loader),
        # }
        # torch.save(checkpoint, save_path + f"_epoch{epoch+1}.pth")
        # Save checkpoint if performance is better
        # Save checkpoint if performance is better
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss / len(train_loader),
            }
            torch.save(checkpoint, save_path + f"_epoch{epoch+1}_best.pth")
            print(f"Checkpoint saved for epoch {epoch + 1} with validation loss: {epoch_val_loss:.4f}")
        else:
            print(f"No improvement for epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
    
    save_model(model, save_path, model_name, image_size, batch_size, epochs, learning_rate)

    # Plotting
    plt.figure(figsize=(12, 5))
    
    epochs_range = range(1, len(train_losses) + 1)  # Ensure x-axis matches length of metrics lists
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path + "_training_metrics.png")
    plt.show()


# # evaluation

# In[7]:


import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
# from src.data_loading import load_data
# from src.utils import load_model
# from src.metrics import calculate_metrics

def evaluate_model():
    _, _, test_loader = load_data()
    model = load_model(
        model_name= model_name, 
        image_size=image_size,  # Replace with actual image size
        batch_size=batch_size, 
        epochs=epochs, 
        learning_rate=learning_rate
    )
   
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    test_metrics = calculate_metrics(model, test_loader, device)
    print(f"Test Metrics: {test_metrics}")
    
    # Load label names and codes from JSON
    label_mapping = load_label_names()
    num_classes = len(label_mapping)
    # Reverse the mapping for confusion matrix display
    inv_label_mapping = {int(k): v for k, v in label_mapping.items()}
    
    # Confusion Matrix
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
           
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=inv_label_mapping.values(), yticklabels=inv_label_mapping.values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path + "_confusion_matrix.png")
    plt.show()


# # metrics

# In[8]:


import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(model, dataloader, device, criterion=None):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    if criterion:
        metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


# # utils
# 

# In[9]:


import torch
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import json
# from src.model import get_model

def save_model(model, path, model_name, image_size, batch_size, epochs, learning_rate):
    # Construct the filename with hyperparameters
    filename = f"{model_name}_img{image_size[0]}x{image_size[1]}_batch{batch_size}_epochs{epochs}_lr{learning_rate:.5f}.pth"
    
    # Combine the base path and the filename
    save_path = os.path.join(path, filename)
    
    # Save the model state dictionary
    torch.save(model.state_dict(), save_path)
    
    print(f"Model saved to: {save_path}")

# def load_model (path):
#     model = get_model("VGG16", 4, True)  # Update this if you have different model or parameters
#     model.load_state_dict(torch.load(path))
#     return model

def load_model(model_name, image_size, batch_size, epochs, learning_rate):
    # Construct the filename from parameters
    filename = f"{model_name}_img{image_size[0]}x{image_size[1]}_batch{batch_size}_epochs{epochs}_lr{learning_rate:.5f}.pth"
    model_path = os.path.join(save_path, filename)
    
    # Ensure that the model file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"The model file at {model_path} does not exist.")
    
    model = get_model(model_name, num_classes, weights=None)  # Adjust parameters as needed
     # Load the state dictionary
    state_dict = torch.load(model_path)
    
    # If model was saved using DataParallel, adjust the keys
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' from the start of each key
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)    
    return model



def get_class_weights(labels):
    # Convert labels list to a numpy array
    labels = np.array(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return torch.tensor(class_weights, dtype=torch.float)



def load_label_names():
    with open(os.path.join(save_path, 'label_mapping.json'), 'r') as f:
        label_mapping = json.load(f)
    return label_mapping['labels']


# # main scipt
# 

# In[10]:


import yaml
# from src.train import train_model
# from src.evaluate import evaluate_model

if __name__ == "__main__":
    #with open('/home/boukhari/projects/dental_image_project/configs/config.yaml', "r") as file:
        #config = yaml.safe_load(file)
    
    train_model()
    evaluate_model()


# In[ ]:


evaluate_model()
