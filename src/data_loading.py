import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_dir, labels_df, base_transform=None, augment_transform=None):
        self.image_dir = image_dir
        self.base_transform = base_transform
        self.augment_transform = augment_transform

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
        image = np.array(image)

        if aug_flag and self.augment_transform:
            image = self.augment_transform(image=image)['image']
        elif self.base_transform:
            image = self.base_transform(image=image)['image']

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
            A.RandomBrightnessContrast(p=1.0, brightness_limit=(0.5), contrast_limit=(0)),
            A.RandomBrightnessContrast(p=1.0, brightness_limit=(0), contrast_limit=(0.5)),
            A.RandomGamma(p=1.0, gamma_limit=(30, 150))], p=0.5),
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

def load_labels(label_file, image_type, save_path):
    labels = pd.read_csv(label_file)
    if 'image_type' not in labels.columns or 'category' not in labels.columns:
        raise KeyError("Label file must contain 'image_type' and 'category' columns.")
    labels = labels[labels['image_type'] == image_type]
    labels['category'] = labels['category'].replace('Environment_cohort', 'Control_cohort')
    labels['label'] = labels['category'].astype('category').cat.codes
    category_mapping = labels['category'].astype('category').cat.categories
    label_mapping = {str(code): name for code, name in enumerate(category_mapping)}

    # Check if directory exists and create it if it doesnt
    os.makedirs(save_path, exist_ok=True)
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
    train_idx, temp_idx = train_test_split(
        indices, test_size=test_split, stratify=labels['label'], random_state=42)

    test_idx, val_idx = train_test_split(
        temp_idx, test_size=validation_split, stratify=labels.iloc[temp_idx]['label'], random_state=42)

    train_dataset = ImageDataset(image_dir, labels.iloc[train_idx], base_transform=augment_transform)
    val_dataset = ImageDataset(image_dir, labels.iloc[val_idx], base_transform=base_transform)
    test_dataset = ImageDataset(image_dir, labels.iloc[test_idx], base_transform=base_transform)

    return train_dataset, val_dataset, test_dataset

def load_data(label_file, image_type, image_size, image_dir, test_split, validation_split, batch_size, save_path):
    labels = load_labels(label_file, image_type, save_path)
    base_transform, augment_transform = create_transforms(image_size)
    train_dataset, val_dataset, test_dataset = create_datasets(
        image_dir, labels, base_transform, augment_transform, test_split, validation_split)
    return create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)