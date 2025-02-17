import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.data_loading import load_data
from src.utils import load_model, load_label_names
from src.metrics import calculate_metrics

def evaluate_model(config):
    _, _, test_loader = load_data(
        config['label_file'], config['image_type'], config['image_size'], config['image_dir'],
        config['test_split'], config['validation_split'], config['batch_size'], config['save_path']
    )
    model = load_model(
        config['model_name'], config['image_size'], config['batch_size'], config['epochs'], config['learning_rate']
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_metrics = calculate_metrics(model, test_loader, device)
    print(f"Test Metrics: {test_metrics}")

    label_mapping = load_label_names(config['save_path'])
    num_classes = len(label_mapping)
    inv_label_mapping = {int(k): v for k, v in label_mapping.items()}

    all_labels, all_preds = [], []
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
    plt.savefig(config['save_path'] + "_confusion_matrix.png")
    plt.show()