import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(model, dataloader, device, criterion=None):
    model.eval()
    all_labels, all_preds = [], []
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