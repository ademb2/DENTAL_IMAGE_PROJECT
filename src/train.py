import torch
import os
import matplotlib.pyplot as plt
import torch.optim as optim
from src.data_loading import load_data
from src.model import get_model
from src.utils import save_model, get_class_weights
from src.metrics import calculate_metrics

def train_model(config):
    train_loader, val_loader, _ = load_data(
        config['label_file'], config['image_type'], config['image_size'], config['image_dir'],
        config['test_split'], config['validation_split'], config['batch_size'], config['save_path']
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config['model_name'], config['num_classes'], config['weights'])

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    train_labels = [label for _, label in train_loader.dataset]
    class_weights = get_class_weights(train_labels)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    model.to(device)

    best_val_loss = float('inf')
    train_losses, val_losses, accuracies = [], [], []

    model.train()
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {running_loss/len(train_loader)}")
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_metrics = calculate_metrics(model, val_loader, device, criterion)
        print(f"Validation Metrics: {val_metrics}")
        epoch_val_loss = val_metrics.get('loss', float('inf'))
        val_losses.append(epoch_val_loss)
        accuracies.append(val_metrics['accuracy'])

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss / len(train_loader),
            }
            torch.save(
                checkpoint,
                os.path.join(config["save_path"], f"epoch{epoch + 1}_best.pth"))
            print(f"Checkpoint saved for epoch {epoch + 1} with validation loss: {epoch_val_loss:.4f}")
        else:
            print(f"No improvement for epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")

    save_model(model, config['save_path'], config['model_name'], config['image_size'], config['batch_size'], config['epochs'], config['learning_rate'])

    plt.figure(figsize=(12, 5))
    epochs_range = range(1, len(train_losses) + 1)
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
    plt.savefig(os.path.join(config["save_path"], "training_metrics.png"))
    plt.show()