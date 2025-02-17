import torch.nn as nn
import torchvision.models as models

def get_model(model_name, num_classes, weights=None):
    if model_name == "VGG16":
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
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
    