import torch
import torch.nn as nn
from torchvision import models

def get_model(architecture, weights=None):
    if architecture == 'mnasnet1_0':
        # Initialize the MNASNet 1.0 model from torchvision models
        model = models.mnasnet1_0(weights=None)
        
        # Change the first layer of the model to have 1 input channel
        # This is done because our input images are grayscale, so they only have one channel
        model.layers[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        # Change the last layer of the model to output a single value
        # This is done because we are performing regression, so our model should output a single continuous value
        model.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=True)
    elif architecture == 'resnet152':
        model = models.resnet152(weights=None)

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model = model
    else:
        raise ValueError(f"Architecture '{architecture}' not supported.")
    
    return model
