
import torch
from torch import nn
from torchvision import models


def build_model():
    # Fetch pretrained model
    model = models.resnet34(pretrained=True)

    # Freeze model weights - For faster training
    for param in model.parameters():
        param.requires_grad = False

    #Edit final layer for our classification problem
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
                    nn.Linear(n_inputs, 256), 
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Linear(64, 1),
                    nn.Sigmoid())

    return model

def load_model(model_name : str):
    useGPU = torch.cuda.is_available()
    model = torch.load(model_name, map_location=torch.device(0 if useGPU else "cpu"))
    return model