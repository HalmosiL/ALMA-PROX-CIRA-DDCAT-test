from network import PSPNet, Dummy

import torch
import torch.nn as nn

def get_new_model(device):
    model = PSPNet(
        layers=50,
        bins=(1, 2, 3, 6),
        dropout=0.1,
        classes=19,
        zoom_factor=8,
        use_ppm=True,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        BatchNorm=nn.BatchNorm2d,
        pretrained=False
    )

    model = model.to(device)
    return model

def load_model(path, device):
    model = PSPNet(
        layers=50,
        bins=(1, 2, 3, 6),
        dropout=0.1,
        classes=19,
        zoom_factor=8,
        use_ppm=True,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        BatchNorm=nn.BatchNorm2d,
        pretrained=False
    )

    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device).eval()
    return model
