import torch
import torch.nn as nn

def init_weights(model):
    """
    Initialize weights for all layers in the given model.
    Applies Xavier initialization for linear layers, He initialization for conv and transposed conv layers,
    and normal initialization for batch normilerden.

    Args:
        model (nn.Module): The neural network model to initialize weights for.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # Xavier initialization for Linear layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # He initialization for Conv2d and ConvTranspose2d layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            # Normal initialization for BatchNorm layers
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            # Initialize RNN weights
            for name, param in m.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)