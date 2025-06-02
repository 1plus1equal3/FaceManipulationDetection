import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            # nn.Linear(in_features=256, out_features = 64),
            # nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(in_features=128, out_features=num_classes)
        )
    def forward(self, x):
        x = self.classifier(x)
        return x