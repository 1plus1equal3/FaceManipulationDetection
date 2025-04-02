import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.backbone_u2_net import U2NET
from src.data.dataloader import get_dataloader
from src.utils.util import *
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = f.read()
    return config

config = load_config('src/config.json')

def training_loop(model, epochs, train_loader, lr=config['model']['lr'], beta=(0.9, 0.999), device=device):
    model = model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=beta)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        for _, _, fake_image, masked_fake_image in train_loader:
            fake_image = fake_image.to(device)
            masked_fake_image = masked_fake_image.to(device)
            
            optimizer.zero_grad()
            output, _, _, _, _, _, _ = model(fake_image)

            # print(output.size())
            # print(masked_fake_image.size())
            
            loss = loss_fn(output, masked_fake_image)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            visualize_results(masked_fake_image, output)
            print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
            torch.save(model.state_dict(), f'checkpoints/u2_net.pt')
            
def main():
    model = U2NET()
    train_loader = get_dataloader(mode='train')

    
    training_loop(
        model=model,
        epochs=config['model']['epochs'],
        train_loader=train_loader
    )
    
if __name__ == '__main__':
    main()