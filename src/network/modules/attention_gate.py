import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__()
        
        # attention gate
        self.attn_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        
    def forward(self, main_branch, sub_branch):
        # combine 2 branch
        combine = torch.cat((main_branch, sub_branch), 1)
        attn = self.attn_gate(combine)
        
        # attention on main branch
        return main_branch * attn