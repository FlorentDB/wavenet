import torch
import torch.nn as nn
import torch.nn.functional as F

class Stacked_Conv(nn.Module):
    def __init__(self, nb_stack = 3, nb_layers = 10):
        super(Stacked_Conv, self).__init__()
        
        dilations = [2**i for i in range(nb_layers)] * nb_stack

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=2,
                dilation=d,
                padding=0
            )
            for d in dilations
        ])

    def forward(self, x):
        for conv in self.convs:
            pad_left = (conv.kernel_size[0] - 1) * conv.dilation[0]
            x = F.pad(x, (pad_left,0))
            x= conv(x)
        return x


