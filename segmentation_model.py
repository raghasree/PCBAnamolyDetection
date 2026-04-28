import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    Placeholder UNet — returns blank mask until properly trained.
    Replace with full UNet when segmentation training is done.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return torch.zeros(x.size(0), 1,
                           x.size(2), x.size(3),
                           device=x.device)