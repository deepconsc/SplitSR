import torch
from torch import nn

class SplitSRBlock(nn.Module):
    def __init__(self, channels, kernel, alpha):
        super(SplitSRBlock, self).__init__()
        self.alpha = alpha
        self.channels = channels
        self.conv = nn.Conv2d(in_channels=int(self.channels*self.alpha), out_channels=int(self.channels*self.alpha), kernel_size=kernel, stride=1, padding=kernel//2)
        self.batchnorm = nn.BatchNorm2d(int(self.channels*self.alpha))
        self.relu = nn.ReLU(inplace=True)

    

    def forward(self, x):
        active, passive = x[:, :int(self.channels*self.alpha)], x[:, int(self.channels*self.alpha):]
        active = self.conv(active)
        active = self.batchnorm(active)
        active = self.relu(active)
        x = torch.cat([passive, active], dim=1)
        return x

if __name__ == '__main__':
    block_alphas = [0.125, 0.256, 0.500, 1.000]

    for alpha in block_alphas:
        block = SplitSRBlock(128, 3, alpha)

        x = torch.randn(1, 128, 112, 112)
        y = block(x)

        assert y.shape == x.shape

        params = sum(p.numel() for p in block.parameters() if p.requires_grad)
        print(f'Total params for block with α={alpha}: {params}')