import torch
from torch import nn

class SplitSRBlock(nn.Module):
    def __init__(self, channels, kernel, alpha):
        super(SplitSRBlock, self).__init__()
        self.alpharatio = int(channels * alpha)
        self.channels = channels
        self.conv = nn.Conv2d(in_channels=self.alpharatio, out_channels=self.alpharatio, kernel_size=kernel, stride=1, padding=kernel//2, bias=True)
        self.batchnorm = nn.BatchNorm2d(self.alpharatio)
        self.relu = nn.ReLU(inplace=True)

    

    def forward(self, x):
        active, passive = x[:, :self.alpharatio], x[:, self.alpharatio:]
        active = self.conv(active) # In: (1, 64 * α, W, H) | Out: (1, 64 * α, W, H)
        active = self.batchnorm(active)
        active = self.relu(active)
        x = torch.cat([passive, active], dim=1) # Out: (1, 64, W, H)
        return x

class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.channels = channels
        self.conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels*4, kernel_size=3, stride=1,  padding=3//2, bias=True)
        self.pixelshuffle = nn.PixelShuffle(2)


    def forward(self, x):
        x = self.conv(x)  # In: (1, 64, W, H) | Out: (1, 256, W, H)
        x = self.pixelshuffle(x) # In: (1, 256, W*2, H*2) | Out: (1, 64, W*2, H*2)

        x = self.conv(x) # In: (1, 64, W*2, H*2) | Out: (1, 256, W*2, H*2)
        x = self.pixelshuffle(x) # In: (1, 256, W*4, H*4) | Out: (1, 64, W*4, H*4)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1,  padding=3//2, bias=True)
        self.batchnorm = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        residual = x
        x = self.conv(x)  
        x = self.batchnorm(x) 
        x = self.relu(x) 

        x = self.conv(x)  
        x = self.batchnorm(x) 
        x += residual
        x = self.relu(x) 

        return x


class MeanShift(nn.Conv2d):
    def __init__(self, coeff):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor([1.0, 1.0, 1.0])
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = coeff * 255 * torch.Tensor([0.4488, 0.4371, 0.4040])
        self.bias.data.div_(std)
        self.requires_grad = False

if __name__ == '__main__':
    block_alphas = [0.125, 0.256, 0.500, 1.000]

    for alpha in block_alphas:
        block = SplitSRBlock(128, 3, alpha)

        x = torch.randn(1, 128, 112, 112)
        y = block(x)

        assert y.shape == x.shape

        params = sum(p.numel() for p in block.parameters() if p.requires_grad)
        print(f'Total params for block with α={alpha}: {params}')