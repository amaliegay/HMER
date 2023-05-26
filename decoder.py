from torch import nn

class Decoder(nn.Module):
    def __init__(self, growth_rate:int):
        super().__init__()
        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels,kernel_size=7,padding=3,stride=2,bias=False)

    def forward(self, x):
        out = self.conv1(x)