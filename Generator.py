import torch

class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(64 * 8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 4),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )
        

    def forward(self, im):
        return self.network(input)