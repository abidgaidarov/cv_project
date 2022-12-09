
from torch import nn

class ConvAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        # encoder
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(64))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(64))
        self.pool = nn.MaxPool2d(2, return_indices=True)
        
        # decoder
        
        self.unpool = nn.MaxUnpool2d(2)
        
        self.conv1_t = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3),
                                     nn.LeakyReLU(),
                                     nn.BatchNorm2d(64))
        self.conv2_t = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3),
                                     nn.LeakyReLU(),
                                     nn.BatchNorm2d(32))
        self.conv3_t = nn.Sequential(nn.ConvTranspose2d(32, 1, kernel_size=3),
                                     nn.LeakyReLU(),
                                     nn.BatchNorm2d(1))

        
    def encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
            
        x, indices = self.pool(x)
        return x, indices
        
        
    def decoder(self, x, indices):
        x = self.unpool(x, indices)
        x = self.conv1_t(x)
        x = self.conv2_t(x)
        x = self.conv3_t(x)
        return x
        
        
    def forward(self, x):
        latent, indices = self.encoder(x)
        result = self.decoder(latent, indices)
        return result