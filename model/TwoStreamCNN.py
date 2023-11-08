import torch
import torch.nn as nn
import torchvision.models as models

class TwoStreamCNN(nn.Module):
    def __init__(self, type='tsma'):
        super().__init()
        self.conv = nn.Conv2d(226, 64, (3, 3))
        self.type = type
        self.block = models.resnet50(pretrained=True)  # Use ResNet-50 as the block

    def forward(self, streamA, streamB):
        ht = nn.LeakyReLU(self.conv(streamA))
        ht1 = nn.LeakyReLU(self.conv(streamB))
        z = torch.add(ht, ht1)

        if self.type == 'tsma':
            y = torch.cat((z, ht), dim=1)
        elif self.type == 'tsmb':
            y = torch.cat((z, ht1), dim=1)
        elif self.type == 'tsmab':
            y = torch.cat((z, ht/2, ht1/2), dim=1)

        yhat = self.block(y)
        return yhat
