import torch
import torch.nn as nn
import torchvision.models as models

class TwoStreamCNN(nn.Module):
    def __init__(self, type='tsma'):
        super().__init__()
        self.conv = nn.LazyConv2d(64,kernel_size = (3,3))
        self.type = type
        self.blockend = self.block()  # Use ResNet-50 as the block

    def forward(self, streamA, streamB):
        ht = nn.LeakyReLU()(self.conv(streamA))
        ht1 = nn.LeakyReLU()(self.conv(streamB))
        z = torch.add(ht, ht1)

        if self.type == 'tsma':
            y = torch.cat((z, ht), dim=1)
        elif self.type == 'tsmb':
            y = torch.cat((z, ht1), dim=1)
        elif self.type == 'tsmab':
            y = torch.cat((z, ht/2, ht1/2), dim=1)

        yhat = self.blockend(y)
        return yhat
    
    def block(self):
        resnet50 = models.resnet50(pretrained=True)
        for param in resnet50.parameters():
            param.requires_grad = False
        fc_inputs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(512,29)
        return resnet50
