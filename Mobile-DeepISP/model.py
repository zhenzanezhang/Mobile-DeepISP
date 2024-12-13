import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetISP(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetISP, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.features = mobilenet.features
        in_features = mobilenet.last_channel

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_features, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feats = self.features(x)
        out = self.decoder(feats)
        return out
