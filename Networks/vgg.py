import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, cfg, in_channels):
        super(VGG, self).__init__()
        self.cfg, self.in_channels = cfg, in_channels
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
                nn.Linear(cfg[-1], 1))
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def make_layers(self, negative_slope=0.2):
        layers: List[nn.Module] = []
        in_channels = self.in_channels
        for v in self.cfg:
            if v == 'M':
                layers += [ nn.MaxPool2d(kernel_size=2) ]
            else:
                layers += [ nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                            nn.LeakyReLU(negative_slope=negative_slope) ]
                in_channels = v
        return nn.Sequential(*layers)
