import torch, torch.nn as nn

class VGG_CNNC(nn.Module):
    def __init__(self, cfg, dropout, in_channels=1):
        super(VGG_CNNC, self).__init__()
        self.cfg = cfg
        self.features = self.make_layers(in_channels)
        self.classifier = nn.Sequential(
                nn.Linear(128*4*4, 512),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(512, 1))
        self._initialize_weights()

    def forward(self, x):
        if x.size(1)!=1:
            x = torch.unsqueeze(x[:,0,:,:], 1)
        out = self.features(x)
        out = torch.flatten(out, 1)
        return self.classifier(out)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def make_layers(self, in_channels):
        layers: List[nn.Module] = []
        for v in self.cfg:
            if v == 'M':
                layers += [ nn.MaxPool2d(kernel_size=2) ]
            else:
                layers += [ nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU() ]
                in_channels = v
        return nn.Sequential(*layers)
