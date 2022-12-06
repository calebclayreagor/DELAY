import torch
import torch.nn as nn

class SiameseVGG(nn.Module):
    def __init__(self, cfg, neighbors):
        super(SiameseVGG, self).__init__()
        self.primary_features = self.make_layers(cfg)
        self.neighbor_features = self.make_layers(cfg)
        self.primary_embedding = nn.Sequential(
            nn.Linear(4*4*128, 512),
            nn.ReLU())
        self.neighbor_embedding = nn.Sequential(
            nn.Linear(4*4*128, 512),
            nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(512*(3+2*neighbors), 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        self._initialize_weights()

    def forward_primary(self, x):
        out = torch.unsqueeze(x, 1)
        out = self.primary_features(out)
        out = torch.flatten(out, 1)
        return self.primary_embedding(out)

    def forward_neighbor(self, x):
        out = torch.unsqueeze(x, 1)
        out = self.neighbor_features(out)
        out = torch.flatten(out, 1)
        return self.neighbor_embedding(out)

    def forward(self, x):
        primary, neighbors = x[:,0,:,:], x[:,1:,:,:]
        out = self.forward_primary(primary)
        for idx in range(neighbors.size(1)):
            out_ = self.forward_neighbor(neighbors[:,idx,:,:])
            out = torch.cat([out, out_], axis=1)
        return self.classifier(out)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def make_layers(self, cfg, in_channels=1, dropout=0.25):
        layers: List[nn.Module] = []
        for v in cfg:
            if v == 'M':
                layers += [ nn.MaxPool2d(kernel_size=2) ]
            elif v == 'D':
                layers += [ nn.Dropout(p=dropout) ]
            else:
                layers += [ nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU() ]
                in_channels = v
        return nn.Sequential(*layers)
