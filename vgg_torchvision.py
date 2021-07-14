import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, cfg, in_channels, dropout):
        super(VGG, self).__init__()
        self.cfg, self.in_channels = cfg, in_channels
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
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

# class SiameseVGG(nn.Module):
#     def __init__(self, max_lag: int,
#                  primary_features: nn.Module,
#                  neighbor_features: nn.Module,
#                  dropout: float = 0.5,
#                  num_classes: int = 1
#                  ) -> None:
#         super(SiameseVGG, self).__init__()
#         self.max_lag = max_lag
#         self.primary_features = primary_features
#         self.neighbor_features = neighbor_features
#         self.maxpool = nn.AdaptiveMaxPool2d(1)
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=dropout),
#             nn.Linear(512*2, num_classes),
#         )
#         self._initialize_weights()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x1, x2 = torch.tensor_split(x, [self.max_lag+1], axis=1)
#         x1 = self.primary_features(x1)
#         x2 = self.neighbor_features(x2)
#         x = torch.cat([x1,x2], axis=1)
#         x = self.maxpool(x)
#         x = torch.flatten(x, 1)
#         return self.classifier(x)
#
#     def _initialize_weights(self) -> None:
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight)

# def siamese_vgg(max_lag: int, context_dims: int, dropout: float, **kwargs: Any) -> SiameseVGG:
#     in_channels_primary, in_channels_neighbor = 1+max_lag, (2+2*context_dims)*(1+max_lag)
#     primary_features = make_layers(cfgs['F'], in_channels_primary)
#     neighbor_features = make_layers(cfgs['F'], in_channels_neighbor)
#     return SiameseVGG(max_lag, primary_features, neighbor_features, dropout, **kwargs)
