import torch
import torch.nn as nn
import torchvision.models as models

class parsingNet(nn.Module):
    def __init__(self, pretrained=True, backbone='18', cls_dim=(101, 56, 4), use_aux=False):
        super(parsingNet, self).__init__()
        
        self.cls_dim = cls_dim
        self.use_aux = use_aux
        
        # Get the backbone
        if backbone == '18':
            self.model = models.resnet18(pretrained=pretrained)
        elif backbone == '34':
            self.model = models.resnet34(pretrained=pretrained)
        elif backbone == '50':
            self.model = models.resnet50(pretrained=pretrained)
        elif backbone == '101':
            self.model = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError('Backbone not supported')
        
        if backbone in ['18', '34']:
            self.pool_channels = 512
        else:
            self.pool_channels = 2048
            
        # Define the classifier layers
        self.cls = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.pool_channels, 8, 1),
                nn.ReLU(),
                nn.Conv2d(8, self.cls_dim[0], 1)
            ),
            nn.Sequential(
                nn.Conv2d(self.pool_channels, 8, 1),
                nn.ReLU(),
                nn.Conv2d(8, self.cls_dim[0], 1)
            ),
            nn.Sequential(
                nn.Conv2d(self.pool_channels, 8, 1),
                nn.ReLU(),
                nn.Conv2d(8, self.cls_dim[0], 1)
            ),
            nn.Sequential(
                nn.Conv2d(self.pool_channels, 8, 1),
                nn.ReLU(),
                nn.Conv2d(8, self.cls_dim[0], 1)
            )
        ])
    
    def forward(self, x):
        # Get features from backbone
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        feat = self.model.layer4(x)
        
        # Process through classifiers
        out = []
        for classifier in self.cls:
            out.append(classifier(feat))
        
        # Stack outputs
        out = torch.stack(out, dim=3)  # (B, cls_dim[0], H, W, 4)
        return {'out': out}
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0) 