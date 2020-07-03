from torch import nn
from torchvision import models


def load_model(model, out_features):
    net = None
    if model == 'VGG16':
        net = models.vgg16(pretrained=True)
        net.classifier[6] = nn.Linear(in_features=4096, out_features=out_features)
        for name, param in net.named_parameters():
            if name in ['classifier.6.weight', 'classifier.6.bias']:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif model == 'ResNeXt':
        net = models.resnext101_32x8d(pretrained=True)
        net.fc = nn.Linear(in_features=2048, out_features=out_features)
        for name, param in net.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    elif model == 'DenseNet':
        net = models.densenet161(pretrained=True)
        net.classifier = nn.Linear(in_features=2208, out_features=out_features)
        for name, param in net.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    return net
