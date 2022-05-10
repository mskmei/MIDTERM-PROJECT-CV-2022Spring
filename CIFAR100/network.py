def network_paddle(x, pretrained = True, num_classes = 100):
    import paddle
    x = x.lower()
    if x == 'resnet18':
        return paddle.vision.models.resnet18(num_classes=num_classes,pretrained=pretrained)
    elif x == 'resnet34':
        return paddle.vision.models.resnet34(num_classes=num_classes,pretrained=pretrained)
    elif x == 'resnet50':
        return paddle.vision.models.resnet50(num_classes=num_classes,pretrained=pretrained)
    elif x == 'resnet101':
        return paddle.vision.models.resnet101(num_classes=num_classes,pretrained=pretrained)
    
    raise ValueError('Network structure %s not supported or found.'%(x))


def network_torch(x, pretrained = True, num_classes = 100, cuda = 'cpu'):
    import torchvision
    import torch
    x = x.lower()
    backbone = None
    if x == 'resnet18':
        backbone = torchvision.models.resnet18(pretrained=pretrained)
    elif x == 'resnet34':
        backbone = torchvision.models.resnet34(pretrained=pretrained)
    elif x == 'resnet50':
        backbone = torchvision.models.resnet50(pretrained=pretrained)
    elif x == 'resnet101':
        backbone = torchvision.models.resnet101(pretrained=pretrained)
    
    if backbone is None:
        raise ValueError('Network structure %s not supported or found.'%(x))

    class Net(torch.nn.Module):
        def __init__(self, backbone, num_classes = 100):
            super().__init__()
            self.head = backbone 
            self.fc = torch.nn.Linear(1000, num_classes)
        
        def forward(self, x):
            return self.fc( self.head(x) )
    

    net = Net(backbone, num_classes=num_classes)
    net.to(cuda)
    return net


if __name__ == '__main__':
    print(network_torch('ResnET18'))