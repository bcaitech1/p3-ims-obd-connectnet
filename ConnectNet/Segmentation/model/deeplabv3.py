import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import vgg16
import torch
from torch.nn import functional as F

def make_layers():
    vgg16 = models.vgg16(pretrained=True)
    features = list(vgg16.features.children())
    
    conv1 = nn.Sequential(*features[0:5])
    conv1[-1] = nn.MaxPool2d(3, stride=2, padding=1)
    
    conv2 = nn.Sequential(*features[5:10])
    conv2[-1] = nn.MaxPool2d(3, stride=2, padding=1)
    
    conv3 = nn.Sequential(*features[10:17])
    conv2[-1] = nn.MaxPool2d(3, stride=2, padding=1)
    
    conv4 = nn.Sequential(*features[17:24])
    conv4[-1] = nn.MaxPool2d(3, stride=1, padding=1)

    conv5 = nn.Sequential(*features[24:31])
    conv5[-1] = nn.MaxPool2d(3, stride=1, padding=1)

    for i in range(len(conv5)):
        if isinstance(conv5[i], nn.Conv2d):
            conv5[i].padding = (2, 2)
            conv5[i].dilation = (2, 2)
    
    
    return nn.Sequential(conv1,conv2,conv3,conv4,conv5) 


class ASPPConv(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, padding, dilation):
        super(ASPPConv, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        output = self.relu(x)
        return output

class ASPPPooling(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPPPooling, self).__init__()
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.globalavgpool(x)
        x = self.conv(x)
        x = self.bn(x)
        output = self.relu(x)
        return output


class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = ASPPConv(inplanes, outplanes, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPConv(inplanes, outplanes, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPConv(inplanes, outplanes, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPConv(inplanes, outplanes, 3, padding=dilations[3], dilation=dilations[3])
        self.global_avg_pool = ASPPPooling(inplanes, outplanes)
        self.project = nn.Sequential(
            nn.Conv2d(outplanes*5, outplanes, 1, bias=False), 
            nn.BatchNorm2d(outplanes), 
            nn.ReLU(), 
            nn.Dropout(0.5)      
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        output = self.project(x)
        return output

class DeepLabHead(nn.Sequential):
    def __init__(self, in_ch, out_ch, num_classes):
        super(DeepLabHead, self).__init__()
        self.add_module("0", ASPP(in_ch, out_ch))
        self.add_module("1", nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1 , bias=False))
        self.add_module("2", nn.BatchNorm2d(out_ch))
        self.add_module("3", nn.ReLU())
        self.add_module("4", nn.Conv2d(out_ch, num_classes, kernel_size=1, stride=1))

class DeepLabV3(nn.Sequential):
    def __init__(self, num_classes):
        super(DeepLabV3, self).__init__()
        self.backbone = make_layers()
        self.classifier = DeepLabHead(in_ch=512, out_ch=256, num_classes=12)

    def forward(self, x): 
        h = self.backbone(x)
        h = self.classifier(h)
        output = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)
        return output


if __name__ =="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DeepLabV3(num_classes=12)
    x = torch.randn([2, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x).to(device)
    print("output shape : ", out.size())

    model = model.to(device)