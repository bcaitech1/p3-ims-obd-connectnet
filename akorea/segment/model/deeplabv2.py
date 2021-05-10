import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import vgg16
import torch
from torch.nn import functional as F

def make_layers():
    vgg16 = models.vgg16(pretrained=True)
    features = list(vgg16.features.children())
    
    conv1 = nn.Sequential(*features[:4])
    conv2 = nn.Sequential(*features[5:9])
    conv3 = nn.Sequential(*features[10:16])
    conv4 = nn.Sequential(*features[17:23])
    conv5 = nn.Sequential(*features[24:30])
    
    for i in range(len(conv5)):
        if isinstance(conv5[i], nn.Conv2d):
            conv5[i].padding = (2, 2)
            conv5[i].dilation = (2, 2)

    return [conv1, conv2, conv3, conv4, conv5]

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=1024, num_classes=21):
        super(ASPP, self).__init__()
        # atrous 3x3, rate=6
        self.conv_3x3_r6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        # atrous 3x3, rate=12
        self.conv_3x3_r12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        # atrous 3x3, rate=18
        self.conv_3x3_r18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        # atrous 3x3, rate=24
        self.conv_3x3_r24 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=24, dilation=24)
        self.drop_conv_3x3 = nn.Dropout2d(0.5)

        self.conv_1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.drop_conv_1x1 = nn.Dropout2d(0.5)

        self.conv_1x1_out = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # 1번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r6 = self.drop_conv_3x3(F.relu(self.conv_3x3_r6(feature_map)))
        out_img_r6 = self.drop_conv_1x1(F.relu(self.conv_1x1(out_3x3_r6)))
        out_img_r6 = self.conv_1x1_out(out_img_r6)
        # 2번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r12 = self.drop_conv_3x3(F.relu(self.conv_3x3_r12(feature_map)))
        out_img_r12 = self.drop_conv_1x1(F.relu(self.conv_1x1(out_3x3_r12)))
        out_img_r12 = self.conv_1x1_out(out_img_r12)
        # 3번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r18 = self.drop_conv_3x3(F.relu(self.conv_3x3_r18(feature_map)))
        out_img_r18 = self.drop_conv_1x1(F.relu(self.conv_1x1(out_3x3_r18)))
        out_img_r18 = self.conv_1x1_out(out_img_r18)
        # 4번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r24 = self.drop_conv_3x3(F.relu(self.conv_3x3_r24(feature_map)))
        out_img_r24 = self.drop_conv_1x1(F.relu(self.conv_1x1(out_3x3_r24)))
        out_img_r24 = self.conv_1x1_out(out_img_r24)

        out = sum([out_img_r6, out_img_r12, out_img_r18, out_img_r24])

        return out
        

class DeepLabV2(nn.Module):
    def __init__(self, num_classes, upsampling=8):
        super(DeepLabV2,self).__init__()
        layers = make_layers()
        
        self.conv1 = layers[0]
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = layers[1]
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv3 = layers[2]
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv4 = layers[3]
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.conv5 = layers[4]
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.fc =  ASPP(in_channels=512, out_channels=256, num_classes=num_classes)
        
        self.upsampling = upsampling

        

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = self.pool5(x)

        
        _, _, feature_map_h, feature_map_w = x.size()

        x = self.fc(x)
        
        out = F.interpolate(x, size=(feature_map_h * self.upsampling, feature_map_w * self.upsampling), mode="bilinear", align_corners=False)

        return out



if __name__ =="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DeepLabV2(num_classes=12)
    x = torch.randn([1, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x).to(device)
    print("output shape : ", out.size())

    model = model.to(device)