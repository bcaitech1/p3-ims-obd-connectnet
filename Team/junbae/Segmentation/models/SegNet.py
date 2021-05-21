import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg16
import segmentation_models_pytorch as smp
from utils import seed_everything

seed_everything(21)


class SegNet(nn.Module):
    def __init__(self, num_classes=12, init_weights=True):
        super(SegNet, self).__init__()

        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            # convolution
            # relu
            # 한블럭
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True))
        # conv1
        self.covn1_1 = CBR(3, 64, 3, 1, 1)
        self.covn1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(
            2, stride=2, ceil_mode=True, return_indices=True)

        # conv2
        self.covn2_1 = CBR(64, 128, 3, 1, 1)
        self.covn2_2 = CBR(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(
            2, stride=2, ceil_mode=True, return_indices=True)

        # conv3
        self.covn3_1 = CBR(128, 256, 3, 1, 1)
        self.covn3_2 = CBR(256, 256, 3, 1, 1)
        self.covn3_3 = CBR(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(
            2, stride=2, ceil_mode=True, return_indices=True)

        # conv4
        self.covn4_1 = CBR(256, 512, 3, 1, 1)
        self.covn4_2 = CBR(512, 512, 3, 1, 1)
        self.covn4_3 = CBR(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(
            2, stride=2, ceil_mode=True, return_indices=True)

        # conv5
        self.covn5_1 = CBR(512, 512, 3, 1, 1)
        self.covn5_2 = CBR(512, 512, 3, 1, 1)
        self.covn5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(
            2, stride=2, ceil_mode=True, return_indices=True)

        # deconv5
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.dcovn5_3 = CBR(512, 512, 3, 1, 1)
        self.dcovn5_2 = CBR(512, 512, 3, 1, 1)
        self.dcovn5_1 = CBR(512, 512, 3, 1, 1)

        # deconv4
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.dcovn4_3 = CBR(512, 512, 3, 1, 1)
        self.dcovn4_2 = CBR(512, 512, 3, 1, 1)
        self.dcovn4_1 = CBR(512, 256, 3, 1, 1)

        # deconv3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.dcovn3_3 = CBR(256, 256, 3, 1, 1)
        self.dcovn3_2 = CBR(256, 256, 3, 1, 1)
        self.dcovn3_1 = CBR(256, 128, 3, 1, 1)

        # deconv2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.dcovn2_2 = CBR(128, 128, 3, 1, 1)
        self.dcovn2_1 = CBR(128, 64, 3, 1, 1)

        # deconv1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.dcovn1_1 = CBR(64, 64, 3, 1, 1)

        # score
        self.score = nn.Conv2d(64, num_classes, 3, 1, 1, 1)

    def forward(self, x):

        # conv1
        dim1 = x.size()
        x = self.covn1_1(x)
        x = self.covn1_2(x)
        x, indic1 = self.pool1(x)

        # conv2
        dim2 = x.size()
        x = self.covn2_1(x)
        x = self.covn2_2(x)
        x, indic2 = self.pool2(x)

        # conv3
        dim3 = x.size()
        x = self.covn3_1(x)
        x = self.covn3_2(x)
        x = self.covn3_3(x)
        x, indic3 = self.pool3(x)

        # conv4
        dim4 = x.size()
        x = self.covn4_1(x)
        x = self.covn4_2(x)
        x = self.covn4_3(x)
        x, indic4 = self.pool4(x)

        # conv5
        dim5 = x.size()
        x = self.covn5_1(x)
        x = self.covn5_2(x)
        x = self.covn5_3(x)
        x, indic5 = self.pool5(x)

        # deconv5
        x = self.unpool5(x, indic5)
        x = self.dcovn5_3(x)
        x = self.dcovn5_2(x)
        x = self.dcovn5_1(x)

        # deconv4
        x = self.unpool4(x, indic4)
        x = self.dcovn4_3(x)
        x = self.dcovn4_2(x)
        x = self.dcovn4_1(x)

        # deconv3
        x = self.unpool3(x, indic3)
        x = self.dcovn3_3(x)
        x = self.dcovn3_2(x)
        x = self.dcovn3_1(x)

        # deconv2
        x = self.unpool2(x, indic2)
        x = self.dcovn2_2(x)
        x = self.dcovn2_1(x)

        # deconv1
        x = self.unpool1(x, indic1)
        x = self.dcovn1_1(x)

        # score
        h = self.score(x)

        return h

# deep v3
