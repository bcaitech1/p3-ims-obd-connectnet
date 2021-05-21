import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg16
import segmentation_models_pytorch as smp
from utils import seed_everything

seed_everything(21)


class FCN16s(nn.Module):
    def __init__(self, num_classes):
        super(FCN16s, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding),
                                 nn.ReLU(inplace=True)
                                 )

        # conv1
        self.conv1_1 = CBR(3, 64, 3, 1, 1)
        self.conv1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv2
        self.conv2_1 = CBR(64, 128, 3, 1, 1)
        self.conv2_2 = CBR(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv3
        self.conv3_1 = CBR(128, 256, 3, 1, 1)
        self.conv3_2 = CBR(256, 256, 3, 1, 1)
        self.conv3_3 = CBR(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv4
        self.conv4_1 = CBR(256, 512, 3, 1, 1)
        self.conv4_2 = CBR(512, 512, 3, 1, 1)
        self.conv4_3 = CBR(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Score pool4 - 중간에 나오는 skip부분
        self.score_pool4_fr = nn.Conv2d(512,
                                        num_classes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

        # conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # Score
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        # UpScore2 using deconv - 2배 증가
        self.upscore2 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)

        # UpScore16 using deconv - 2배증가와 중간에 나온것을 더하여 진행
        self.upscore16 = nn.ConvTranspose2d(num_classes,
                                            num_classes,
                                            kernel_size=32,
                                            stride=16,
                                            padding=8)

    def forward(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = self.pool1(h)

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = self.pool2(h)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        h = self.pool3(h)

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)
        pool4 = h = self.pool4(h)  # skip진행

        # Score
        score_pool4c = self.score_pool4_fr(pool4)  # 따로 1*1 진행

        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)
        h = self.pool5(h)

        h = self.fc6(h)
        h = self.drop6(h)

        h = self.fc7(h)
        h = self.drop7(h)

        h = self.score_fr(h)

        # Up Score I
        upscore2 = self.upscore2(h)  # 2배 transpose진행

        # Sum I
        h = upscore2 + score_pool4c  # 더하기 진행

        # Up Score II
        upscore16 = self.upscore16(h)  # 더한것을

        return upscore16
