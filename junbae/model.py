import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg16
import segmentation_models_pytorch as smp
from utils import seed_everything

seed_everything(21)


class FCN8sBase(nn.Module):  # basecode
    def __init__(self, num_classes):
        super(FCN8sBase, self).__init__()
        self.pretrained_model = vgg16(pretrained=True)
        features, classifiers = list(self.pretrained_model.features.children()), list(
            self.pretrained_model.classifier.children())

        self.features_map1 = nn.Sequential(*features[0:17])
        self.features_map2 = nn.Sequential(*features[17:24])
        self.features_map3 = nn.Sequential(*features[24:31])

        # Score pool3
        self.score_pool3_fr = nn.Conv2d(256, num_classes, 1)

        # Score pool4
        self.score_pool4_fr = nn.Conv2d(512, num_classes, 1)

        # fc6 ~ fc7
        self.conv = nn.Sequential(nn.Conv2d(512, 4096, kernel_size=1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(),
                                  nn.Conv2d(4096, 4096, kernel_size=1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout()
                                  )

        # Score
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        # UpScore2 using deconv
        self.upscore2 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)

        # UpScore2_pool4 using deconv
        self.upscore2_pool4 = nn.ConvTranspose2d(num_classes,
                                                 num_classes,
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1)

        # UpScore8 using deconv
        self.upscore8 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=16,
                                           stride=8,
                                           padding=4)

    def forward(self, x):
        pool3 = h = self.features_map1(x)
        pool4 = h = self.features_map2(h)
        h = self.features_map3(h)

        h = self.conv(h)
        h = self.score_fr(h)

        score_pool3c = self.score_pool3_fr(pool3)
        score_pool4c = self.score_pool4_fr(pool4)

        # Up Score I
        upscore2 = self.upscore2(h)

        # Sum I
        h = upscore2 + score_pool4c

        # Up Score II
        upscore2_pool4c = self.upscore2_pool4(h)

        # Sum II
        h = upscore2_pool4c + score_pool3c

        # Up Score III
        upscore8 = self.upscore8(h)

        return upscore8


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


class FCN32s(nn.Module):
    def __init__(self, num_classes=12):
        super(FCN32s, self).__init__()

        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            # convolution
            # relu
            # 한블럭
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU(inplace=True))

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

        # conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # fc6
        self.fc6 = CBR(512, 4096, 1, 1, 0)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = CBR(4096, 4096, 1, 1, 0)
        self.drop7 = nn.Dropout2d()

        self.score = nn.Conv2d(4096, num_classes, 1, 1, 0)

        self.upscore32 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=64, stride=32, padding=16)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        # conv2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        # conv3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        # conv4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        # conv5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        x = self.fc6(x)
        x = self.drop6(x)

        x = self.fc7(x)
        x = self.drop7(x)

        x = self.score(x)
        output = self.upscore32(x)

        return output


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


def conv3x3_relu(in_ch, out_ch, rate=1):
    conv3x3_relu = nn.Sequential(nn.Conv2d(in_ch,
                                           out_ch,
                                           kernel_size=3,
                                           stride=1,
                                           padding=rate,
                                           dilation=rate),
                                 nn.ReLU())
    return conv3x3_relu


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(conv3x3_relu(3, 64),
                                      conv3x3_relu(64, 64),
                                      nn.MaxPool2d(3, stride=2, padding=1),
                                      conv3x3_relu(64, 128),
                                      conv3x3_relu(128, 128),
                                      nn.MaxPool2d(3, stride=2, padding=1),
                                      conv3x3_relu(128, 256),
                                      conv3x3_relu(256, 256),
                                      conv3x3_relu(256, 256),
                                      nn.MaxPool2d(3, stride=2, padding=1),
                                      conv3x3_relu(256, 512),
                                      conv3x3_relu(512, 512),
                                      conv3x3_relu(512, 512),
                                      # 마지막 stride=1로 해서 두 layer 크기 유지
                                      nn.MaxPool2d(3, stride=1, padding=1),
                                      # and replace subsequent conv layer r=2
                                      conv3x3_relu(512, 512, rate=2),
                                      conv3x3_relu(512, 512, rate=2),
                                      conv3x3_relu(512, 512, rate=2),
                                      nn.MaxPool2d(3, stride=1, padding=1))  # 마지막 stride=1로 해서 두 layer 크기 유지

    def forward(self, x):
        output = self.features(x)
        return output


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
        self.aspp1 = ASPPConv(inplanes, outplanes, 1,
                              padding=0, dilation=dilations[0])
        self.aspp2 = ASPPConv(inplanes, outplanes, 3,
                              padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPConv(inplanes, outplanes, 3,
                              padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPConv(inplanes, outplanes, 3,
                              padding=dilations[3], dilation=dilations[3])
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
        x5 = F.interpolate(x5, size=x.size()[
                           2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        output = self.project(x)
        return output


class DeepLabHead(nn.Sequential):
    def __init__(self, in_ch, out_ch, n_classes):
        super(DeepLabHead, self).__init__()
        self.add_module("0", ASPP(in_ch, out_ch))
        self.add_module("1", nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module("2", nn.BatchNorm2d(out_ch))
        self.add_module("3", nn.ReLU())
        self.add_module("4", nn.Conv2d(
            out_ch, n_classes, kernel_size=1, stride=1))


class DeepLabV3(nn.Sequential):

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV3, self).__init__()
        self.backbone = VGG16()
        self.classifier = DeepLabHead(in_ch=512, out_ch=256, n_classes=12)

    def forward(self, x):
        h = self.backbone(x)
        h = self.classifier(h)
        output = F.interpolate(
            h, size=x.shape[2:], mode="bilinear", align_corners=False)
        return output
