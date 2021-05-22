import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import vgg16
import torch

def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, 
                            out_channels=out_channels,
                            kernel_size=kernel_size, 
                            stride=stride,
                            padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())        

def make_layers():
    vgg16 = models.vgg16(pretrained=True)
    features = list(vgg16.features.children())
    
    conv1 = nn.Sequential(*features[:4])
    conv2 = nn.Sequential(*features[5:9])
    conv3 = nn.Sequential(*features[10:16])
    conv4 = nn.Sequential(*features[17:23])
    conv5 = nn.Sequential(*features[24:30])

    return [conv1, conv2, conv3, conv4, conv5]


class SegNet(nn.Module):
        
    def __init__(self, num_classes):
        super(SegNet,self).__init__()
        self.pretrained_model = vgg16(pretrained = True)
        features, classifiers = list(self.pretrained_model.features.children()), list(self.pretrained_model.classifier.children())

        layers = make_layers()
        self.features_map1 = layers[0]

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) #

        self.features_map2 = layers[1]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) #

        self.features_map3 = layers[2]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) #
        
        self.features_map4 = layers[3]
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) #

        self.features_map5 = layers[4]
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) #
         
        
        
        # Deconv
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.deconv5 =  nn.Sequential(
                                        CBR(512, 512, 3, 1, 1),
                                        CBR(512, 512, 3, 1, 1),
                                        CBR(512, 512, 3, 1, 1)
                                    )
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.deconv4 =  nn.Sequential(
                                CBR(512, 512, 3, 1, 1),
                                CBR(512, 512, 3, 1, 1),
                                CBR(512, 256, 3, 1, 1)
                            )
        
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv3 =  nn.Sequential(
                                CBR(256, 256, 3, 1, 1),
                                CBR(256, 256, 3, 1, 1),
                                CBR(256, 128, 3, 1, 1)
                            )
        
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2 =  nn.Sequential(
                                CBR(128, 128, 3, 1, 1),
                                CBR(128, 64, 3, 1, 1)
                            )

        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1 =  nn.Sequential(
                                CBR(64, 64, 3, 1, 1),
                                CBR(64, 64, 3, 1, 1)
                            )

        self.score = nn.Conv2d(64,
                                    num_classes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    dilation=1)
        
        
    
    def forward(self, x):
        h= self.features_map1(x)
        h,pool1_indices = self.pool1(h)

        h= self.features_map2(h)
        h , pool2_indices= self.pool2(h)
        
        h= self.features_map3(h)
        h , pool3_indices= self.pool3(h)
        
        h= self.features_map4(h)
        h, pool4_indices = self.pool4(h)
        
        h= self.features_map5(h)
        h, pool5_indices = self.pool5(h)
            
        
        h = self.unpool5(h, pool5_indices)
        h = self.deconv5(h)              

        h = self.unpool4(h, pool4_indices)
        h = self.deconv4(h)  

        h = self.unpool3(h, pool3_indices)
        h = self.deconv3(h)  
                          
        h = self.unpool2(h, pool2_indices)
        h = self.deconv2(h)  
                                          
        h = self.unpool1(h, pool1_indices)
        h = self.deconv1(h)  
                                    
        
        h = self.score(h)   
        
        return h

if __name__ =="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SegNet(num_classes=12)
    x = torch.randn([1, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x).to(device)
    print("output shape : ", out.size())

    model = model.to(device)