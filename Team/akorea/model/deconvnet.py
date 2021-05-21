import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import vgg16
import torch
'''
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) conv1
    
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) conv2
    
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  conv3
    
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) conv4
    
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) conv5
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
'''
def DCB(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, 
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

class DeconvNet(nn.Module):
        
    def __init__(self, num_classes):
        super(DeconvNet,self).__init__()
        
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
         
        
        # fc6 ~ fc7
        self.fc = nn.Sequential(nn.Conv2d(512, 4096, kernel_size = 7),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(),
                                  nn.Conv2d(4096, 4096, kernel_size = 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout()
                                )
        
        # Deconv
        self.fc_deconv = DCB(4096, 512, 7, 1,0)
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.deconv5 =  nn.Sequential(
                                        DCB(512, 512, 3, 1, 1),
                                        DCB(512, 512, 3, 1, 1),
                                        DCB(512, 512, 3, 1, 1)
                                    )
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.deconv4 =  nn.Sequential(
                                DCB(512, 512, 3, 1, 1),
                                DCB(512, 512, 3, 1, 1),
                                DCB(512, 256, 3, 1, 1)
                            )
        
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv3 =  nn.Sequential(
                                DCB(256, 256, 3, 1, 1),
                                DCB(256, 256, 3, 1, 1),
                                DCB(256, 128, 3, 1, 1)
                            )
        
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2 =  nn.Sequential(
                                DCB(128, 128, 3, 1, 1),
                                DCB(128, 64, 3, 1, 1)
                            )

        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1 =  nn.Sequential(
                                DCB(64, 64, 3, 1, 1),
                                DCB(64, 64, 3, 1, 1)
                            )

        self.score = nn.Conv2d(64,
                                    num_classes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
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
        
        h = self.fc(h)

         
        h = self.fc_deconv(h)     
        
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

    model = DeconvNet(num_classes=12)
    x = torch.randn([1, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x).to(device)
    print("output shape : ", out.size())

    model = model.to(device)