import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

def base_model(encoder_name,encoder_weights,in_channels =3 , classes = 12 ):
    return smp.FPN(
        encoder_name="inceptionresnetv2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7     
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=12,                      # model output channels (number of classes in your dataset)
    )
    
class custom_model(nn.module):
    def __init__(self):
        super().__init__()


    def forward(self,x):

        return x