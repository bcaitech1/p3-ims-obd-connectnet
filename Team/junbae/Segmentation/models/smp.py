import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg16
import segmentation_models_pytorch as smp
from utils import seed_everything

seed_everything(21)


def get_smp_model(seg_model, encoder_name, weight='imagenet'):
    smp_model = getattr(smp, seg_model)
    model = smp_model(
        encoder_name=encoder_name,
        encoder_weights=weight,
        in_channels=3,
        classes=12)
    return model
