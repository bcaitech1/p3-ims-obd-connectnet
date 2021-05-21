import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import vgg16
from dataset import *

flip()

# pretrained_model = vgg16(pretrained = True)
# print(pretrained_model)
#features, classifiers = list(pretrained_model.features.children()), list(pretrained_model.classifier.children())

