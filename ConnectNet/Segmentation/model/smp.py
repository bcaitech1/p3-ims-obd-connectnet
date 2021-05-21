import segmentation_models_pytorch as smp
import torch 
from importlib import import_module

def get_smp_model(seg_model, encoder_name):
    smp_model =getattr(smp,seg_model)
    model =  smp_model(
                 encoder_name=encoder_name,
                 encoder_weights='imagenet',
                 in_channels=3,
                 classes=12)
    return model

if __name__ =="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_smp_model("UnetPlusPlus","efficientnet-b0")
    x = torch.randn([1, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x).to(device)
    print("output shape : ", out.size())

    model = model.to(device)
