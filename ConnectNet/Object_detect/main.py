import warnings
import segmentation_models_pytorch as smp
warnings.filterwarnings('ignore')
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import *
from torch.utils.data import DataLoader
from train import *
from pseudo import *
from inference import *
import pandas as pd

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())

device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장

dataset_path = '../input/data'

batch_size = 16
num_epochs = 1
val_every = 1

train_path = dataset_path + '/train.json'
val_path = dataset_path + '/val.json'
test_path = dataset_path + '/test.json'

category_names = ['Backgroud','UNKNOWN','General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']

# 데이터셋
test_transform = A.Compose([A.Resize(256, 256),
                            ToTensorV2()])

train_transform = A.Compose([A.Resize(256, 256),
                            ToTensorV2()])

val_transform = A.Compose([A.Resize(256, 256),
                            ToTensorV2()])

test_dataset = COCODataLoader(data_dir=test_path,
                              dataset_path=dataset_path,
                              mode='test',
                              category_names=category_names,
                              transform=test_transform)

train_dataset = COCODataLoader(data_dir=train_path,
                               dataset_path=dataset_path,
                               mode='train',
                               category_names=category_names,
                               transform=train_transform)

val_dataset = COCODataLoader(data_dir=val_path,
                             dataset_path=dataset_path,
                             mode='val',
                             category_names=category_names,
                             transform=val_transform)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         num_workers=4,
                         collate_fn=collate_fn)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=collate_fn)

model = smp.FPN(encoder_name="efficientnet-b3", encoder_weights="imagenet", in_channels=3, classes=12)
model = model.to(device)

train(num_epochs, model, train_loader, val_loader, val_every, device, 'test.pt')

checkpoint = torch.load('saved/test.pt', map_location=device)
model = model.to(device)
model.load_state_dict(checkpoint)
pseudo_labeling(num_epochs, model, train_loader, val_loader, test_loader, device, val_every, 'test_sudo.pt')

submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
checkpoint = torch.load('saved/test_sudo.pt', map_location=device)
model = model.to(device)
model.load_state_dict(checkpoint)

file_names, preds = test(model, test_loader, device)

# PredictionString 대입
for file_name, string in zip(file_names, preds):
    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())},
                                   ignore_index=True)

# submission.csv로 저장
submission.to_csv("./submission/Torchvision.csv", index=False)