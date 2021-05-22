
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from dataset import * 
import time, os
import copy, random
from adamp import AdamP
import wandb
import argparse
import json
from importlib import import_module
from loss import create_criterion
from sklearn.model_selection import train_test_split
import math
from model.fcn8s import FCN8s
from utils import *
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

def inference(config):
    # best model 저장된 경로
    dataset_path = '/opt/ml/input/data'
    test_path = dataset_path + '/test.json'


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model_path = os.path.join(config.saved_dir, config.f)
    model_path = config.f
    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
  

    if config.enc_name == "custom":
        mode_str = "model."+config.model.lower()
        model_module = getattr(import_module(mode_str), config.model)  
        model = model_module(num_classes=12).to(device)
    else:
        model_module = get_smp_model(config.model, config.enc_name)
        model = model_module.to(device)

    model.load_state_dict(checkpoint)

    test_transform = A.Compose([ 
                    ToTensorV2()
        ])

    
    # test_transform = A.Compose([
    #                         A.Normalize(
    #                             mean=(0.485, 0.456, 0.406),
    #                             std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
    #                         ),    
    #                 ToTensorV2(transpose_mask=True)
    #     ])

    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=config.batch_size,
                                          num_workers=4,
                                          collate_fn=collate_fn)

    file_names, preds = test(model, test_loader, device)

    # sample_submisson.csv 열기
    submission = pd.read_csv('/opt/ml/code/submission/sample_submission.csv', index_col=None)


    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    filename = os.path.basename(config.f).split(".")[0]

    output_path = os.path.join(config.output_dir, filename+".csv")

    submission.to_csv(output_path, index=False)



def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(data_loader):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

            # oms =0
            # for i in range(5):
            #     outs = model[i](torch.stack(imgs).to(device))
            #     oms += outs.detach().cpu().numpy()
            # oms = torch.argmax(oms, dim=1)

            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default="saved/basic.pt", help='model file')
    parser.add_argument('--model', type=str, default="FPN", help='model')
    parser.add_argument('--enc_name', type=str, default="efficientnet-b3", help='encoder name')
    #parser.add_argument('--enc_name', type=str, default="se_resnext101_32x4d", help='encoder name')
    
    config = parser.parse_args()

    config.seed = 21
    config.batch_size = 16
    config.output_dir = "./output"
    seed_everything(config.seed)
    
    if not os.path.isdir(config.output_dir):                                                           
        os.mkdir(config.output_dir)

 
    inference(config)