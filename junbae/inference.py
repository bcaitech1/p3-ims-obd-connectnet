import argparse
import os
from importlib import import_module
import segmentation_models_pytorch as smp

import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import COCODataLoader
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import seed_everything
from models.DeepV3 import *
from models.smp import *

def collate_fn(batch):
    return tuple(zip(*batch))


def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(
                outs.squeeze(), dim=1).detach().cpu().numpy()

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
    seed_everything(21)
    # best model 저장된 경로
    model_path = './saved/fpn_b16_e20.pt'
    # submission저장
    output_file = "./submission/fpn_b16_e20.csv"
    dataset_path = '../../input/data'
    test_path = dataset_path + '/test.json'
    batch_size = 16   # Mini-batch size

    # 모델
    # model = DeepLabV3_vgg16pretrained(
    #     n_classes=12, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
    model = get_smp_model('FPN','efficientnet-b0')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    category_names = ['Backgroud',
                      'UNKNOWN',
                      'General trash',
                      'Paper',
                      'Paper pack',
                      'Metal',
                      'Glass',
                      'Plastic',
                      'Styrofoam',
                      'Plastic bag',
                      'Battery',
                      'Clothing']

    test_transform = A.Compose([
        ToTensorV2()
    ])
    # test dataset
    test_dataset = COCODataLoader(
        data_dir=test_path, dataset_path=dataset_path,  mode='test',  category_names=category_names, transform=test_transform)

    checkpoint = torch.load(model_path, map_location=device)

    model = model.to(device)

    model.load_state_dict(checkpoint)
    # sample_submisson.csv 열기
    submission = pd.read_csv(
        './submission/sample_submission.csv', index_col=None)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              collate_fn=collate_fn)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
                                       ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(
        output_file, index=False)
    print(f'Inference Done!')
