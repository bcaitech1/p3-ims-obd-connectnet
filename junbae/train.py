import wandb
import segmentation_models_pytorch as smp
import seaborn as sns
from loss import create_criterion
from dataset import COCODataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from importlib import import_module
import re
import glob
import argparse
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torchvision.transforms as transforms
import torchvision
from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import cv2
from utils import label_accuracy_score, seed_everything, add_hist
import torch.nn as nn
import torch
import os
import random
import time
import json
import warnings
warnings.filterwarnings('ignore')


def collate_fn(batch):
    return tuple(zip(*batch))


def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device, file_name, n_class):
    print('Start training..')
    best_mIoU = 0
    for epoch in range(num_epochs):
        hist = np.zeros((n_class, n_class))
        model.train()
        for step, (images, masks, _) in enumerate(data_loader):
            # (batch, channel, height, width)
            images = torch.stack(images)
            # (batch, channel, height, width)
            masks = torch.stack(masks).long()

            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)

            # inference
            outputs = model(images)

            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs = torch.argmax(
                outputs.squeeze(), dim=1).detach().cpu().numpy()
            hist = add_hist(hist, masks.detach().cpu().numpy(),
                            outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
            wandb.log({"loss": loss, "mIoU": mIoU})  # wandb 로그출력
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, mIoU:{:.4f}'.format(
                    epoch+1, num_epochs, step+1, len(train_loader), loss.item(), mIoU))

        # validation 주기에 따른 loss 출력 및 best model 저장
        # mIoU에 따라 모델 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, val_mIoU = validation(
                epoch + 1, model, val_loader, criterion, device, n_class)
            if val_mIoU > best_mIoU:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_mIoU = val_mIoU
                save_model(model, saved_dir, file_name)


def validation(epoch, model, data_loader, criterion, device, n_class):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        hist = np.zeros((n_class, n_class))  # 중첩을위한 변수
        for step, (images, masks, _) in enumerate(data_loader):

            # (batch, channel, height, width)
            images = torch.stack(images)
            # (batch, channel, height, width)
            masks = torch.stack(masks).long()

            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.argmax(
                outputs.squeeze(), dim=1).detach().cpu().numpy()

            # 계산을 위한 중첩
            hist = add_hist(hist, masks.detach().cpu().numpy(),
                            outputs, n_class=n_class)

            # mIoU = label_accuracy_score(
            #     masks.detach().cpu().numpy(), outputs, n_class=12)[2]
            # mIoU_list.append(mIoU)

        # mIoU가 전체에대해 계산
        acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(
            epoch, avrg_loss, mIoU))
    return avrg_loss, mIoU


def save_model(model, saved_dir, file_name):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)


if __name__ == '__main__':
    # 시드 고정
    wandb.init(project='seg_det', entity='deokisys',
               name="efficient-b0 unet b8 e20")
    seed_everything(21)

    file_name = "efficientnet_baseline.pt"
    batch_size = 8   # Mini-batch size
    num_epochs = 20
    learning_rate = 0.0001
    weight_decay = 1e-6
    # 모델 저장 함수 정의
    val_every = 1

    dataset_path = '../../input/data'
    # anns_file_path = dataset_path + '/' + 'train.json'

    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'

    saved_dir = './saved'
    if not os.path.isdir(saved_dir):
        os.mkdir(saved_dir)

    device = "cuda" if torch.cuda.is_available(
    ) else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장

    train_transform = A.Compose([
        ToTensorV2()
    ])

    val_transform = A.Compose([
        ToTensorV2()
    ])

    # create own Dataset 1 (skip)
    # validation set을 직접 나누고 싶은 경우
    # random_split 사용하여 data set을 8:2 로 분할
    # train_size = int(0.8*len(dataset))
    # val_size = int(len(dataset)-train_size)
    # dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create own Dataset 2
    # train dataset

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

    train_dataset = COCODataLoader(
        data_dir=train_path, dataset_path=dataset_path, mode='train', category_names=category_names, transform=train_transform)

    # validation dataset
    val_dataset = COCODataLoader(
        data_dir=val_path, dataset_path=dataset_path,  mode='val', category_names=category_names, transform=val_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               collate_fn=collate_fn,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             collate_fn=collate_fn,
                                             drop_last=True)

    # model = DeepLabV3(n_classes=12, n_blocks=[
    #     3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])

    # model 불러오기
    # 출력 레이블 수 정의 (classes = 12)
    model = smp.Unet(encoder_name='efficientnet-b0', in_channels=3, classes=12,
                     encoder_weights="imagenet", activation=None)

    x = torch.randn([2, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x).to(device)
    print("output shape : ", out.size())

    model = model.to(device)

    # Loss function 정의
    criterion = nn.CrossEntropyLoss()

    # Optimizer 정의
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train(num_epochs, model, train_loader, val_loader,
          criterion, optimizer, saved_dir, val_every, device, file_name, len(category_names))
