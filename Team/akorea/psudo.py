import torch
from dataset import CustomDataLoader
from torch.utils.data import Subset, DataLoader, Dataset
import torchvision.transforms as transforms

from utils import seed_everything
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch.nn.functional as F
from utils import label_accuracy_score, seed_everything, add_hist
import os
import wandb
from model.smp import *

import torch.nn as nn
import torch.nn.functional as F

def collate_fn(batch):
    return tuple(zip(*batch))


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


def alpha_weight(epoch):
    T1 = 100
    T2 = 700
    af = 3
    if epoch < T1:
        return 0.0
    elif epoch > T2:
        return af
    else:
        return ((epoch-T1) / (T2-T1))*af


def psudo_labeling(num_epochs, model, data_loader, val_loader, unlabeled_loader, criterion, optimizer, device, n_class, saved_dir, file_name, val_every):
    # Instead of using current epoch we use a "step" variable to calculate alpha_weight
    # This helps the model converge faster
    step = 100
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    preds_array = np.empty((0, size*size), dtype=np.long)
    file_name_list = []
    best_mIoU = 0
    model.train()
    for epoch in range(num_epochs):
        hist = np.zeros((n_class, n_class))
        for batch_idx, (imgs, image_infos) in enumerate(unlabeled_loader):

            # Forward Pass to get the pseudo labels
            # --------------------------------------------- test(unlabelse)를 모델에 통과
            model.eval()
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(
                outs.squeeze(), dim=1).detach().cpu().numpy()
            oms = torch.Tensor(oms)
            oms = oms.long()
            oms = oms.to(device)

            # --------------------------------------------- 학습

            model.train()
            # Now calculate the unlabeled loss using the pseudo label
            imgs = torch.stack(imgs)
            imgs = imgs.to(device)
            # preds_array = preds_array.to(device)

            output = model(imgs)

            unlabeled_loss = alpha_weight(
                step) * criterion(output, oms)

            # Backpropogate
            optimizer.zero_grad()
            unlabeled_loss.backward()
            optimizer.step()
            output = torch.argmax(
                output.squeeze(), dim=1).detach().cpu().numpy()
            hist = add_hist(hist, oms.detach().cpu().numpy(),
                            output, n_class=n_class)

            if (batch_idx + 1) % 25 == 0:
                acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, mIoU:{:.4f}'.format(
                    epoch+1, num_epochs, batch_idx+1, len(unlabeled_loader), unlabeled_loss.item(), mIoU))
            # For every 50 batches train one epoch on labeled data
            # 50배치마다 라벨데이터를 1 epoch학습
            if batch_idx % 50 == 0:

                # Normal training procedure
                for batch_idx, (images, masks, _) in enumerate(train_loader):
                    images = torch.stack(images)
                    # (batch, channel, height, width)
                    masks = torch.stack(masks).long()

                    # gpu 연산을 위해 device 할당
                    images, masks = images.to(device), masks.to(device)

                    output = model(images)
                    labeled_loss = criterion(output, masks)

                    optimizer.zero_grad()
                    labeled_loss.backward()
                    optimizer.step()

                # Now we increment step by 1
                step += 1

        if (epoch + 1) % val_every == 0:
            avrg_loss, val_mIoU = validation(
                epoch + 1, model, val_loader, criterion, device, n_class)
            if val_mIoU > best_mIoU:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_mIoU = val_mIoU
                save_model(model, saved_dir, file_name)
            wandb.log({"val_loss": avrg_loss, "val_mIoU": val_mIoU,
                      "best_mIoU": best_mIoU})

        model.train()


def save_model(model, saved_dir, file_name):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)


if __name__ == '__main__':
    seed_everything(21)
    num = 4
    wandb.init(project='segmeatation-psudo')
    dataset_path = '/opt/ml/input/data'
    test_path = dataset_path + '/test.json'
    train_path = dataset_path + '/train_data'+str(num)+'.json'
    val_path = dataset_path + '/valid_data'+str(num)+'.json'
    saved_dir = './saved'
    file_name = 'psudo_test_best_moiu.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    acc_scores = []
    unlabel = []
    pseudo_label = []

    alpha_log = []
    test_acc_log = []
    test_loss_log = []
    batch_size = 4
    num_epochs = 20
    learning_rate = 0.0001
    weight_decay = 1e-6
    val_every = 1
    # 모델
    model_path = './saved/hardy-FPNt-b3-5_23.pt'

    model = get_smp_model('FPN', 'efficientnet-b3')

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
    # 데이터셋
    test_transform = A.Compose([
        A.Resize(512, 512),
        ToTensorV2()
    ])
    test_dataset =CustomDataLoader(data_dir=test_path,   mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=2,
                                              collate_fn=collate_fn,
                                              drop_last=True)
    train_transform = A.Compose([
        A.Resize(512, 512),
        ToTensorV2()
    ])

    train_dataset =CustomDataLoader(data_dir=train_path,   mode='train', transform=test_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               collate_fn=collate_fn,
                                               drop_last=True)
    val_transform = A.Compose([
        A.Resize(512, 512),
        ToTensorV2()
    ])
    val_dataset =CustomDataLoader(data_dir=val_path,   mode='val', transform=test_transform)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=2,
                                             collate_fn=collate_fn,
                                             drop_last=True)

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

    checkpoint = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.load_state_dict(checkpoint)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    psudo_labeling(num_epochs, model, train_loader, val_loader, test_loader, criterion,
                   optimizer, device, n_class=12, saved_dir=saved_dir, file_name=file_name, val_every=val_every)
    filename = "psudo_"+str(num)+".pt"
    save_model(model, './saved', filename)