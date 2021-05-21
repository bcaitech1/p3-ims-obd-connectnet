import os
import torch
import numpy as np
from utils import label_accuracy_score, add_hist
from lookahead import Lookahead
saved_dir = './saved'

def save_model(model, file_name='before_sudo.pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)

def train(num_epochs, model, data_loader, val_loader, val_every, device, file_name):
    learning_rate = 0.0001
    from torch.optim.swa_utils import AveragedModel, SWALR
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from segmentation_models_pytorch.losses import SoftCrossEntropyLoss, JaccardLoss
    from adamp import AdamP

    criterion = [SoftCrossEntropyLoss(smooth_factor=0.1), JaccardLoss('multiclass', classes=12)]
    optimizer = AdamP(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)
    swa_scheduler = SWALR(optimizer, swa_lr=learning_rate)
    swa_model = AveragedModel(model)
    look = Lookahead(optimizer, la_alpha=0.5)

    print('Start training..')
    best_miou = 0
    for epoch in range(num_epochs):
        hist = np.zeros((12, 12))
        model.train()
        for step, (images, masks, _) in enumerate(data_loader):
            loss = 0
            images = torch.stack(images)  # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)

            # inference
            outputs = model(images)
            for i in criterion:
                loss += i(outputs, masks)
            # loss 계산 (cross entropy loss)

            look.zero_grad()
            loss.backward()
            look.step()

            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)
            acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
            # step 주기에 따른 loss, mIoU 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, mIoU: {:.4f}'.format(
                    epoch + 1, num_epochs, step + 1, len(data_loader), loss.item(), mIoU))

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, val_miou = validation(epoch + 1, model, val_loader, criterion, device)
            if val_miou > best_miou:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_miou = val_miou
                save_model(model, file_name = file_name)

        if epoch > 3:
            swa_model.update_parameters(model)
            swa_scheduler.step()


def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        hist = np.zeros((12, 12))  # 중첩을위한 변수
        for step, (images, masks, _) in enumerate(data_loader):

            loss = 0
            # (batch, channel, height, width)
            images = torch.stack(images)
            # (batch, channel, height, width)
            masks = torch.stack(masks).long()

            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            for each in criterion:
                loss += each(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

            # 계산을 위한 중첩
            hist = add_hist(hist, masks.detach().cpu().numpy(),
                            outputs, n_class=12)

            # mIoU = label_accuracy_score(
            #     masks.detach().cpu().numpy(), outputs, n_class=12)[2]
            # mIoU_list.append(mIoU)

        # mIoU가 전체에대해 계산
        acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(
            epoch, avrg_loss, mIoU))
    return avrg_loss, mIoU