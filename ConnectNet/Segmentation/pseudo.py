from train import validation, save_model
from utils import add_hist, label_accuracy_score
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lookahead import *

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

saved_dir = './saved'

def pseudo_labeling(num_epochs, model, data_loader, val_loader, unlabeled_loader, device, val_every, file_name):
    # Instead of using current epoch we use a "step" variable to calculate alpha_weight
    # This helps the model converge faster
    from torch.optim.swa_utils import AveragedModel, SWALR
    from segmentation_models_pytorch.losses import SoftCrossEntropyLoss, JaccardLoss
    from adamp import AdamP

    criterion = [SoftCrossEntropyLoss(smooth_factor=0.1), JaccardLoss('multiclass', classes=12)]
    optimizer = AdamP(params=model.parameters(), lr=0.0001, weight_decay=1e-6)
    swa_scheduler = SWALR(optimizer, swa_lr=0.0001)
    swa_model = AveragedModel(model)
    optimizer = Lookahead(optimizer, la_alpha=0.5)

    step = 100
    size = 256
    best_mIoU = 0
    model.train()
    print('Start Pseudo-Labeling..')
    for epoch in range(num_epochs):
        hist = np.zeros((12, 12))
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
            loss = 0
            for each in criterion:
                loss += each(output, oms)

            unlabeled_loss = alpha_weight(
                step) * loss

            # Backpropogate
            optimizer.zero_grad()
            unlabeled_loss.backward()
            optimizer.step()
            output = torch.argmax(
                output.squeeze(), dim=1).detach().cpu().numpy()
            hist = add_hist(hist, oms.detach().cpu().numpy(),
                            output, n_class=12)

            if (batch_idx + 1) % 25 == 0:
                acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, mIoU:{:.4f}'.format(
                    epoch + 1, num_epochs, batch_idx + 1, len(unlabeled_loader), unlabeled_loss.item(), mIoU))
            # For every 50 batches train one epoch on labeled data
            # 50배치마다 라벨데이터를 1 epoch학습
            if batch_idx % 50 == 0:

                # Normal training procedure
                for batch_idx, (images, masks, _) in enumerate(data_loader):
                    labeled_loss = 0
                    images = torch.stack(images)
                    # (batch, channel, height, width)
                    masks = torch.stack(masks).long()

                    # gpu 연산을 위해 device 할당
                    images, masks = images.to(device), masks.to(device)

                    output = model(images)

                    for each in criterion:
                        labeled_loss += each(output, masks)

                    optimizer.zero_grad()
                    labeled_loss.backward()
                    optimizer.step()

                # Now we increment step by 1
                step += 1

        if (epoch + 1) % val_every == 0:
            avrg_loss, val_mIoU = validation(
                epoch + 1, model, val_loader, criterion, device)
            if val_mIoU > best_mIoU:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_mIoU = val_mIoU
                save_model(model, file_name = file_name)

        model.train()

        if epoch > 3:
            swa_model.update_parameters(model)
            swa_scheduler.step()