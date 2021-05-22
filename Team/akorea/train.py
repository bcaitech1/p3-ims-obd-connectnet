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
from utils import *
from cutmix.cutmix import CutMix
from model.smp import *
from scheduler import *
from segmentation_models_pytorch.losses import *

#pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
#pip install sklearn, adamp
#https://github.com/ildoonet/pytorch-gradual-warmup-lr



def start(config, wandb):
     # Loss function 정의
    dataset_path = '/opt/ml/input/data'
    test_path = dataset_path + '/test.json'
    
    num = config.data_ratio
    
    if num == -1:    
        train_path = dataset_path + '/train.json'
        val_path = dataset_path + '/val.json'
    else :
        train_path = dataset_path + '/train_data'+str(num)+'.json'
        val_path = dataset_path + '/valid_data'+str(num)+'.json'
    
    print(train_path)
    print(val_path)
    seed_everything(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())


    train_transform =  getattr(import_module("dataset"), "data_"+config.transform)()

    
    train_dataset = CustomDataLoader(data_dir=train_path,  mode='train', transform=train_transform)
    #train_dataset=CutMix(train_dataset, num_class=12, beta=1.0, prob=0.5, num_mix=2)

    # validation dataset
    val_transform = getattr(import_module("dataset"), "data_"+config.vtransform)()
    val_dataset = CustomDataLoader(data_dir=val_path,   mode='val', transform=val_transform)

    batch_size = config.batch_size
    # DataLoader

    # create own Dataset 1 (skip)
    # validation set을 직접 나누고 싶은 경우
    # random_split 사용하여 data set을 8:2 로 분할
    # train_size = int(0.8*len(dataset))
    # val_size = int(len(dataset)-train_size)
    # dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=3,
                                            drop_last=True,
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            drop_last=True,
                                            num_workers=2,
                                            collate_fn=collate_fn)

    if config.enc_name == "basic":
        mode_str = "model."+config.model.lower()
        model_module = getattr(import_module(mode_str), config.model)  
        model = model_module(num_classes=12).to(device)
    else:
        model_module = get_smp_model(config.model, config.enc_name)
        model = model_module.to(device)

    #Loss
    criterion = create_criterion(config.criterion)
    
    #criterion = [SoftCrossEntropyLoss(smooth_factor=0.1), JaccardLoss('multiclass', classes = 12)]
    
    #Optimizer
    optimizer= optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    
    if config.optim == "AdamP":
        optimizer = AdamP(model.parameters(), lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optim  == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config.lr , weight_decay=config.weight_decay)
    elif config.optim  == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr , weight_decay=config.weight_decay)
    
    lookahead = Lookahead(optimizer, k=5, alpha=0.5) # Initialize Lookahead
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    if config.lr_scheduler == "cosine":
        print('cosine')
        #Q = 2
        Q = config.epochs
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = Q, eta_min=1e-7)
    elif config.lr_scheduler == "cosinew":
        print(" ConsineAnnealingWarmRestarts ")
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(lookahead, T_0=30, T_mult=2, eta_min=0)
    elif  config.lr_scheduler == "cosinew_custom":
        print("https://gaussian37.github.io/dl-pytorch-lr_scheduler/#cosineannealingwarmrestarts-1")
        optimizer = torch.optim.Adam(model.parameters(), lr=0)
        lookahead = Lookahead(optimizer, k=5, alpha=0.5)
        scheduler = CustomCosineAnnealingWarmUpRestarts(optimizer, T_0=config.epochs, T_mult=1, eta_max=config.lr,  T_up=8, gamma=0.5)
    elif config.lr_scheduler == "gradual_warmuplr":
        print("#https://www.kaggle.com/pukkinming/pytorchgradualwarmuplr")
    
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, config , device, lookahead)

    psudo_labeling(model, train_loader, val_loader, criterion, optimizer, scheduler, config , device, lookahead)


def train(model, data_loader, val_loader, criterion, optimizer, scheduler, config, device, lookahead):
    print('Start training..')
    best_loss = 9999999
    best_miou =0
    low_train = 0
    for epoch in range(config.epochs):
        # if epoch > 5 and low_train == 5:
        #     print("break")
        #     break
        
        #hist = np.zeros((12, 12))
        train_total_loss =0

        model.train()
        for step, (images, masks, _) in enumerate(data_loader):
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
                  
            # # inference
            # outputs = model(images).to(device)
            
            # # loss 계산 (cross entropy loss)
            # loss = criterion(outputs, masks)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            # inference
            outputs = model(images)
            # loss 계산 (cross entropy loss)
            lookahead.zero_grad()
            loss = criterion(outputs, masks)
            #optimizer.zero_grad() lookahed 
            loss.backward()
            lookahead.step()
            #optimizer.step()
            #scheduler.step()
            train_total_loss +=loss
            # step 주기에 따른 loss 출력
            if (step + 1) % 10 == 0:
                # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} lr: {}'.format(
                #     epoch+1, config.epochs, step+1, len(data_loader), loss.item(),scheduler.get_last_lr()[0]))
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} lr: {}'.format(
                     epoch+1, config.epochs, step+1, len(data_loader), loss.item(),scheduler.get_lr()))
        train_loss = train_total_loss /(step+1)
        

        #scheduler.step()

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % config.val_every == 0:
            print('Enter validation')
            avrg_loss, avrg_miou = validation(epoch + 1, model, val_loader, criterion, device)
            #if avrg_loss < best_loss:
            if avrg_miou > best_miou:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', config.saved_dir)
                best_loss = avrg_loss
                best_miou = avrg_miou
                name = config.name+"_"+str(low_train)
                save_model(model, config.saved_dir, name)
                if epoch < 5: 
                    low_train = 0
            else :
                print(f'Low training:{low_train}')
                low_train += 1
            if config.wandb:
                print('Send to wandb')
                #wandb.log({"Train Loss": train_loss, "Epoch": epoch+1, "LR":scheduler.get_last_lr()[0], "Average Loss":avrg_loss, "mIoU":avrg_miou})
                wandb.log({"Train Loss": train_loss, "Epoch": epoch+1, "Saved Number":low_train, "Average Loss":avrg_loss, "mIoU":avrg_miou})
                print('Send to wandb')

def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch))
    n_class =12
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        total_acc, total_fwavacc, total_acc_cls =0,0,0
        hist = np.zeros((n_class, n_class))

        for step, (images, masks, infos) in enumerate(data_loader):
            
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)            

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
            # for lt, lp in zip(outputs, masks.detach().cpu().numpy()):
            #     hist += fast_hist(lt.flatten(), lp.flatten(), n_class)

            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=n_class)
            acc, acc_cls, avrg_miou, fwavacc = label_accuracy_score(hist)
            #mIoU_list.append(mIoU)
           
            # total_acc += acc
            # total_fwavacc += fwavacc
            # total_acc_cls += acc_cls
            
            # if step ==200 and config.wandb:
            #     wandb.log(wandb.Image(original_image, masks={
            #         "predictions" : {
            #             "mask_data" : output[0],
            #             "class_labels" : infos[0]
            #         },
            #         "ground_truth" : {
            #             "mask_data" : mask[0],
            #             "class_labels" : infos[0]
            #         }
            #     }))

                
        
        #avrg_miou = label_accuracy_score(hist)

        avrg_loss = total_loss / cnt
        # avrg_acc  = total_acc /cnt
        # avrg_fwavacc  = total_fwavacc /cnt
        # avrg_acc_cls  = total_acc_cls /cnt
        #avrg_miou=np.mean(mIoU_list)
        
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}, Acc: {:.4f}'.format(epoch, avrg_loss, avrg_miou,acc_cls))
    
            
    return avrg_loss, avrg_miou


def save_model(model, saved_dir, file_name='output'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name+".pt")
    torch.save(model.state_dict(), output_path)


def parse_args(parser, name):
    with open(os.path.join("./config/", name+".json"), 'r', encoding='utf-8') as f:
        config = json.load(f)
    parser.add_argument('--seed', type=int, default=config["seed"], help='random seed')
    parser.add_argument('--epochs', type=int, default=config["epochs"], help='number of epochs to train ')
    parser.add_argument('--transform', type=str, default=config["transform"], help='data augmentation type ')
    parser.add_argument('--vtransform', type=str, default=config["vtransform"], help='val data augmentation type ')
    parser.add_argument('--batch_size', type=int, default=config["batch_size"], help='input batch size for training ')
    parser.add_argument('--model', type=str, default=config["model"], help='model type ')
    parser.add_argument('--optim', type=str, default=config["optim"], help='optimizer type')
    parser.add_argument('--lr', type=float, default=config["lr"], help='learning rate')
    parser.add_argument('--data_ratio', type=int, default=config["data_ratio"], help='ratio for validaton ')
    parser.add_argument('--criterion', type=str, default=config["criterion"], help='criterion type ')
    parser.add_argument('--weight_decay', type=float, default=config["weight_decay"], help='weight_decay')
    parser.add_argument('--lr_scheduler', type=str, default=config["lr_scheduler"], help='lr_scheduler')
    parser.add_argument('--test', type=str, default=config["test"], help='test')
    parser.add_argument('--enc_name', type=str, default=config["enc_name"], help='encoder name')
    
    


    args = parser.parse_args()
    return args



if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    #settings :  ./config/base.json
    conf_name = "eff-b0"  

    config=parse_args(parser ,conf_name)    
    run_wandb = True
    

    if run_wandb:
        wandb.init(project="segmentation")
        wandb.config.update(config) 
        run_name = wandb.run.name.split("-")[0]
        wandb.run.name = run_name+"-"+config.model+config.enc_name[-4:]+"-"+str(random.randrange(0,8))
        wandb.run.save()
        config.name = wandb.run.name
         
    else:
        config.name = conf_name 
    

    config.wandb = run_wandb
    config.val_every =1
    config.saved_dir = "./saved"
    
    if not os.path.isdir(config.saved_dir):                                                           
        os.mkdir(config.saved_dir)


    
    print(config)

    start(config, wandb)

    wandb.finish()

   