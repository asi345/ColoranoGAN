import os
import glob
import time
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
from fastai.data.external import untar_data, URLs

import torch
from torch import nn, optim
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

from dataloader import LabDataset
from model import ColoranoGAN
from utils import create_loss_meters, update_losses, lab_to_rgb, visualize, log_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = untar_data(URLs.COCO_SAMPLE)
path = str(path) + "/train_sample"
paths = glob.glob(path + "/*.jpg") 
train_paths = paths[:16000]
val_paths = paths[16000:19000]

train_dataset = LabDataset(paths=train_paths, split='train')
train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=4,
                            pin_memory=True)

val_dataset = LabDataset(paths=val_paths, split='val')
val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=4,
                            pin_memory=True)

def init_resnet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), L.size(0))
        if e%5==0:
          torch.save(net_G.state_dict(), "./drive/MyDrive/02456/resnet/res18-unet"+str(e)+".pt")
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")

net_G = init_resnet(n_input=1, n_output=2, size=256)
opt = optim.Adam(net_G.parameters(), lr=1e-4)
criterion = nn.L1Loss()        
pretrain_generator(net_G, train_dataloader, opt, criterion, 20)
torch.save(net_G.state_dict(), "./drive/MyDrive/02456/resnet/res18-unet.pt")

def train_model(model, train_dl, start_epoch=0, epochs=100, display_every=200):
    data = next(iter(val_dataloader)) 
    drive_path = "./drive/MyDrive/02456/"
    for e in range(start_epoch, epochs):
        loss_meter_dict = create_loss_meters()  
        i = 0                                  
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) 
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                #log_results(loss_out, loss_meter_dict) # function to print out the losses
                #visualize(model, data, save=False) # function displaying the model's outputs
                visualize(model, data, e+1, drive_path)
            if i % 25 == 0:
                loss_out.write(f"Epoch {e+1}/{epochs}\t")
                loss_out.write(f"Iteration {i}/{len(train_dl)}\t")
                log_results(loss_out, loss_meter_dict)
                loss_out.write("\n")
                loss_out.flush()
        if (e+1) % 5 == 0:
            EPOCH = e+1
            PATH = drive_path+"models_pre/epoch"+str(EPOCH)+".pt"
            torch.save({
                'epoch': EPOCH,
                'model_state_dict_D': model.net_D.state_dict(),
                'model_state_dict_G': model.net_G.state_dict(),
                'optimizer_state_dict_D': model.opt_D.state_dict(),
                'optimizer_state_dict_G': model.opt_G.state_dict()
            }, PATH)

drive_path = "./drive/MyDrive/02456/"
loss_out = open(drive_path + 'losses_pre.txt', 'a')

#model = MainModel()
net_G = init_resnet(n_input=1, n_output=2, size=256)
net_G.load_state_dict(torch.load("./drive/MyDrive/02456/resnet/res18-unet.pt", map_location=device))
model = MainModel(net_G=net_G)
PATH = './drive/MyDrive/02456/models_pre/epoch10.pt'
checkpoint = torch.load(PATH)
model.net_D.load_state_dict(checkpoint['model_state_dict_D'])
model.net_G.load_state_dict(checkpoint['model_state_dict_G'])
model.opt_D.load_state_dict(checkpoint['optimizer_state_dict_D'])
model.opt_G.load_state_dict(checkpoint['optimizer_state_dict_G'])
epoch = checkpoint['epoch']
train_model(model, train_dataloader, epoch, 100, 200)
#train_model(model, train_dl, 0, 100, 200)
