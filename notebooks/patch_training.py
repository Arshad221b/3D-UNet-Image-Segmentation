import torch.nn as nn 
import torch 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from glob import glob 
import os 

from skimage import io
from patchify import patchify, unpatchify

import random 
import torch 
from torch.utils import data 
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image

import cv2
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import nibabel as nib

path= '/home/arshad/Downloads/amos22/amos22'
input_paths   = sorted(glob(os.path.join(path, "imagesVa","*.nii.gz")))[:85]
target_paths  = sorted(glob(os.path.join(path, "labelsVa","*.nii.gz")))[:85]

IMAGE_SIZE = 64
BATCH_SIZE = 1
NUM_CLASS = 15
PATCH_SIZE = 8

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AmosDataLoader(data.Dataset):
  def __init__(
      self, 
      input_paths: list, 
      target_paths: list, 
      transform_input = None, 
      transform_target = None
  ): 

    self.input_paths      = input_paths
    self.target_paths     = target_paths
    self.transform_input  = transform_input
    self.transform_target = transform_target

  def __len__(self):
    return len(self.input_paths)

  def preprocess_img_input(self, input_im):
    img_patches = patchify(input_im, (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE), step=IMAGE_SIZE)
    input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
    # input_im = np.stack((input_img,)*3, axis=-1)
    input_im = torch.tensor(input_img).float()/255
    input_im = input_im.unsqueeze(4)
    input_im = input_im.permute(0,4,1,2,3)
    return input_im
  
  def preprocess_img_output(self, output_im):
    img_patches = patchify(output_im, (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE), step=IMAGE_SIZE)
    output_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
    output_im = np.expand_dims(output_img, axis = 4)
    output_im = torch.tensor(output_im).float()/255
    output_im = output_im.permute(0,4,1,2,3)
    # output_im = output_im.squeeze(1)
    return output_im

#   def preprocess_output(self, output_im):
#     # z_factor_output      = output_im.shape[2]*int(output_im.shape[0]/IMAGE_SIZE)**2
#     # output_im            = torch.tensor(output_im).float()/255
#     # print('output shape before mask', output_im.shape)
#     mask_cat              = np.zeros((NUM_CLASS, *output_im.shape), dtype=np.float32)
#     for i in range(NUM_CLASS):
#         mask_cat[i][output_im == i] = 1
#     # output_im = torch.tensor(mask_cat).float()
#     output_im             = torch.tensor(mask_cat).float()/255
#     # print(output_im.shape)
#     output_im             = output_im.permute(0,2,3,1)
#     # print(output_im.shape)
#     output_im             = output_im.unsqueeze(0)#.unsqueeze(0)
#     # print('output shape befor inter',output_im.shape)
#     output_size_input     = (82, IMAGE_SIZE, IMAGE_SIZE)
#     output_im             = F.interpolate(output_im, size=output_size_input, mode='trilinear', align_corners=False)
#     output_im             = output_im#.squeeze(0)
    
    
#     # print('output shape final', output_im.shape)
#     return output_im

  def __getitem__(self,x):
    input_image = self.input_paths[x]
    mask_image  = self.target_paths[x]
    input_im    = nib.load(input_image).get_fdata()
    mask_im     = nib.load(mask_image).get_fdata()

    return input_im, mask_im
    
  def collate_fn(self, batch):
    # print(len(batch[0][0]))
    im_ins, im_outs = [], []
    for im_in, im_out  in batch: 
      im_in = self.preprocess_img_input(im_in)
      im_out = self.preprocess_img_output(im_out)

      # im_out = self.preprocess_output(im_out)
      # print(im_in.shape, im_out.shape)
      im_ins.append(im_in)
      im_outs.append(im_out)

    # print(torch.tensor(im_ins).shape)
    return torch.cat(im_ins, dim = 0), torch.cat(im_outs, dim= 0)
  

train_dl      = AmosDataLoader(input_paths, target_paths)
train_loader  = DataLoader(train_dl, batch_size = BATCH_SIZE, drop_last= True, collate_fn=train_dl.collate_fn)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)

class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=2, stride=2)

        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        # print(x1.shape, x2.shape)
        x1    = self.up(x1)
        # print(x1.shape)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1    = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2))
        x     = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.bilinear     = bilinear

        self.conv1    = DoubleConv3D(in_channels, 64)
        self.down1    = Down3D(64, 128)
        self.down2    = Down3D(128, 256)
        self.down3    = Down3D(256, 512)
        self.down4    = Down3D(512, 1024)
        self.up1      = Up3D(1024, 512, bilinear)
        self.up2      = Up3D(512, 256, bilinear)
        self.up3      = Up3D(256, 128, bilinear)
        self.up4      = Up3D(128, 64, bilinear)
        self.outconv  = OutConv3D(64, out_channels)

    def forward(self, x):
        # print(x.shape)
        # x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.shape, x4.shape)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2) 
        x9 = self.up4(x8, x1)
        output= self.outconv(x9)
        # print(x6.shape)
        # up network

        return output
    

model = UNet3D(1, NUM_CLASS).to(device)

def dice_loss(input_im, target):
    smooth = 1.0
    

    iflat = input_im#.flatten()
    tflat = target#.flatten()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


def train_batch(data, model, optimizer, criteria):
    model.train()
    total_loss = []
    ims_in, ims_out = data
    # print(ims_in.shape)
    optimizer.zero_grad()
    # pred_img = torch.tensor(ims_out.shape)
    # print(ims_in.shape)
    for k in range(0, len(ims_in), PATCH_SIZE):
        ims_in_temp = ims_in[k:k+PATCH_SIZE]
        ims_out_temp = ims_out[k:k+PATCH_SIZE]
        pred_img_temp = model(ims_in_temp.to(device))  
        # pred_img_temp = F.softmax(pred_img_temp, 1)
        # pred_img_temp = pred_img_temp.argmax(dim=1)
        softmax_output = F.softmax(pred_img_temp, dim=1)
        index_tensor = torch.arange(0, 15, dtype=torch.float, device=device)
        index_tensor = index_tensor.view(1, -1, 1, 1,1).expand_as(pred_img_temp)
        pred_img_temp = torch.sum(index_tensor * softmax_output, dim=1, keepdim=True)
        # pred_img_temp = torch.argmax(pred_img_temp, 1)
        
        # pred_img_temp = pred_img_temp.contiguous().view(-1)
        # ims_out_temp = ims_out_temp.contiguous().view(-1)
        # batch, dims, height, width, depth = pred_img_temp.shape
        
        # print(type(pred_img_temp))
        # _, pred_img_temp = torch.max(pred_img_temp, dim=1)
        # print(pred_img_temp.shape, ims_out_temp.shape)
        optimizer.zero_grad()
        # loss = dice_loss(pred_img_temp.to('cpu'), ims_out_temp)
        loss = criteria(pred_img_temp.to('cpu'), ims_out_temp)
        loss.backward()
        optimizer.step()
        total_loss.append(loss)
    # print(pred_img.shape)
    # gender_criterion, age_criterion = criteria
    # gender_loss = gender_criterion(pred_gender.squeeze(), gender)
    # age_loss = age_criterion(pred_age.squeeze(), age)
        
    total_loss = sum(total_loss)/len(total_loss)
    
    # total_loss.backward()
    # optimizer.step()
    return total_loss, model

n_epoch = 100
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def save_model(epoch, model):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    savepath="checkpoint_{}.t7".format(str(epoch))
    torch.save(state,savepath)
    

train_loss = []
for epoch in range(n_epoch):
  print(epoch)
  epoch_loss = []

  for ix, ims in enumerate(train_loader):
    loss, model = train_batch(ims, model, optimizer, criteria)
    epoch_loss.append(loss)
    
  if epoch % 2 == 0:
    save_model(epoch, model)
  print('avg_loss', sum(epoch_loss)/len(epoch_loss))
  train_loss.append(sum(epoch_loss)/len(epoch_loss))