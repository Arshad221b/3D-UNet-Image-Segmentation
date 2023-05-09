import nibabel as nib
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
from config.model_config import Device_config, Image_config, Model_config
from patchify import patchify, unpatchify
from config import model_config

class AmosDataLoader(data.Dataset):
  def __init__(
      self, 
      input_paths: list, 
      target_paths: list, 
      transform_input = None, 
      transform_target = None,
      image_size = model_config.Image_config['IMAGE_SIZE']
  ): 

    self.input_paths      = input_paths
    self.target_paths     = target_paths
    self.transform_input  = transform_input
    self.transform_target = transform_target
    self.image_size       = image_size

  def __len__(self):
    return len(self.input_paths)

  def preprocess_img_input(self, input_im):
    img_patches = patchify(input_im, (self.image_size, self.image_size, self.image_size), step=self.image_size)
    input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
    input_im = torch.tensor(input_img).float()/255
    input_im = input_im.unsqueeze(4)
    input_im = input_im.permute(0,4,1,2,3)
    return input_im
  
  def preprocess_img_output(self, output_im):
    img_patches = patchify(output_im, (self.image_size, self.image_size, self.image_size), step=self.image_size)
    output_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
    output_im = np.expand_dims(output_img, axis = 4)
    output_im = torch.tensor(output_im).float()/255
    output_im = output_im.permute(0,4,1,2,3)
    return output_im

  def __getitem__(self,x):
    input_image = self.input_paths[x]
    mask_image  = self.target_paths[x]
    input_im    = nib.load(input_image).get_fdata()
    mask_im     = nib.load(mask_image).get_fdata()

    return input_im, mask_im
    
  def collate_fn(self, batch):
    im_ins, im_outs = [], []
    for im_in, im_out  in batch: 
      im_in = self.preprocess_img_input(im_in)
      im_out = self.preprocess_img_output(im_out)
      im_ins.append(im_in)
      im_outs.append(im_out)

    return torch.cat(im_ins, dim = 0), torch.cat(im_outs, dim= 0)