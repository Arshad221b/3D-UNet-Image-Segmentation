import nibabel as nib
import numpy as np
import torch
from torch.utils import data

from config.model_config import Device_config, Image_config, Model_config


class AmosDataLoader(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        target_paths: list,
        transform_input=None,
        transform_target=None
    ):

        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.image_size = Image_config['IMAGE_SIZE']
        self.device = Device_config['device']
        self.num_class = Model_config['NUM_CLASS']
        self.output_chan = Model_config['OUTPUT_CHANNEL']

    def __len__(self):
        return len(self.input_paths)

    def preprocess_img_input(self, input_im):
        # z_factor_input      = input_im.shape[2]*int(input_im.shape[0]/IMAGE_SIZE)**2
        input_im = np.stack((input_im,)*3, axis=-1)
        input_im = torch.tensor(input_im).float()/255
        input_im = input_im.permute(3, 2, 0, 1)

        input_im = input_im.unsqueeze(0)
        output_size_input = (
            self.output_chan, self.image_size, self.image_size)
        input_im = F.interpolate(
            input_im, size=output_size_input, mode='trilinear', align_corners=False)
        input_im = input_im  # .squeeze(0)
        return input_im

    def preprocess_output(self, output_im):
        # z_factor_output      = output_im.shape[2]*int(output_im.shape[0]/IMAGE_SIZE)**2
        # output_im            = torch.tensor(output_im).float()/255
        mask_cat = np.zeros(
            (self.num_class, *output_im.shape), dtype=np.float32)
        for i in range(self.num_class):
            mask_cat[i][output_im == i] = 1
        # output_im = torch.tensor(mask_cat).float()
        output_im = torch.tensor(mask_cat).float()/255
        output_im = output_im.permute(0, 2, 3, 1)
        output_im = output_im.unsqueeze(0)  # .unsqueeze(0)
        output_size_input = (
            self.output_chan, self.image_size, self.image_size)
        output_im = F.interpolate(
            output_im, size=output_size_input, mode='trilinear', align_corners=False)
        # output_im = output_im  # .squeeze(0)

        return output_im

    def __getitem__(self, x):
        input_image = self.input_paths[x]
        target_image = self.target_paths[x]
        input_im = nib.load(input_image).get_fdata()
        target_im = nib.load(target_image).get_fdata()

        return input_im, target_im

    def collate_fn(self, batch):
        im_ins, im_outs = [], []
        for im_in, im_out in batch:
            im_in = self.preprocess_img_input(im_in)
            im_out = self.preprocess_output(im_out)
            im_ins.append(im_in)
            im_outs.append(im_out)

        return torch.cat(im_ins, dim=0).to(self.device), torch.cat(im_outs, dim=0).to(self.device)
