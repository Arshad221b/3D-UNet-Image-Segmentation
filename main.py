import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config.model_config import Device_config, Image_config, Model_config
from source.dataloader import AmosDataLoader
from source.model import UNet3D
from source.train import train_model, validate_model


class Run_Segmentation():
    def __init__(self, input_paths, target_paths) -> None:
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.batch_size = Model_config['BATCH_SIZE']
        self.num_class = Model_config['NUM_CLASS']
        self.input_chan = Model_config['INPUT_DIM']
        self.output_dim = Model_config['OUTPUT_CHANNEL']
        self.num_epochs = Model_config['EPOCHS']
        self.device = Device_config['device']
        
        
    def run_train_model(self):
        data = AmosDataLoader(self.input_paths, self.target_paths)
        train_loader= DataLoader(data, batch_size = self.batch_size, drop_last= True, collate_fn=data.collate_fn)
        model = UNet3D(self.input_chan, self.num_class).to(self.device)
        
        n_epoch = self.num_epochs
        criteria = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        train = train_model(model, optimizer, criteria, n_epoch, train_loader)
        train.train_model()


import os
from glob import glob

path = '/Users/arshad_221b/Downloads/Projects/create_shorts-main/MedSeg/AMOS/amos22/'


input_paths   = sorted(glob(os.path.join(path, "imagesVa","*.nii.gz")))
target_paths  = sorted(glob(os.path.join(path, "labelsVa","*.nii.gz")))

r = Run_Segmentation(input_paths, target_paths)
r.run_train_model()


        
        
                