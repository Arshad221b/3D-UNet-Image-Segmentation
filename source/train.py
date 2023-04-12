import torch
from source.model import UNet3D
from config.model_config import Device_config, Image_config, Model_config

class train_model():
    def __init__(self, model, optimizer, criteria, epochs, train_loader, valid_loader= None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criteria = criteria
        # self.data = data
        self.epochs = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train_batch(self, data):
        self.model.train()
        ims_in, ims_out = data
        self.optimizer.zero_grad()
        pred_img = self.model(ims_in)
        total_loss = self.criteria(pred_img, ims_out)
        total_loss.backward()
        self.optimizer.step()
        return total_loss
    
    def train_model(self):
        for epoch in range(self.epochs):
            print(epoch)
            epoch_loss = []

        for ix, ims in enumerate(self.train_loader):
            loss = self.train_batch(ims)
            epoch_loss.append(loss)
        print('avg_loss', sum(epoch_loss)/len(epoch_loss))


class validate_model():
    def __init__(self, model, criteria) -> None:
        self.model = model
        self.criteria = criteria

    def validate_batch(self):
        self.model.eval()
        ims_in, ims_out = self.data
        pred_img = self.model(ims_in)
        total_loss = self.criteria(pred_img, ims_out)
        total_loss.backward()
        return total_loss

