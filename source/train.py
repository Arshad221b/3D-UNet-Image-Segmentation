import torch
from config.model_config import Device_config, Image_config
import torch.nn.functional as F

class train_model():
    def __init__(self, model, optimizer, criteria, epochs, train_loader, valid_loader= None) -> None:
        self.model        = model
        self.optimizer    = optimizer
        self.criteria     = criteria
        self.epochs       = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.patch_size   = Image_config['BATCH_PATH_SIZE']
        self.device       = Device_config['device']

    def dice_loss(self, input_im, target):
        smooth          = 1.0
        iflat           = input_im#.flatten()
        tflat           = target#.flatten()
        intersection    = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

    def train_batch(self, data, model, optimizer, criteria):
        model.train()
        total_loss        = []
        ims_in, ims_out   = data
        optimizer.zero_grad()
        
        for k in range(0, len(ims_in), self.patch_size):
            ims_in_temp       = ims_in[k:k+self.patch_size]
            ims_out_temp      = ims_out[k:k+self.patch_size]
            pred_img_temp     = model(ims_in_temp.to(self.device))
            softmax_output    = F.softmax(pred_img_temp, dim=1)
            index_tensor      = torch.arange(0, 15, dtype=torch.float, device=self.device)
            index_tensor      = index_tensor.view(1, -1, 1, 1,1).expand_as(pred_img_temp)
            pred_img_temp     = torch.sum(index_tensor * softmax_output, dim=1, keepdim=True)

            optimizer.zero_grad()
            loss = criteria(pred_img_temp.to('cpu'), ims_out_temp)
            loss.backward()
            optimizer.step()
            total_loss.append(loss)

        total_loss = sum(total_loss)/len(total_loss)
        
        return total_loss, model
    
    def save_model(self,epoch, model):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        savepath="checkpoint_{}.t7".format(str(epoch))
        torch.save(state,savepath)
    
        
    def train_model(self):
        train_loss = []
        for epoch in range(self.epoch):
            print(epoch)
            epoch_loss = []

        for ix, ims in enumerate(self.train_loader):
            loss, model = self.train_batch(ims, model, self.optimizer, self.criteria)
            epoch_loss.append(loss)
            
        if epoch % 2 == 0:
            self.save_model(epoch, model)
        print('avg_loss', sum(epoch_loss)/len(epoch_loss))
        train_loss.append(sum(epoch_loss)/len(epoch_loss))


class validate_model():
    def __init__(self, model, criteria) -> None:
        self.model    = model
        self.criteria = criteria

    def validate_batch(self):
        self.model.eval()
        ims_in, ims_out   = self.data
        pred_img          = self.model(ims_in)
        total_loss        = self.criteria(pred_img, ims_out)
        total_loss.backward()
        return total_loss

