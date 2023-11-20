from dataset.dataset import ASLDataset
from model.TwoStreamCNN import TwoStreamCNN
import torch
from torch import nn
import numpy as np
from torch.optim import Adam
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from utils import utils
device = utils.get_device()

class TwoStreamCNNrunner():
    def __init__(self,config, logger, transform = None):
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.image_size = config.data.image_size 
        
        
        self.data_path = config.path.raw_path
        
        
        self.epoch = config.train.epoch
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = self._get_optim(config.train.optimizer)
        self.lr = config.train.lr
        
        
        self.transform = transform
        self.model = TwoStreamCNN(type = "tsma")
        self.train_loader, self.test_loader = self._init_data(0.2)
        
        
    def train(self):
        print('Start training on device {}'.format(device))
        self.model = self.model.to(device)
        best_val_loss = 1e9

        for epoch in range(self.epoch):
            print('Start epoch {}'.format(epoch))

        # set up model state to training
            self.model.train()

            for (x, y) in self.train_loader:
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)

                y_pred = self.model(x)

                loss = self.criterion(y_pred, y)
                loss.backward()  # calculate gradient
                self.optimizer.step()  # update model parameters by gradient
                self.optimizer.zero_grad()  # set gradient to zero for next loop
                self.global_step += 1

                # need to detach loss from calculating tree of pytorch
                self.logger.add_scalar('train_loss', loss, global_step=self.global_step)

        # set up model state to evaluating
        self.model.eval()

        with torch.no_grad():
            avg_loss = 0
            avg_acc = 0
            for (x, y) in self.val_loader:
                x = x.permute((0, 3, 1, 2))
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)

                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                avg_loss += loss

                avg_acc += torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)) / y.shape[0]

            avg_loss /= len(self.val_loader)
            avg_acc /= len(self.val_loader)

            self.logger.add_scalar('val_loss', avg_loss, global_step=epoch)
            self.logger.add_scalar('accuracy', avg_acc, global_step=epoch)

            if best_val_loss > avg_loss:
                best_val_loss = avg_loss
                best_path = './ckpt/best_epoch.pth'

                cur_state = {
                    'last_step': self.global_step,
                    'last_model': self.model.state_dict()
                }
                torch.save(cur_state, best_path)  
    
    
    
    def test(self):
        print('Start testing on device {}'.format(device))
        self.model.eval()

        pred_labels = []

        with torch.no_grad():
            for x in self.test_loader:
                x = x.to(device=device, dtype=torch.float32)

                y_pred = self.model(x)
                pred_labels.append(torch.argmax(y_pred, dim=1).numpy())

        save_data = np.concatenate(pred_labels, axis=None)
        results = pd.Series(save_data, name="Label")

        os.makedirs(self.config.path.save_path, exist_ok=True)

        submission = pd.concat([pd.Series(range(1, len(save_data) + 1), name="ImageId"), results], axis=1)
        submission.to_csv(os.path.join(self.config.path.save_path, 'result.csv'), index=False)
    
    
    def _init_data(self,validation_split):
        data = ASLDataset(data_path= str(self.data_path), transform=self.transform)
        indices = int(np.floor((1-validation_split) * dataset_size)) 
        train_data = data[:indices]
        test_data = data[indices:] 
        train_loader = DataLoader(dataset = train_data,batch_size= self.batch_size)
        test_loader  = DataLoader(dataset= test_data,batch_size= self.batch_size)
        return train_loader, test_loader
        
        
    def _get_optim(self,optim):
        if str(optim) == "Adam":
            return Adam(
                self.model.parameters(),
                lr=self.lr,
            )
        return None
        
        