from dataset.dataset import ASLDataset
from model.TwoStreamCNN import TwoStreamCNN
import torch
from torch import nn
import numpy as np
from torch.optim import Adam
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import utils
device = utils.get_device()

class TwoStreamCNNrunner():
    def __init__(self,config, logger, transform = None):
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.image_size = config.data.image_size 
        
        
        self.train_path = str(config.path.train_path)
        self.test_path = str(config.path.test_path)
        self.lr = config.train.lr
        
        self.epoch = config.train.epoch
        self.criterion = nn.CrossEntropyLoss()
        self._model = TwoStreamCNN(29,type = "tsma")
        self.optimizer = self._get_optim(config.train.optimizer)
        
        
        self.transform = transform
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self._init_data()
        
        # load checkpoint
        if 'ckpt_path' in config.path:
            if config.path.ckpt_path is not None:
                last_state = torch.load(config.path.ckpt_path)
                self.global_step = last_state['last_step']
                self.model.load_state_dict(last_state['last_model'])
    
    def _init_data(self):
        train_data = ASLDataset(data_path= (self.train_path), transform=self.transform)
        test_data = ASLDataset(data_path= (self.test_path), transform=self.transform)
        val_data = ASLDataset(data_path = self.val_loader,transform=self.transform)
        self.train_loader = DataLoader(dataset = train_data,
                                  batch_size= self.batch_size,
                                  num_workers= self.num_workers
                                  )
        self.test_loader  = DataLoader(dataset=test_data,
                                  batch_size= self.batch_size,
                                  num_workers= self.num_workers)
        self.val_loader = DataLoader(dataset=val_data,
                                     batch_size = self.batch_size,
                                     num_workers = self.num_workers)
        
        
    def _get_optim(self,optim):
        if str(optim) == "Adam":
            return Adam(self._model.parameters(),lr=self.lr,)
        return None
    def train(self):
        print('Start training on device {}'.format(device))

        self.model = self.model.to(device)
        best_val_loss = 1e9

        for epoch in range(self.epoch):
            print('Start epoch {}'.format(epoch))

            # set up model state to training
            self.model.train()

            for (x,XN, y) in self.train_loader:
                XN = XN.to(device=device)
                x = x.to(device=device)
                y = y.to(device=device)

                y_pred = self.model(x,XN)

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
                for (x,XN, y) in self.val_loader:
                    XN = XN.to(device=device)
                    x = x.to(device=device)
                    y = y.to(device=device)

                    y_pred = self.model(x,XN)
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
            for x,XN,y in self.test_loader:
                XN = XN.to(device=device)
                x = x.to(device=device)

                y_pred = self.model(x,XN)
                pred_labels.append(torch.argmax(y_pred, dim=1).numpy())

        save_data = np.concatenate(pred_labels, axis=None)
        results = pd.Series(save_data, name="Label")

        os.makedirs(self.config.path.save_path, exist_ok=True)

        submission = pd.concat([pd.Series(range(1, len(save_data) + 1), name="ImageId"), results], axis=1)
        submission.to_csv(os.path.join(self.config.path.save_path, 'result.csv'), index=False)
        
        