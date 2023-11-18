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

class TwoStreamCNNrunner():
    def __init__(self,config, logger, transform = None):
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.image_size = config.data.image_size 
        
        
        self.data_path = config.path.raw_path
        
        self.criteria = nn.CrossEntropyLoss()
        self.lr = config.train.lr
        self.transform = transform
        self.model = TwoStreamCNN(type = "tsma")
        self._init_data(0.2)
        
        
    def train(self):
        pass
    
    
    
    def test(self):
        pass
    
    
    
    def _init_data(self,validation_split):
        data = ASLDataset(data_path= str(self.data_path), transform=self.transform)
        indices = int(np.floor((1-validation_split) * dataset_size)) 
        train_data = data[:indices]
        test_data = data[indices:] 
        self.train_loader = DataLoader(dataset = train_data,batch_size= self.batch_size)
        self.test_loader  = DataLoader(dataset= test_data,batch_size= self.batch_size)
        
        
        
    def _get_optim(self,optim):
        if str(optim) == "Adam":
            return Adam(
                self.model.parameters(),
                lr=self.lr,
            )
        return None
        
        