import os
import pathlib
import torch
import PIL
from PIL import Image
from utils import get_class
from torch.utils.data import Dataset


class ASLDataset(Dataset):
    def __init__(self, 
                 data_path,
                 transform):
        self.img_path = list(pathlib.Path(data_path).glob('*/*.jpg'))
        self.class_name,self.class_to_idx = utils.get_class(data_dir)
        
    def load_image(self,
                   index):
        path = self.img_path[index]
        return Image.open(path)
    
    def __getitem__(self,
                    idx):
        img = self.load_image(idx)
        class_name = self.img_path[idx].parents.name
        class_name_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(img),class_name_idx
        else: 
            return img, class_name_idx
    
    def __len__(self):
        len(self.img_path) 
        