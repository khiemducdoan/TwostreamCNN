import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.HandwrittenDigitDataset import HandwrittenDigitDataset
from models.RandomNet import RandomNet
from utils.utils import get_device

device = get_device()


class RandomNetRunner:
    def __init__(self, config, logger):
        # config
        self.config = config

        # logger
        self.logger = logger
        self.global_step = 0

        # data
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.image_size = config.data.image_size

        # train
        self.epoch = config.train.epoch
        self.lr = config.train.lr

        # model
        self.model = RandomNet()

        # optimizer
        self.optimizer = self.get_optimizer(config.train.optimizer)

        # criterion
        self.criterion = nn.CrossEntropyLoss()

        # data_loader
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # initialize data
        self.init_data()

        # load checkpoint

    def get_optimizer(self, optim):
        if str(optim) == "Adam":
            return Adam(
                self.model.parameters(),
                lr=self.lr,
            )
        return None

    def split_data(self, data):
        x = data.iloc[:, 1:]
        x = x.to_numpy().reshape(-1, self.image_size, self.image_size, 1)

        y = data.iloc[:, 0].values
        y = np.eye(10)[y]

        # print(y.shape, y)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
        return x_train, x_val, y_train, y_val

    def init_data(self):
        # read data from csv
        raw_data = pd.read_csv(self.config.path.raw_path)
        blur_data = pd.read_csv(self.config.path.blur_path)
        gaussian_data = pd.read_csv(self.config.path.gaussian_path)
        random_transform_data = pd.read_csv(self.config.path.random_transform_path)

        eval_data = pd.read_csv(self.config.path.eval_path)

        # mix data
        tot_data = pd.concat([raw_data, blur_data], axis=0)
        tot_data = pd.concat([tot_data, gaussian_data], axis=0)
        tot_data = pd.concat([tot_data, random_transform_data], axis=0)
        tot_data = tot_data.sample(frac=1).reset_index(drop=True)  # shuffle tot_data

        print(tot_data.shape)

        # spit data for training
        x_train, x_val, y_train, y_val = self.split_data(tot_data)

        # reshape eval_data
        eval_data = eval_data.to_numpy().reshape(-1, self.image_size, self.image_size, 1)

        # create data_loaders
        train_data = HandwrittenDigitDataset(x_train, y_train)
        val_data = HandwrittenDigitDataset(x_val, y_val)
        test_data = HandwrittenDigitDataset(eval_data, None)

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,  # use GPU to load data (num_workers > 0)
            drop_last=True,  # Drop some last sample
        )

        self.val_loader = DataLoader(
            dataset=val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # use GPU to load data (num_workers > 0)
            drop_last=False,
        )

        self.test_loader = DataLoader(
            dataset=test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # use GPU to load data (num_workers > 0)
            drop_last=False,
        )

    def train(self):
        print('Start training on device {}'.format(device))
        self.model = self.model.to(device)
        best_val_loss = 1e9

        for epoch in range(self.epoch):
            print('Start epoch {}'.format(epoch))

            # set up model state to training
            self.model.train()

            for (x, y) in self.train_loader:
                x = x.permute((0, 3, 1, 2))  # reshape x to (batch, channels, height, width)
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
                x = x.permute((0, 3, 1, 2))
                x = x.to(device=device, dtype=torch.float32)

                y_pred = self.model(x)
                pred_labels.append(torch.argmax(y_pred, dim=1).numpy())

        save_data = np.concatenate(pred_labels, axis=None)
        results = pd.Series(save_data, name="Label")

        os.makedirs(self.config.path.save_path, exist_ok=True)

        submission = pd.concat([pd.Series(range(1, len(save_data) + 1), name="ImageId"), results], axis=1)
        submission.to_csv(os.path.join(self.config.path.save_path, 'result.csv'), index=False)