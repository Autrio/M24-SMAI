import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
from tqdm.auto import tqdm
# from tqdm import tqdm #!used for direct testing

import os
import cv2
from pathlib import Path
from icecream import ic
import sys

class InvalidTaskException(Exception):
    def __init__(self,message):
        self.message = message
        super().__init__(self.message)

class Data:
    def __init__(self):
        self.df = pd.DataFrame()
        
    def load_raw_data(self,path):
        df = []
        for split in ['test', 'train', 'val']:
            files, outcome = [], []
    
            for folder in tqdm(os.listdir(os.path.join(path, split))):
                for file in os.listdir(os.path.join(path, split, folder)):
                    files.append(os.path.join(path, split, folder, file))
                    outcome.append(folder)
                    

            df.append(pd.DataFrame({
                'filename': files,
                'outcome': outcome
            }))
            
        print(f"Data Split into {df[0].shape} test samples, {df[1].shape} training samples and {df[2].shape} Validation samples")
        self.test_df = df[0]
        self.train_df = df[1]
        self.val_df = df[2]
        self.df = pd.concat(df,axis=0)
        return self.df


    def load_image(self,file):
        img = plt.imread(file)
        img = cv2.resize(img, (32, 32))
        return img


class MNISTDataset(Data):
    def __init__(self,task):
        super().__init__()
        self.transform = tf.Compose([tf.ToTensor(),tf.Normalize(0, 1)])
        if task not in ["single","multi","regression"]:
            raise InvalidTaskException("The specified task is either empty or invlaid")
        self.task = task
    
    def Split(self,split="train"):
        self.split = split
        self.df = self.test_df if self.split=="test" else self.train_df if self.split=="train" else self.val_df 

    def load_image(self, file):
        return super().load_image(file)
    
    def load_raw_data(self, path):
        return super().load_raw_data(path)
        
    def __len__(self):
        self.n_samples = len(self.df)
        return self.n_samples

    def __getitem__(self, idx):
        x = self.load_image(self.df.iloc[idx, 0])
        if self.transform:
            x = self.transform(x)
            
        y = self.df.iloc[idx, 1]
        
        if self.task!="multi":
            y_out = len(y)
            return x, torch.tensor(y_out)
        else:
            y_one_hot = np.zeros(10)
            for digit in y:
                y_one_hot[int(digit)] = 1
            return x, torch.tensor(y_one_hot)
    

def Loader(path,task="single",batch_size=256):
    datasets = [MNISTDataset(task=task),MNISTDataset(task=task),MNISTDataset(task=task)]
    splits = ["test","train","val"]

    for dataset,split in zip(datasets,splits):
        dataset.load_raw_data(path)
        dataset.Split(split)
    

    train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets[2], batch_size=batch_size, shuffle=False)

    return train_loader,test_loader,val_loader

        