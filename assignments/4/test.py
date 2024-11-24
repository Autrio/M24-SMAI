
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
from tqdm.notebook import tqdm

import os
import cv2
from pathlib import Path
from icecream import ic
from models.CNN.data import Loader
import models.CNN.cnn as cnn
from models.CNN.metrics import Metrics



import sys
sys.path.append("/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.style.use('seaborn')
ic(np.__version__, device)

params = {
    'lr' : 1e-4,
    'batch_size': 256,
    'epoch': 100, 
    'use_dropout': True,
    'kernel_size': 3,
    'padding': 1,
    'task' : 'multi'
}


path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/double_mnist"
train_L,test_L,val_L = Loader(path,task=params['task'],batch_size = params['batch_size'])

single_batch = next(iter(train_L))
inputs, labels = single_batch

sample_tensor = inputs[0]
ic(sample_tensor.shape)

net = cnn.net(params)

def multi_class_net():
    #! 1 x 32 x 32
    net.add(net.Convlayer(1,32))
    #! 32 x 16 x 16
    net.add(net.Convlayer(32,64))
    net.add(net.drop())
    #! 64 x 8 x 8
    net.add(net.Convlayer(64,128))
    #! 128 x 4 x 4
    net.add(nn.Flatten())
    net.add(net.drop())
    net.add(nn.Linear(128*4*4,128*1*1))
    net.add(nn.Linear(128,10))
    net.add(nn.Sigmoid())

def single_class_net():
    net.add(net.Convlayer(1,32))
    net.add(net.Convlayer(32,64))
    net.add(net.drop())
    net.add(net.Convlayer(64,128))
    net.add(nn.Flatten())
    net.add(net.drop())
    net.add(nn.Linear(128*4*4,128*1*1))
    net.add(nn.Linear(128,64))
    net.add(nn.Linear(64,32))
    net.add(nn.Linear(32,3))
    net.add(nn.Softmax(dim=1))

def regression_net():
    net.add(net.Convlayer(1,32))
    net.add(net.Convlayer(32,64))
    net.add(net.drop())
    net.add(net.Convlayer(64,128))
    net.add(nn.Flatten())
    net.add(net.drop())
    net.add(nn.Linear(128*4*4,128*1*1))
    net.add(nn.Linear(128,64))
    net.add(nn.Linear(64,32))
    net.add(nn.Linear(32,1))
    
multi_class_net() if params['task'] == "multi" else single_class_net() if params['task'] == "single" else regression_net() if params['task'] == "regression" else exit()

ic(net)

metrics = Metrics(task=params['task'])

model = cnn.Model(model=net)
model.set_attr(loss_fn=metrics.loss_fn(),train_loader=train_L,val_loader=val_L,
               device=device,params=params,metrics=metrics,logger=None)

model.train()
model.save()





