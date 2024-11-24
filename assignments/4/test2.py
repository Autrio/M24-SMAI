
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
from tqdm.notebook import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from sklearn.datasets import fetch_openml
from scipy.io import loadmat

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"

plt.style.use('seaborn')
np.__version__, device

from models.Autoencoders.data import *
import models.Autoencoders.cnnAutoencoder as autoencoder



params = {
    'lr' : 1e-3,
    'batch_size': 512,
    'epoch': 20, 
    'use_dropout': True,
    'kernel_size': 3,
    'padding': 1,
    'task' : 'single',
    'latent_dims':7
}


train_loader,test_loader = Loader(params=params)


single_batch = next(iter(test_loader))
inputs, labels = single_batch

sample_tensor = inputs[0]
print(sample_tensor.shape)


net = autoencoder.Autoencoder(params=params)


metrics = None

model = autoencoder.Model(model=net)
model.set_attr(loss_fn=nn.MSELoss(),train_loader=train_loader,val_loader=test_loader,
               device=device,params=params,metrics=metrics,logger=None)


model.train()


model.save(path="./cnnAutoencoder.pth")


