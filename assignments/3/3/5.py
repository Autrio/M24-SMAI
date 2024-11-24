import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic
import seaborn as sns
import wandb

import models.MLP.MLP2 as mlp

def setup1():
    params = {
    'lr' : 0.01,
    'batch_size':128,
    'epoch': 4000,
    'optimizer':"mini-batch",
    'loss_fn':mlp.loss.MSELoss(),
    'activation':mlp.activations.Tanh(),
    "type":"regression",
    "early_stopping":True
    }      

    batch_training = 'batched' if params['batch_size'] == 512 else ('SGD' if params['batch_size'] == 1 else 'mini-batch')
    # wandb.init(
    # project="SMAI-A3-MLP-regression",
    # config=params,
    # name=f"{params['activation']}|{batch_training}")

    params["logger"] = None

    return params

def run1(params):
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/diabetes.csv"
    model = mlp.Model(params)
    model.load_raw_data(path=path)
    model.Preprocess(drop=None,nan=None)
    model.Split(targ="Outcome",task="regression")
    model.Normalize()

    model.add(mlp.network.Layer(model.Xtrain.shape[1], 16, mlp.activations.Tanh()))
    model.add(mlp.network.Layer(16, 32, mlp.activations.Tanh()))
    model.add(mlp.network.Layer(32, 16, mlp.activations.Tanh()))
    model.add(mlp.network.Layer(16,8, mlp.activations.Tanh()))
    model.add(mlp.network.Layer(8, 1, mlp.activations.Sigmoid()))

    

    print(model)
    model.train(epochs=params["epoch"])

    return model.history


def setup2():
    params = {
    'lr' : 0.01,
    'batch_size':128,
    'epoch': 4000,
    'optimizer':"mini-batch",
    'loss_fn':mlp.loss.BCELoss(),
    'activation':mlp.activations.Tanh(),
    "type":"regression",
    "early_stopping":True
    }      

    batch_training = 'batched' if params['batch_size'] == 512 else ('SGD' if params['batch_size'] == 1 else 'mini-batch')
    # wandb.init(
    # project="SMAI-A3-MLP-regression",
    # config=params,
    # name=f"{params['activation']}|{batch_training}")

    params["logger"] = None

    return params

def run2(params):
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/diabetes.csv"
    model = mlp.Model(params)
    model.load_raw_data(path=path)
    model.Preprocess(drop=None,nan=None)
    model.Split(targ="Outcome",task="regression")
    model.Normalize()

    model.add(mlp.network.Layer(model.Xtrain.shape[1], 16, mlp.activations.Tanh()))
    model.add(mlp.network.Layer(16, 32, mlp.activations.Tanh()))
    model.add(mlp.network.Layer(32, 16, mlp.activations.Tanh()))
    model.add(mlp.network.Layer(16,8, mlp.activations.Tanh()))
    model.add(mlp.network.Layer(8, 1, mlp.activations.Sigmoid()))

    

    print(model)
    model.train(epochs=params["epoch"])

    return model.history

if __name__=="__main__":
    history1 = run1(setup1())
    history2 = run2(setup2())

    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.plot(history1['train_loss'])
    plt.plot(history1['val_loss'])
    plt.title('Training Loss Over Epochs for MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(history2['train_loss'])
    plt.plot(history2['val_loss'])
    plt.title('Training Loss Over Epochs for BCE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()

    plt.savefig("/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/assignments/3/figures/3/MSEvsBCE")


