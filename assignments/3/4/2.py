import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import pickle

import models.MLP.MLP2 as mlp

def setup():
    params = {
    'lr' : 0.01,
    'batch_size':256,
    'epoch': 1000,
    'optimizer':"mini-batch",
    'loss_fn':mlp.loss.MSELoss(),
    'activation':mlp.activations.Tanh(),
    "type":"regression"
    }      

    batch_training = 'batched' if params['batch_size'] == 512 else ('SGD' if params['batch_size'] == 1 else 'mini-batch')
    # wandb.init(
    # project="SMAI-A3-MLP-regression",
    # config=params,
    # name=f"{params['activation']}|{batch_training}")

    params["logger"] = None

    return params

def run(params):
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/spotify.csv"

    AE = mlp.Autoencoder(params,encoding_dim=7)
    AE.load_raw_data(path=path)
    AE.Preprocess()
    AE.Split(test=0.1,valid=0.1)
    AE.Normalize()
    AE.set_arch()
    print(AE)
    AE.train(epochs=params["epoch"])

    with open('autoencoder.pkl', 'wb') as model_file:
        pickle.dump(AE, model_file)
    

    model_file =  open('/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/autoencoder.pkl', 'rb')
    AE = pickle.load(model_file)

    print(AE)

    encoded_dataset = AE.encode(AE.Xtrain)

    reconstructed_dataset = AE.decode(encoded_dataset)

    loss = mlp.loss.MSELoss()
    print("============== Reconstruction Error ============")
    print(loss(AE.Xtrain,reconstructed_dataset))


if __name__=="__main__":
    run(setup())