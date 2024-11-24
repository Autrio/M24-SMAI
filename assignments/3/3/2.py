import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

import models.MLP.MLP2 as mlp

def setup():
    params = {
    'lr' : 0.0001,
    'batch_size':128,
    'epoch': 4000,
    'optimizer':"mini-batch",
    'loss_fn':mlp.loss.MSELoss(),
    'activation':mlp.activations.Tanh(),
    "type":"regression",
    "early_stopping":True
    }      

    batch_training = 'batched' if params['batch_size'] == 512 else ('SGD' if params['batch_size'] == 1 else 'mini-batch')
    wandb.init(
    project="SMAI-A3-MLP-regression",
    config=params,
    name=f"{params['activation']}|{batch_training}")

    params["logger"] = wandb

    return params

def run(params):
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/HousingData.csv"
    model = mlp.Model(params)
    model.load_raw_data(path=path)
    model.Preprocess(drop=None,nan=['CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT'])
    model.Split(targ="MEDV",task="regression")
    model.Normalize()

    architecture = mlp.Model.arch1
    activation_fn = params["activation"]

    model.set_arch(arch=architecture,activation=activation_fn)
    print(model)
    model.train(epochs=params["epoch"])

    metrics = mlp.Metrics()
    predictions = model.predict(model.Xtest,task="regression")
    ground_truth = model.Ytest.reshape(-1,1)


    print("================Test set metrics======================")
    print()
    print("R2 Score :: ", metrics.r2_score(ground_truth,predictions))
    print("MSE :: ", metrics.mse(ground_truth,predictions))
    print("RMSE :: ", metrics.rmse(ground_truth,predictions))
    print("MAE :: ", metrics.mae(ground_truth,predictions))
    print()
    print("======================================================")



if __name__=="__main__":
    run(setup())