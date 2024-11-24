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
    'lr' : 0.036847,
    'batch_size':512,
    'epoch': 2000,
    'optimizer':"mini-batch",
    'loss_fn':mlp.loss.CELoss(),
    'activation':mlp.activations.Linear(),
    "type":"single_label_classification",
    "early_stopping":True,
    "patience":512
    }      

    batch_training = 'batched' if params['batch_size'] == 512 else ('SGD' if params['batch_size'] == 1 else 'mini-batch')
    wandb.init(
    project="SMAI-A3-MLP",
    config=params,
    name=f"{params['activation']}|{batch_training}")

    params["logger"] = wandb

    return params

def run(params):
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/WineQT.csv"
    model = mlp.Model(params)
    model.load_raw_data(path=path)
    model.Preprocess()
    model.Split(task="single_label_classification")
    model.Normalize()

    architecture = mlp.Model.arch5
    activation_fn = mlp.activations.Tanh()

    model.set_arch(arch=architecture,activation=activation_fn)
    print(model.Xtrain[0].shape)

    model.train(epochs=params["epoch"])
    predictions = model.predict(model.Xtest)
    ground_truth = model.Ytest
    metrics = mlp.Metrics()
    print("================Test set metrics======================")
    print()
    print("accuracy :: ", metrics.accuracy_score(ground_truth,predictions))
    print("precision :: ", metrics.precision_score(ground_truth,predictions))
    print("recall :: ", metrics.recall_score(ground_truth,predictions))
    print("F1-score :: ", metrics.f1_score(ground_truth,predictions,average="macro"))
    print()
    print("======================================================")


if __name__=="__main__":
    run(setup())