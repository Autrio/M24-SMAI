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
    'lr' : 0.001,
    'batch_size':128,
    'epoch': 5000,
    'optimizer':"mini-batch",
    'loss_fn':mlp.loss.BCELoss(),
    'activation':mlp.activations.Tanh(),
    "type":"multi_label_classification",
    "early_stopping":True
    }      

    batch_training = 'batched' if params['batch_size'] == 512 else ('SGD' if params['batch_size'] == 1 else 'mini-batch')
    wandb.init(
    project="SMAI-A3-MLP-MultiLabel",
    config=params,
    name=f"{params['activation']}|{batch_training}")

    params["logger"] = wandb

    return params

def run(params):
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/advertisement.csv"
    model = mlp.Model(params)
    model.load_raw_data(path=path)
    model.Preprocess(categorical=['gender', 'education', 'married','city', 'occupation', 'most bought item'],drop=None,multi_hot=["labels"])
    print(model.df.columns)
    model.Split(targ="labels",task=params["type"])
    model.Normalize()

    architecture = mlp.Model.arch1
    activation_fn = mlp.activations.Tanh()

    model.set_arch(arch=architecture,activation=activation_fn)

    model.train(epochs=params["epoch"])
    print(model.all_labels)
    metrics = mlp.Metrics()
    ground_truth = model.Ytest
    predictions = model.predict(model.Xtest)
    print("================Test set metrics======================")
    print()
    print("accuracy :: ", metrics.accuracy_score(ground_truth,predictions))
    print("precision :: ", metrics.precision_score(ground_truth,predictions))
    print("recall :: ", metrics.recall_score(ground_truth,predictions))
    print("F1-score :: ", metrics.f1_score(ground_truth,predictions,average="macro"))
    print("Hamming Loss :: ", metrics.hamming_loss(ground_truth,predictions))
    print()
    print("======================================================")

    print()


if __name__=="__main__":
    run(setup())