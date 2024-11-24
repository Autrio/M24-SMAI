import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

import models.MLP.MLP2 as mlp

def setup():
    sweep_config = {
        "method": "bayes",
        "name": "classification Sweep",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            "batch_size": {"values": [32, 64, 128, 256, 512]},
            "epoch": {"values": [1000, 2000, 3000, 4000, 5000]},
            "loss_fn": {"values": ["MSELoss", "CELoss","BCELoss"]},
            "activations": {"values": ["Linear", "Relu", "Sigmoid","Tanh"]},
            "type": {"values": ["single_label_classification"]},
            "lr": {"max": 0.1, "min": 0.0001},
            "model_architecture": {
                "values": ["arch1", "arch2", "arch3", "arch4", "arch5"]
            }
        }
    }      

    params = {
        'lr' : 1e-3,
        'batch_size':256,
        'epoch': 1000,
        'optimizer':"mini-batch",
        'loss_fn':mlp.loss.MSELoss(),
        'activation':mlp.activations.Tanh(),
        "type":"single_label_classification"
    }               

    batch_training = 'batched' if params['batch_size'] == 512 else ('SGD' if params['batch_size'] == 1 else 'mini-batch')
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="SMAI-A3-MLP")


    return sweep_id


def run():

    params = {
        'lr' : 1e-3,
        'batch_size':256,
        'epoch': 1000,
        'optimizer':"mini-batch",
        'loss_fn':mlp.loss.MSELoss(),
        'activation':mlp.activations.Tanh(),
        "type":"single_label_classification",
        "early_stopping":True
    }


    with wandb.init(config=params):
        config = wandb.config
        path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/WineQT.csv"
        
        # Mapping strings back to actual classes
        loss_fn_mapping = {
            "MSELoss": mlp.loss.MSELoss(),
            "BCELoss": mlp.loss.BCELoss(),
            "CELoss": mlp.loss.CELoss()
        }
        
        activation_fn_mapping = {
            "Linear": mlp.activations.Linear,
            "Relu": mlp.activations.Relu,
            "Sigmoid": mlp.activations.Sigmoid,
            "Softmax": mlp.activations.Softmax,
            "Tanh": mlp.activations.Tanh
        }
        
        architecture_mapping = {
            "arch_base": mlp.Model.arch_base,
            "arch1": mlp.Model.arch1,
            "arch2": mlp.Model.arch2,
            "arch3": mlp.Model.arch3,
            "arch4": mlp.Model.arch4,
            "arch5": mlp.Model.arch5
        }

        loss_fn = loss_fn_mapping[config.loss_fn]
        activation_fn = activation_fn_mapping[config.activations]()
        architecture = architecture_mapping[config.model_architecture]


        model = mlp.Model(config)
        model.loss_fn = loss_fn

        model.load_raw_data(path=path)
        model.Preprocess()
        model.Split(task="single_label_classification")
        model.Normalize()

        model.set_arch(arch=architecture, activation=activation_fn)
        print(model)
        model.train(epochs=config.epoch)
        metrics = mlp.Metrics()
        predictions = model.predict(model.Xtest)
        ground_truth = model.Ytest

        print("================Test set metrics======================")
        print()
        print("accuracy :: ", metrics.accuracy_score(ground_truth,predictions))
        print("precision :: ", metrics.precision_score(ground_truth,predictions))
        print("recall :: ", metrics.recall_score(ground_truth,predictions))
        print("F1-score :: ", metrics.f1_score(ground_truth,predictions,average="macro"))
        print()
        print("======================================================")


if __name__=="__main__":
    sweep_id = setup()
    wandb.agent(sweep_id=sweep_id,function=run)