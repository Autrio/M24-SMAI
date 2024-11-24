import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import pickle
from sklearn.preprocessing import OneHotEncoder

import models.MLP.MLP2 as mlp

def run():

    path = "./data/external/spotify.csv"

    params = {
    'lr' : 0.01,
    'batch_size':64,
    'epoch': 1000,
    'optimizer':"mini-batch",
    'loss_fn':mlp.loss.MSELoss(),
    'activation':mlp.activations.Tanh(),
    "type":"single_label_classification",
    "early_stopping":True,
    "patience":512,
    "logger":None
    }      

    model_file =  open('/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/autoencoder.pkl', 'rb')
    AE = pickle.load(model_file)

    print(AE)
    encoded_Xtrain = AE.encode(AE.Xtrain)
    encoded_Xtest = AE.encode(AE.Xtest)
    encoded_Xval = AE.encode(AE.Xval)

    model = mlp.Model(params)    
    model.Xtrain = np.array([x[:-1] for x in encoded_Xtrain])
    model.Xtest =  np.array([x[:-1] for x in encoded_Xtest])
    model.Xval =   np.array([x[:-1] for x in encoded_Xval])
    
    Y = AE.targets
    num_classes = np.unique(Y).shape[0]
    Y = Y.reshape(-1,1)

    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(Y)

    test_size = len(model.Xtest)
    val_size = len(model.Xval)
    train_size = len(model.Xtrain)

    total_size = test_size + val_size + train_size

    model.Ytest = Y[:test_size]
    model.Yval = Y[test_size:test_size + val_size]
    model.Ytrain = Y[test_size + val_size:]    

    architecture = mlp.Model.arch1
    activation_fn = mlp.activations.Tanh()

    model.set_arch(arch=architecture,activation=activation_fn,num_classes=num_classes)
    print(model)

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

if __name__ == "__main__":
    run()



