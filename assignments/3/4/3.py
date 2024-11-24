import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import pickle

import models.MLP.MLP2 as mlp
from models.knn.knn import *
from performanceMeasures.macros import *


def run():
    model_file =  open('/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/autoencoder.pkl', 'rb')
    AE = pickle.load(model_file)

    print(AE)

    encoded_Xtrain = AE.encode(AE.Xtrain)
    encoded_Xtest = AE.encode(AE.Xtest)
    encoded_Xval = AE.encode(AE.Xval)

    encoded_Ytrain = AE.encode(AE.Ytrain)
    encoded_Ytest = AE.encode(AE.Ytest)
    encoded_Yval = AE.encode(AE.Yval)


    reconstructed_dataset = AE.decode(encoded_Xtrain)

    loss = mlp.loss.MSELoss()
    print("============== Reconstruction Error ===============")
    print(loss(AE.Xtrain,reconstructed_dataset))
    print("===================================================")

    knn = KNN()
    knn.Xtrain = np.array([x[:-1] for x in encoded_Xtrain])
    knn.Xtest =  np.array([x[:-1] for x in encoded_Xtest])
    knn.Xval =   np.array([x[:-1] for x in encoded_Xval])
    knn.Ytrain = np.array([y[-1] for y in encoded_Ytrain])
    knn.Ytest =  np.array([y[-1] for y in encoded_Ytest])
    knn.Yval =   np.array([y[-1] for y in encoded_Yval])

    knn.predict(disType="euclid",k=1,batch_size=100,optimized=True)

    performance = knn.inference()
    report = Macros(performance)

    print(report.macros(mode="macro",encoded=False))

if __name__ == "__main__":
    run()