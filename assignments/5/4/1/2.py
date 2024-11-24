import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models.RNN.count as count
import matplotlib.pyplot as plt
import warnings
from icecream import ic
import pickle

warnings.filterwarnings("ignore")


bits = np.load("./data/interim/5/4/bits_padded.npy")
labels = np.load("./data/interim/5/4/labels.npy")


def run():
    train_loader,val_loader,test_loader = count.loader((bits,labels))
    model = count.model(input_size=1, hidden_size=128, output_size=1, num_layers=3, dropout=0.5, normalize=False)

    tl, vl = model.train_model(train_loader, val_loader, epochs=20)

    with open("./assignments/5/4/1/model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    plt.figure(figsize=(10, 5))
    plt.plot(tl, label="Training Loss")
    plt.plot(vl, label="Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("./assignments/5/figures/loss-basic.png")

if __name__ == "__main__":
    run()