from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models.RNN.count as count
import matplotlib.pyplot as plt
import warnings
from icecream import ic
from tqdm import tqdm

warnings.filterwarnings("ignore")

hidden_sizes = [64, 128]
output_sizes = [1]
num_layers_list = [2, 3]
dropout_rates = [0.5,0.8]
normalize_options = [True, False]
bits = np.load("./data/interim/5/4/bits_padded.npy")
labels = np.load("./data/interim/5/4/labels.npy")

def run():
    hyperparameter_combinations = list(product(hidden_sizes, output_sizes, num_layers_list, dropout_rates, normalize_options))
    train_loader,val_loader,test_loader = count.loader((bits,labels))

    for hidden_size, output_size, num_layers, dropout, normalize in tqdm(hyperparameter_combinations,desc="Hyperparameters",total=len(hyperparameter_combinations),leave=True):
        tqdm.write("="*200)
        tqdm.write(f"Training model with hidden_size={hidden_size}, output_size={output_size}, num_layers={num_layers}, dropout={dropout}, normalize={normalize}")
        tqdm.write("_"*200)
        
        model = count.model(input_size=1, hidden_size=hidden_size, output_size=output_size, 
                        num_layers=num_layers, dropout=dropout, normalize=normalize)
        
        tl, vl = model.train_model(train_loader, val_loader, epochs=10)

        tqdm.write("="*200)

        
        plt.figure(figsize=(10, 5))
        plt.plot(tl, label="Training Loss")
        plt.plot(vl, label="Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f"hidden_size={hidden_size}, output_size={output_size}, num_layers={num_layers}, dropout={dropout}, normalize={normalize}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./assignments/5/figures/LOSS_paramcomb_{hidden_size}_{output_size}_{num_layers}_{dropout}_{normalize}.png")

if __name__ == "__main__":
    run()