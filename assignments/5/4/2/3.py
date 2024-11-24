import models.RNN.ocr as ocr 
import warnings
from icecream import ic
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


warnings.filterwarnings("ignore")
hidden_sizes = [64, 128]
output_sizes = [1]
num_layers_list = [2, 3]
dropout_rates = [0.5,0.8]
normalize_options = [True, False]
bits = np.load("./data/interim/5/4/bits_padded.npy")
labels = np.load("./data/interim/5/4/labels.npy")

path = "./data/interim/5/4/nltk"

train_loader,val_loader,char_set,max_len = ocr.loader(path,ratio=0.8,batch_size=32,percentage=0.1)  
ic(max_len)

model = ocr.model(n_hidden=256,max_length=max_len,char_set=char_set,lr=0.001)

ocr.train(model,train_loader,val_loader,epochs=10)

def run():
    hyperparameter_combinations = list(product(hidden_sizes, output_sizes, num_layers_list, dropout_rates, normalize_options))

    for hidden_size, output_size, num_layers, dropout, normalize in tqdm(hyperparameter_combinations,desc="Hyperparameters",total=len(hyperparameter_combinations),leave=True):
        tqdm.write("="*200)
        tqdm.write(f"Training model with hidden_size={hidden_size}, output_size={output_size}, num_layers={num_layers}, dropout={dropout}, normalize={normalize}")
        tqdm.write("_"*200)
        
        model = ocr.model()
        
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
