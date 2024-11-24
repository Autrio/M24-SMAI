import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")


def generate_bits(num_sequences = 100000,min_length=1,max_length=16):
    sequences = []
    labels = []

    for _ in range(num_sequences):
        seq_length = np.random.randint(min_length, max_length + 1)  
        sequence = np.random.randint(0, 2, seq_length).tolist()  
        count_of_ones = sum(sequence) 
        sequences.append(sequence)
        labels.append(count_of_ones)


    max_seq_length = max_length
    bits_padded = np.array([np.pad(seq, (0,max_seq_length - len(seq)), 'constant', constant_values=0) for seq in sequences])
    labels = np.array(labels)
    bits_padded = bits_padded.reshape((num_sequences,max_length,1))
    labels =  labels.reshape((num_sequences,1))

    return bits_padded,labels

def run():
    mse = []
    with open("./assignments/5/4/1/model.pkl", "rb") as f:
        model = pickle.load(f)
        for i in range(1,33):
            sequences,labels = generate_bits(1000,i,i)
            gen_train_tensor = torch.tensor(sequences, dtype=torch.float32)
            gen_target_tensor = torch.tensor(labels, dtype=torch.float32)    
            gen_dataset = TensorDataset(gen_train_tensor,gen_target_tensor)
            gen_loader = DataLoader(gen_dataset,batch_size=64,shuffle=True)
            me,_,_ = model.evaluate_model(gen_loader) 
            mse.append(me)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 33), mse, marker='o', linestyle='-', color='b')
    plt.xlabel('Sequence Length')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE vs Sequence Length')
    plt.grid(True)
    plt.savefig("./assignments/5/figures/MSEvsLen.png")

if __name__ == "__main__":
    run()