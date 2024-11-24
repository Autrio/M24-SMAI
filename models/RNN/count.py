import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader, TensorDataset


def loader(data,split_ratio = 0.8,batch_size = 64):
    bits,labels = data

    indices = np.random.permutation(len(bits))
    train_size = int(split_ratio * len(bits))
    val_size = int(0.1 * len(bits))
    test_size = len(bits) - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    x_train = bits[train_indices]
    y_train = labels[train_indices]

    x_val = bits[val_indices]
    y_val = labels[val_indices]

    x_test = bits[test_indices]
    y_test = labels[test_indices]

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    batch_size = batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader,test_loader

class model(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=2, dropout=0.2, normalize=True):
        super(model, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.normalize = normalize
        if self.normalize:
            self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(device)  # Move model to device
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  
        if self.normalize:
            out = self.bn(out)
        out = self.fc(out)
        return out

    def train_model(self, train_loader, val_loader, epochs=10):
        loss_fn = nn.MSELoss() 
        new_loss = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=0.001) 
        scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
        tl = []
        vl = []

        for epoch in tqdm(range(epochs), desc="Training", unit="epoch",leave=True):  
            self.train()  
            loss_total = 0.0
            train_mae = 0.0

            for sequences, label in tqdm(train_loader,leave=False,desc="[train] processing batches",unit="batch"):
                sequences, label = sequences.to(device), label.to(device)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():  # Mixed precision
                    op = self(sequences)
                    loss = loss_fn(op, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_total += loss.item()
                train_mae += new_loss(op, label).item()

            val_loss,val_mae,rloss = self.evaluate_model(val_loader)
            tqdm.write(f"Epoch :: [{epoch+1}/{epochs}] || Train ::[Loss: {loss_total/len(train_loader):.4f} MAE: {train_mae/len(train_loader)}] || Validation :: [Loss: {val_loss} MAE: {val_mae:.4f}] || Random Baseline MSE: {rloss:.4f}")
            tl.append(loss_total/len(train_loader))
            vl.append(val_loss)

        return tl, vl

    def random_baseline(self, sequences):
        counts = [np.random.randint(0, len(seq) + 1) for seq in sequences]
        return np.array(counts)

    def evaluate_model(self, val_loader):
        self.eval()  
        all_sequences = []
        all_labels = []
        all_predictions = []
        loss_fn = nn.MSELoss()
        new_loss = nn.L1Loss()
        loss_total = 0.0
        val_mae = 0.0
        with torch.no_grad():  
            for sequences, labels in tqdm(val_loader,leave=False,desc="[validation] processing batches",unit="batch"): 
                sequences, labels = sequences.to(device), labels.to(device)
                predictions = self(sequences)
                loss = loss_fn(predictions, labels)
                val_mae += new_loss(predictions, labels).item() 
                loss_total += loss.item()

                all_sequences.extend(sequences.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())  
                all_predictions.extend(predictions.cpu().numpy())  

        random_base = self.random_baseline(all_sequences)
        random_baseline_loss = np.mean((np.array(all_predictions) - random_base) ** 2)        

        return loss_total/len(val_loader),val_mae/len(val_loader), random_baseline_loss
