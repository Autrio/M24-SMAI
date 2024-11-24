from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import numpy as np
torch.autograd.set_detect_anomaly(True)
from tqdm import tqdm
from icecream import ic

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.type(torch.float32)),
    transforms.Normalize((0.5,), (0.5,))
])

class WordImageDataset(Dataset):
    def __init__(self, image_dir,charset,transform=None,percentage=1.0):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.image_files = self.image_files[:int(percentage*len(self.image_files))]
        self.transform = transform
        self.max_length = max(len(f.split('.')[0]) for f in self.image_files)
        self.charset = charset

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB').resize((256, 64))
        image = np.array(image).mean(axis=2)  
        if self.transform:
            image = self.transform(image)
        label = self.image_files[idx].split('.')[0]
        label = [self.charset.index(char) for char in label]  
    
        if any(c < 0 or c >= len(self.charset) for c in label):
            raise ValueError(f"Invalid label {label} in file {self.image_files[idx]}")
        
        label = torch.tensor(label, dtype=torch.long)
        if len(label) < self.max_length:
            label = torch.cat([label, torch.full((self.max_length - len(label),), len(self.charset), dtype=torch.long)])
        return image, label


# path = "./data/interim/5/4/nltk"
def loader(path,ratio=0.8,batch_size=32,percentage=1.0):
    """
    path : str : path to the directory containing the images
    ratio : float : ratio of training data to total data    
    batch_size : int : batch size for training
    percentage : float : percentage of data to be used
    
    RETURNS:
    train_loader : DataLoader : training data loader
    val_loader : DataLoader : validation data loader
    char_set : list : list of unique characters
    max_length : int : maximum length of the labels
    """
    unique_chars = set()
    for filename in tqdm(os.listdir(path),desc="forming Characterset",unit="file"):
        if filename.endswith('.png'):
            unique_chars.update(filename.split('.')[0])

    char_set = list(unique_chars)

    dataset = WordImageDataset(path,charset=char_set,transform=transform,percentage=percentage)
    print("length of dataset :: ",len(dataset))

    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader,char_set,dataset.max_length


class model(nn.Module):
    def __init__(self, n_hidden, max_length,char_set,lr=0.001):
        super(model, self).__init__()
        self.max_length = max_length
        self.charset = char_set
        self.num_classes=len(char_set) + 1

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(256 * 4 * 16, 128)
        self.rnn = nn.RNN(
            input_size=128,
            hidden_size=n_hidden,
            num_layers=2,
            batch_first=True,
            nonlinearity='relu',
            dropout=0.3
        )
        self.fc_out = nn.Linear(n_hidden, self.num_classes)

        self.criterion = nn.CrossEntropyLoss(ignore_index=len(char_set))
        self.optimizer = torch.optim.Adam(self.parameters(),lr)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = x.unsqueeze(1).expand(-1, self.max_length, -1)
        x, _ = self.rnn(x)
        x = self.fc_out(x)
        return x
    
def train(model, train_loader, val_loader, epochs=10, device="gpu"):
    device = torch.device('cuda' if torch.cuda.is_available() and device == "gpu" else 'cpu')
    model.to(device)  
    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm(range(epochs), desc="epoch", position=0, leave=True):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for images, labels in tqdm(train_loader, desc="Training", unit="batch", position=1, leave=False):
            images, labels = images.to(device), labels.to(device)
            model.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = model.criterion(outputs.permute(0, 2, 1), labels)

            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()

            train_loss += loss.item()
            train_acc += accuracy(outputs, labels,ignore_index=len(model.charset))

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        rand_acc = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", unit="batch", position=1, leave=False):
                images, labels = images.to(device), labels.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = model.criterion(outputs.permute(0, 2, 1), labels)

                val_loss += loss.item()
                val_acc += accuracy(outputs, labels,ignore_index=len(model.charset))
                random_preds = torch.randint(0, model.num_classes - 1, outputs.shape).to(device)
            rand_acc += accuracy(random_preds, labels,ignore_index=len(model.charset))

        tqdm.write(f"""Epoch {epoch+1}/{epochs}, Train :: [Loss: {train_loss / len(train_loader):.4f}, Acc: {train_acc/len(train_loader):.4f}] | Validation :: [Loss: {val_loss / len(val_loader):.4f}, Acc: {val_acc/len(val_loader):.4f}] | Random Baseline :: [Accuracy: {rand_acc:.4f}]""")


def accuracy(outputs, labels, ignore_index):
    _, predicted = torch.max(outputs, 2)
    correct_chars = 0
    total_chars = 0

    predicted = predicted.view(-1)
    true_labels = labels.view(-1)

    non_padding_mask = true_labels != ignore_index

    correct_chars += ((predicted == true_labels) & non_padding_mask).sum().item()
    total_chars += non_padding_mask.sum().item()

    return correct_chars / total_chars


