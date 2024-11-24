from PIL import Image, ImageDraw, ImageFont
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import numpy as np
torch.autograd.set_detect_anomaly(True)
from tqdm import tqdm
from icecream import ic

class WordImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform
        self.max_length = max(len(f.split('.')[0]) for f in self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB').resize((256, 64))
        image = np.array(image).mean(axis=2)  
        if self.transform:
            image = self.transform(image)
        label = self.image_files[idx].split('.')[0]
        label = [char_set.index(char) for char in label]  
    
        if any(c < 0 or c >= len(char_set) for c in label):
            raise ValueError(f"Invalid label {label} in file {self.image_files[idx]}")
        
        label = torch.tensor(label, dtype=torch.long)
        if len(label) < self.max_length:
            label = torch.cat([label, torch.full((self.max_length - len(label),), len(char_set), dtype=torch.long)])
        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.type(torch.float32)),
    transforms.Normalize((0.5,), (0.5,))
])

path = "./data/interim/5/4/nltk"
unique_chars = set()
for filename in os.listdir(path):
    if filename.endswith('.png'):
        unique_chars.update(filename.split('.')[0])

char_set = list(unique_chars)

dataset = WordImageDataset(path, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class CNN_RNN(nn.Module):
    def __init__(self, n_hidden, max_length, num_classes=len(char_set) + 1):
        super(CNN_RNN, self).__init__()
        self.max_length = max_length
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Linear(256 * 4 * 16, max_length)
        self.rnn = nn.RNN(
            input_size=max_length,
            hidden_size=n_hidden,
            num_layers=2,
            batch_first=True,
            nonlinearity='relu',
            dropout=0.3
        )
        self.fc_out = nn.Linear(n_hidden, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, -1)
        cnn_features = self.fc(cnn_features)
        cnn_features = cnn_features.unsqueeze(1).expand(-1, self.max_length, -1)
        rnn_output, _ = self.rnn(cnn_features)
        output = self.fc_out(rnn_output)
        return output

n_hidden = 256
num_layers = 2
ic(len(char_set) + 1)
ic(dataset.max_length)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN_RNN(n_hidden, max_length=dataset.max_length).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=len(char_set))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

scaler = torch.amp.GradScaler('cuda')

for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs.permute(0, 2, 1), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs.permute(0, 2, 1), labels)
            val_loss += loss.item()
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
