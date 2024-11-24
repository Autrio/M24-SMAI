import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct
import os

def read_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
    return torch.tensor(images, dtype=torch.float32) / 255.0  

def read_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return torch.tensor(labels, dtype=torch.long)

class FashionMNISTDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = read_images(images_path)
        self.labels = read_labels(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        label_tensor = torch.full_like(image, label, dtype=torch.float32)
        
        return image, label_tensor
    
def Loader(params):
    train_images_path = './data/external/FashionMNIST/train-images-idx3-ubyte'
    train_labels_path = './data/external/FashionMNIST/train-labels-idx1-ubyte'
    test_images_path = './data/external/FashionMNIST/t10k-images-idx3-ubyte'
    test_labels_path = './data/external/FashionMNIST/t10k-labels-idx1-ubyte'

    train_dataset = FashionMNISTDataset(train_images_path, train_labels_path)
    test_dataset = FashionMNISTDataset(test_images_path, test_labels_path)

    train_loader = DataLoader(dataset=train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=params['batch_size'], shuffle=False)
    y_train = np.array([label.cpu().numpy() for _, label in train_dataset])
    y_test = np.array([label.cpu().numpy() for _, label in test_dataset])

    return train_loader,test_loader,y_train,y_test


