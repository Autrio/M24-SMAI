import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import hamming_loss, f1_score
from tqdm.auto import tqdm
# from tqdm import tqdm #!used for direct testing code 
from icecream import ic

class Autoencoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.encoder = nn.ModuleList(
                [
                    self.create_encoder_block(1, 32), # 32, 14, 14
                    self.create_encoder_block(32, 64), # 64, 7, 7
                    self.create_encoder_block(64, 128) # 128, 3, 3
                ]
            )
            
        self.decoder = nn.ModuleList(
            [
                self.create_decoder_block(128, 128, kernel_size=4), # 128, 4, 4
                self.create_decoder_block(128, 64, kernel_size=4, stride=1, padding=0), # 64, 7, 7 
                self.create_decoder_block(64, 32, kernel_size=4, padding=1), # 32, 14, 14                
                self.create_decoder_block(32, 1, kernel_size=4, stride=2, padding=1, last=True),  # 3, 28, 28
            ]
        )
        
        self.fc1 = nn.Sequential(nn.Linear(128 * 3 * 3, 128 * 1 * 1),nn.LeakyReLU(),
                                 nn.Linear(128,params["latent_dims"]),nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(params["latent_dims"],128),nn.LeakyReLU())

        
    def create_encoder_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding = 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
            nn.Dropout(0.1)
        )
        
    def create_decoder_block(self, in_c, out_c, kernel_size=3, padding=0, stride=2, last=False):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, in_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.ConvTranspose2d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(), 
            
            nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_c), 
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(out_c, out_c, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU() if last else nn.Sigmoid(),
            
            nn.Dropout(0.2) if last else nn.Identity()
            
        )
        
    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        x = x.view(-1, 128 * 3 * 3)
        x = self.fc1(x) # ld * 1 * 1
        x = self.fc2(x) # 128 * 1 * 1
        x = x.view(-1, 128, 1, 1)
        
        for layer in self.decoder:
            x = layer(x)
        
        return x
    
    def feature_map(self,x):
        pass

class Model:
    def __init__(self, model):
        self.model = model

    def set_attr(self, loss_fn, train_loader, val_loader, device, params,metrics,logger=None):
        self.device = device
        self.params = params
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        self.metrics = metrics

    def step_train(self, x ,y):
        self.model.train()
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def step_val(self, x, y):
        with torch.no_grad():
            y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        return loss.item()

    
    
    def train_batch(self, loader, step_fn):
        loss = 0
        for x, y in tqdm(loader,position=1,leave=False,ascii=" ▖▘▝▗▚▞█"):
            x, y = x.to(self.device), y.to(self.device)
            l = step_fn(x, x) #! passing y as x for autoencoder
            loss = loss + l
        return loss/len(loader)
    
    def train(self):
        for epoch in tqdm(range(self.params['epoch']),position=0,desc="Training",unit='epoch',ascii="░▒█"):
            train_loss = self.train_batch(self.train_loader, self.step_train)
            val_loss = self.train_batch(self.val_loader, self.step_val)
            

            tqdm.write(f"[Epoch: {epoch}] Train:[loss:{train_loss}]  Val:[loss:{val_loss}]")
            

    def save(self, path="wb.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'params': self.params
        }, path)
        print(f"Model saved to {path}")

    def load(self, path="wb.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.params = checkpoint['params']
        print(f"Model loaded from {path}")