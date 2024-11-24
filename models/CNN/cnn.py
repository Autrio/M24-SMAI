import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import hamming_loss, f1_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
# from tqdm import tqdm #!used for direct testing code 
from icecream import ic
import copy

class net(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.layers = nn.ModuleList()
        
    def drop(self,rate=0.2):
        return nn.Dropout(rate) if self.params['use_dropout'] else nn.Identity()

    def Convlayer(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=self.params['kernel_size'], stride=1, padding=self.params['padding']),
            nn.ReLU(),
            nn.MaxPool2d(2))
    
    def add(self, layer):
        self.layers.append(layer)

    def __str__(self):
        out = ""
        for layer in self.layers:
            out += layer.__str__() + "\n"
        return out

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def feature_map(self,x):
        self.eval()
        if x.ndim == 3:  # If x has shape [channels, height, width]
            x = x.unsqueeze(0)  # Add batch dimension: [1, channels, height, width]

        feature_maps = []
        with torch.no_grad():
            for layer in self.layers:
                x = layer(x)
                if isinstance(layer, nn.Sequential) and isinstance(layer[0], nn.Conv2d):
                    feature_maps.append(x)
        
        for i, fmap in enumerate(feature_maps):
            num_filters = fmap.shape[1]
            fig, axes = plt.subplots(1, min(num_filters, 8), figsize=(15, 15))
            fig.suptitle(f"Feature Maps after Convolutional Layer {i+1}")
            
            for j in range(min(num_filters, 8)):
                axes[j].imshow(fmap[0, j].cpu().numpy(), cmap='viridis')
                axes[j].axis('off')
            
            plt.show()

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
        self.task = self.metrics.task

    def step_train(self, x, y):
        y_pred = self.model(x).squeeze(1) if self.task == "regression" else self.model(x)
        y = y.float() if self.task == "regression" else y
        loss = self.loss_fn(y_pred, y)  
        y_true,y_pred_post = self.metrics.process_labels(y,y_pred)
        acc = self.metrics.acc_score(y_true, y_pred_post)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc 
    
    def step_val(self, x, y):
        with torch.no_grad():
            y_pred = self.model(x).squeeze(1) if self.task == "regression" else self.model(x)
            y = y.float() if self.task == "regression" else y
        loss = self.loss_fn(y_pred, y)
        
        y_true,y_pred_post = self.metrics.process_labels(y,y_pred)
        acc = self.metrics.acc_score(y_true, y_pred_post)
        
        return loss.item(), acc  
    
    def train_batch(self, loader, step_fxn):
        total_loss, total_acc = 0, 0
        for x, y in tqdm(loader,position=1,leave=False,ascii=" ▖▘▝▗▚▞█"):
            x, y = x.to(self.device), y.to(self.device)
            l, a = step_fxn(x, y)
            total_loss += l
            total_acc += a
        return total_loss / len(loader), total_acc / len(loader)
    
    def train(self,patience=5):
        best_loss = float('inf')
        best_acc = -float('inf')
        pc = 0
        for epoch in tqdm(range(self.params['epoch']),position=0,desc="Training",unit='epoch',ascii="░▒█"):
            train_loss, train_acc = self.train_batch(self.train_loader, self.step_train)
            val_loss, val_acc = self.train_batch(self.val_loader, self.step_val)

            tqdm.write(f"[Epoch: {epoch}] Train:[loss:{train_loss:.3f} acc:{train_acc:.3f}] Val:[loss:{val_loss:.3f} acc:{val_acc:.3f}]")
            
            if val_acc>best_acc:
                best_acc = val_acc

            if val_loss < best_loss:
                best_loss = val_loss
                pc = 0
                best_wnb = copy.deepcopy(self.model.state_dict())
            else:
                pc += 1 
            if pc >= patience:
                tqdm.write(f"Early stopping triggered after {epoch + 1} epochs || best model accuracy : {best_acc}")
                break
        
            if self.logger:
                self.logger.log({'train_loss': train_loss, 'train_acc': train_acc, 
                                 'val_loss': val_loss, 'val_acc': val_acc})
        if self.logger:     
            self.logger.finish()
        
        self.model.load_state_dict(best_wnb)

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

    def eval(self, test_loader):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        all_y_true = []
        all_y_pred = []
        performance = {}
        
        with torch.no_grad():
            for x, y in tqdm(test_loader, position=1, leave=False, ascii=" |=="):
                x, y = x.to(self.device), y.to(self.device)
                
                y_pred = self.model(x).squeeze(1) if self.task == "regression" else self.model(x)
                y = y.float() if self.task == "regression" else y
                
                loss = self.loss_fn(y_pred, y)
                total_loss += loss.item()
                
                y_true, y_pred_post = self.metrics.process_labels(y, y_pred)
                acc = self.metrics.acc_score(y_true, y_pred_post)
                total_acc += acc

                all_y_true.extend(y_true.cpu().numpy() if isinstance(y_true,torch.Tensor) else y_true)
                all_y_pred.extend(y_pred_post.cpu().numpy() if isinstance(y_pred_post,torch.Tensor) else y_pred_post)
        
        avg_loss = total_loss / len(test_loader)
        avg_acc = total_acc / len(test_loader)

        performance["loss"] = avg_loss
        performance["accuracy"] = avg_acc

        if self.task == "regression":
            mse = torch.nn.functional.mse_loss(torch.tensor(all_y_pred), torch.tensor(all_y_true), reduction='mean').item()
            mae = torch.nn.functional.l1_loss(torch.tensor(all_y_pred), torch.tensor(all_y_true), reduction='mean').item()
            performance["MSE"] = mse
            performance["MAE"] = mae
        else:
            hamming = hamming_loss(all_y_true, all_y_pred)
            f1 = f1_score(all_y_true, all_y_pred, average='macro')
            performance["hamming loss"] = hamming
            performance["f1 score"] = f1

        return performance
