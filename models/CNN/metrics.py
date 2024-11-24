from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss, f1_score
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import numpy as np

class InvalidTaskException(Exception):
    def __init__(self,message):
        self.message = message
        super().__init__(self.message)

class Metrics:
    def __init__(self,task=None):
        if task not in ["single","multi","regression"]:
            raise InvalidTaskException("The specified task is either empty or invlaid")

        self.task = task

    def process_labels(self,y,y_pred):
        if self.task == "multi":
            return (y > 0.5).detach().cpu().numpy(),(y_pred > 0.5).detach().cpu().numpy()
        elif self.task=="single":
            return y,torch.argmax(y_pred,dim=1)
        else:
            return y,y_pred.squeeze()

    def acc_score(self,y,y_pred):
        fn = accuracy_score if self.task=="single" else hamming_loss if self.task == 'multi' else r2_score if self.task == "regression" else None
        assert fn is not None
        return fn(y.detach().cpu().numpy(),y_pred.detach().cpu().numpy()) if self.task != "multi" else 1-fn(y,y_pred)
    
    def loss_fn(self):
        if self.task == "multi":
            return nn.CrossEntropyLoss()
        elif self.task == "single":
            return nn.CrossEntropyLoss()
        elif self.task == "regression":
            return nn.MSELoss()
        else:
            raise InvalidTaskException("Specified task is empty or invalid")