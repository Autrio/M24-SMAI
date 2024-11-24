import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm



class Macros:
    def __init__(self,performance) -> None:

        self.performance = performance
        self.accuracies = np.array([performance[x]['accuracy'] for x in list(performance.keys())[:-4]])
        self.recalls = np.array([performance[x]['recall'] for x in list(performance.keys())[:-4]])
        self.precisions = np.array([performance[x]['precision'] for x in list(performance.keys())[:-4]])
        self.F1scores = np.array([performance[x]['F1-score'] for x in list(performance.keys())[:-4]])

        self.TP = performance["TP"]
        self.FP = performance["FP"]
        self.FN = performance["FN"]
    
    def Precision(self,mode):
        if(mode=='macro'):
            return np.mean(self.precisions)
        else:
            self.pres_micro =  self.TP / (self.TP + self.FP) if (self.TP + self.FP)!=0 else 0 
            return self.pres_micro
    
    def Recall(self,mode):
        if(mode=='macro'):
            return np.mean(self.recalls)
        else:
            self.rcl_micro = self.TP / (self.TP + self.FN) if (self.TP + self.FN)!=0 else 0 
            return self.rcl_micro
            
    def F1Score(self,mode):
        if(mode=='macro'):
            return np.mean(self.F1scores)
        else:
            return 2 * (self.pres_micro * self.rcl_micro) / (self.pres_micro + self.rcl_micro) if (self.pres_micro + self.rcl_micro)!=0 else 0


    def macros(self,mode,encoded=False):
        if encoded:
            return {
            "accuracy" : 10*self.performance["overall"],
            "precision" : 10*self.Precision(mode),
            "recall" : 10*self.Recall(mode),
            "F1-score" : 10*self.F1Score(mode),
        }
        return {
            "accuracy" : self.performance["overall"],
            "precision" : self.Precision(mode),
            "recall" : self.Recall(mode),
            "F1-score" : self.F1Score(mode),
        }

