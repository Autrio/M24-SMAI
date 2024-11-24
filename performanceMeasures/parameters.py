import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
import pandas as pd
from itertools import product
from tqdm import tqdm

class Tuning:
    def __init__(self,model,K,metrics) -> None:
        self.model = model
        self.K = K
        self.metrics = metrics
        self.results = []

    def tune(self):
        for k,metric in tqdm(iterable=product(self.K,self.metrics),desc="tuning",unit=" units done",colour="red"):
            print(" ")
            self.model.predict(metric,k,optimized=True)
            performance = self.model.inference()
            self.results.append((k,metric,performance["overall"]))
        return self.results
    
    def disptopN(self,N=10):
        self.results = sorted(self.results, key=lambda x: x[2], reverse=True) # 3rd index is accuracy
        print("top 10 {k,metric,accuracy} triplets are: ")
        for result in self.results[:N]:
            k, metric, acc = result
            print(f"k={k}, Metric={metric}, Accuracy={acc:.4f}")
