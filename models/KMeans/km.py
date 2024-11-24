import numpy as np
import pandas as pd
from tqdm import tqdm
from icecream import ic
import matplotlib.pyplot as plt

class Data:
    def __init__(self) -> None:
        pass

    def load_raw_data(self, path):
        self.path = path

    def preprocess(self):
        self.df = pd.read_feather(self.path)
        self.words = self.df.iloc[:, 0].values
        self.embeds = np.vstack(self.df.iloc[:, 1].values)

    def preprocess2D(self):
        self.df = pd.read_csv(self.path)
        self.embeds = np.vstack(self.df.iloc[:,:2].values)
        self.words = self.df.iloc[:,2].values


class KMeans(Data):
    def __init__(self, K, max_iter=100, tol=1e-4) -> None:
        super().__init__()
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.mu = None
        self.labels = None

    def load_raw_data(self, path):
        return super().load_raw_data(path)
    
    def preprocess(self):
        return super().preprocess()

    def init_centroids(self, X=None):
        if X is None:
            X = self.embeds
        np.random.seed(28)
        rand_idx = np.random.permutation(X.shape[0])[:self.K]
        self.mu = X[rand_idx]

    def compute_distances(self, X=None):
        if X is None:
            X = self.embeds
        self.pbar = tqdm(desc="compute distances", total=self.K)
        self.distances = np.zeros((X.shape[0], self.K))
        for i, mu in enumerate(self.mu):
            self.distances[:, i] = np.linalg.norm(X - mu, axis=1)
            self.pbar.update(1)
        self.pbar.close()
        return self.distances

    def cluster(self,dist=None):
        if(dist is None):
            dist = self.distances
        return np.argmin(dist, axis=1)

    def compute_centroids(self, X=None):
        if X is None:
            X = self.embeds
        self.new_mu = np.zeros((self.K, X.shape[1]))
        for i in tqdm(range(self.K), desc="compute centroids"):
            if np.any(self.labels == i):
                self.new_mu[i] = np.mean(X[self.labels == i], axis=0)
        return self.new_mu

    def fit(self, X=None):
        if X is None:
            X = self.embeds
        self.init_centroids(X)
        for i in tqdm(range(self.max_iter), desc="fit"):
            self.compute_distances(X)
            self.labels = self.cluster()
            self.new_mu = self.compute_centroids(X)

            if np.linalg.norm(self.mu - self.new_mu) < self.tol:
                break
            self.mu = self.new_mu

    def predict(self, X=None):
        if(X is None):
            X = self.embeds
        distances = self.compute_distances(X)
        return self.cluster(distances)

    def get_cost(self, X=None):
        if X is None:
            X = self.embeds
        self.cost = 0
        for i, mu in enumerate(self.mu):
            cluster_points = X[self.labels == i]
            self.cost += np.sum((cluster_points - mu) ** 2)
        return self.cost

    
def visualise2D(X,km):
        plt.scatter(X[:,0],X[:,1],c=km.labels,cmap="viridis",marker='o',label='datapoints')
        plt.scatter(km.mu[:, 0], km.mu[:, 1], c='red', marker='x', s=100, label='Centroids')
        plt.title(f"K-Means Clustering (k={km.K})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()



