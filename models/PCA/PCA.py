from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from icecream import ic

class Data:
    def __init__(self) -> None:
        pass

    def load_raw_data(self,path):
        self.path = path
        self.df = pd.read_feather(self.path)
        
    def preprocess(self):
        self.words = self.df.iloc[:,0].values
        self.embeds = np.stack(self.df.iloc[:,1].values)    

class PCA(Data):
    def __init__(self,num_comp) -> None:
        super().__init__()
        self.num_comp = num_comp

    def load_raw_data(self, path):
        return super().load_raw_data(path)
    
    def preprocess(self):
        return super().preprocess()

    def fit(self, X=None,sp=False):
        if(X is None):
            X = self.embeds
        
        self.mean = np.mean(X, axis=0)
        X_cent = X - self.mean
        
        CovMat = np.cov(X_cent, rowvar=False)
        
        eigenvalues, eigenvectors = np.linalg.eig(CovMat)
        
        idx_sorted = np.argsort(eigenvalues)[::-1]
        self.eigenvectors = np.real(eigenvectors[:, idx_sorted])
        self.eigenvalues =  np.real(eigenvalues[idx_sorted])
        self.components = eigenvectors[:, :self.num_comp]
        if sp:
            return self.eigenvectors,self.eigenvalues
        

    def transform(self, X=None):
        if(X is None):
            X = self.embeds
        if self.components is None:
            raise RuntimeError("PCA is not fitted yet. Call fit() before transform().")
        X_cent = X - self.mean
        return np.dot(X_cent, self.components)

    def checkPCA(self, X=None, tol=0.05):
        if(X is None):
            X = self.embeds
        if self.components is None:
            raise RuntimeError("PCA is not fitted yet. Call fit() before checkPCA().")

        self.X_transformed = self.transform(X)
        self.X_reconstructed = np.dot(self.X_transformed, self.components.T) + self.mean
        reconstruction_error = np.mean(np.square(X - self.X_reconstructed)**2)
        return reconstruction_error < tol
    
    def scree_plot(self, X=None, save=False, name=None):
        if X is None:
            X = self.embeds
        eigenvectors, eigenvalues=self.fit(X,sp=True)
        tot_var = np.sum(eigenvalues)
        self.evr = eigenvalues[:20] / tot_var

        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(1, len(self.evr) + 1), self.evr, 'o-', linewidth=2)
        plt.title("Scree Plot")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.grid(True)
        if save:
            plt.savefig(f'./assignments/2/figures/PCA/{name}.png')
        plt.show()



