import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, num_comp) -> None:
        self.num_comp = num_comp
        self.mean = None
        self.components = None

    def fit(self, X=None, sp=False):
        if X is None:
            raise ValueError("Input data X cannot be None.")
        
        self.mean = torch.mean(X, dim=0)
        X_cent = X - self.mean
        
        CovMat = torch.cov(X_cent.T)  # Transpose for correct covariance computation
        
        eigenvalues, eigenvectors = torch.linalg.eigh(CovMat)  # Use eigh for symmetric matrices
        
        idx_sorted = torch.argsort(eigenvalues,descending=True  )
        self.eigenvectors = torch.real(eigenvectors[:, idx_sorted])
        self.eigenvalues = torch.real(eigenvalues[idx_sorted])
        self.components = self.eigenvectors[:, :self.num_comp]  # Correct component assignment
        if sp:
            return self.eigenvectors, self.eigenvalues

    def transform(self, X=None):
        if X is None:
            raise ValueError("Input data X cannot be None.")
        if self.components is None:
            raise RuntimeError("PCA is not fitted yet. Call fit() before transform().")
        X_cent = X - self.mean
        return torch.mm(X_cent, self.components)  # Correct matrix multiplication

    def checkPCA(self, X=None, tol=0.05):
        if X is None:
            raise ValueError("Input data X cannot be None.")
        if self.components is None:
            raise RuntimeError("PCA is not fitted yet. Call fit() before checkPCA().")

        self.X_transformed = self.transform(X)
        self.X_reconstructed = torch.mm(self.X_transformed, self.components.T) + self.mean
        reconstruction_error = torch.mean(torch.square(X - self.X_reconstructed))
        return reconstruction_error < tol

    def scree_plot(self, X=None, save=False, name=None):
        if X is None:
            raise ValueError("Input data X cannot be None.")
        eigenvectors, eigenvalues = self.fit(X, sp=True)
        tot_var = torch.sum(eigenvalues)
        self.evr = eigenvalues[:20] / tot_var

        plt.figure(figsize=(8, 6))
        plt.plot(torch.arange(1, len(self.evr) + 1), self.evr, 'o-', linewidth=2)
        plt.title("Scree Plot")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.grid(True)
        if save:
            plt.savefig(f'./assignments/2/figures/PCA/{name}.png')
        plt.show()


class Model:
    def __init__(self, n):
        self.PCA = PCA(n)
        self.n = n
        self.data = None
        self.reduced = None
        self.mean = None
        self.recon = None

    def gather(self, loader):
        data = []
        for x, _ in loader:
            x = x.view(x.size(0), -1)  
            data.append(x)
        self.data = torch.cat(data, dim=0)
        return self.data

    def fit(self, enc=True):
        if enc:
            X = self.data
        else:
            if self.reduced is None:
                raise ValueError("Reduced data is not available. Run encode first.")
            X = self.reduced
        self.PCA.fit(X)
        self.PC = self.PCA.transform(X)
        return self.PC

    def encode(self):
        self.mean = torch.mean(self.data, dim=0)
        self.data = self.data - self.mean
        self.reduced = self.PCA.transform(self.data)
        return self.reduced

    def decode(self):
        if self.reduced is None:
            raise ValueError("Reduced data is not available. Run encode first.")
        self.recon = torch.mm(self.reduced, self.PCA.components.T) + self.mean
        return self.recon

    def calc_recon_error(self):
        if self.recon is None or self.data is None:
            raise ValueError("Reconstructed data or original data is not available.")
        return torch.mean((self.data - self.recon) ** 2) 
