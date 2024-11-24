from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from icecream import ic
from models.PCA.PCA import *

if __name__ == "__main__":
    np.random.seed(28)
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/word-embeddings.feather"

    
    pca_2D = PCA(num_comp=2)
    pca_2D.load_raw_data(path)
    pca_2D.preprocess()
    
    pca_2D.fit()
    
    X_transformed_2D = pca_2D.transform()
    
    is_working = pca_2D.checkPCA()
    print(f"PCA 2D check passed: {is_working}")

    pca_3D = PCA(num_comp=3)
    pca_3D.load_raw_data(path)
    pca_3D.preprocess()
    
    pca_3D.fit()
    
    X_transformed_3D = pca_3D.transform()
    
    is_working = pca_3D.checkPCA()
    print(f"PCA 3D check passed: {is_working}")

    filename = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/assignments/2/figures/PCA/PCA2D.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(X_transformed_2D[:, 0], X_transformed_2D[:, 1], c='blue', s=10)
    plt.title('2D PCA Projection')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.savefig(filename)

    filename = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/assignments/2/figures/PCA/PCA3D.png"
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_transformed_3D[:, 0], X_transformed_3D[:, 1], X_transformed_3D[:, 2], c='red', s=10)
    ax.set_title('3D PCA Projection')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.savefig(filename)