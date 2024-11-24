from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from models.KMeans.km import *
from models.GMM.GMM import *
from models.PCA.PCA import *

if __name__=="__main__":
    optimal_dims = 5
    pca = PCA(optimal_dims)
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/word-embeddings.feather"

    pca.load_raw_data(path)
    pca.preprocess()
    pca.fit()

    reduced = pca.transform()
    n_components_range = range(1, 11)

    aic_values = []
    bic_values = []

    for n_components in n_components_range:
        gmm = GMM(num_comp=n_components,max_iter=1000)
        gmm.fit(reduced)  
        aic_values.append(gmm.AIC(reduced))
        bic_values.append(gmm.BIC(reduced))
    save_path="./assignments/2/figures/GMM/GMM_AIC_BIC.png"

    plt.figure(figsize=(12, 6))
    plt.plot(n_components_range, aic_values, label='AIC', marker='o')
    plt.plot(n_components_range, bic_values, label='BIC', marker='x')
    plt.title('AIC and BIC for GMMs with Different Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)    

    kgmm3 = 4
    gmm2 = GMM(kgmm3,max_iter=1000)
    gmm2.fit(reduced)
    clusters = gmm2.predict(reduced)
    save_path="./assignments/2/figures/clusters/clustered_words_kgmm3.txt"

    with open(save_path, 'w') as f:
        for i in range(kgmm3):
            index0 = np.asarray(np.where(clusters == i))
            words = pca.df['words'].iloc[index0[0]]  

            f.write(f"Cluster {i}:\n")
            for word in words:
                f.write(word + "\n")
            f.write("\n")  

    print("All clustered words saved to "+save_path)