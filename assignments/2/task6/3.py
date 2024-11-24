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
    k2 = 3
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/word-embeddings.feather"

    data = Data()
    data.load_raw_data(path)
    data.preprocess()
    X = data.embeds
    gmm = GaussianMixture(n_components=k2, covariance_type='full', random_state=42,verbose=2)
    gmm.fit(X)
    clusters = gmm.predict(X)
    save_path="./assignments/2/figures/clusters/clustered_words_gmmk2.txt"
    with open(save_path, 'w') as f:
        for i in range(k2):
            index0 = np.asarray(np.where(clusters == i))
            words = data.df['words'].iloc[index0[0]]  

            
            f.write(f"Cluster {i}:\n")
            for word in words:
                f.write(word + "\n")
            f.write("\n")

    print("All clustered words saved to "+save_path)
