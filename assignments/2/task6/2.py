from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from models.KMeans.km import *
from models.GMM.GMM import *
from models.PCA.PCA import *

if __name__=="__main__":
    pca = PCA(2)
    path = "./data/external/word-embeddings.feather"
    pca.load_raw_data(path)
    pca.preprocess()
    pca.fit(sp=True)
    pca.scree_plot(save=True,name="512Dscree")

    
    optimal_dims = 5
    pca_opt = PCA(optimal_dims)
    pca_opt.load_raw_data(path)
    pca_opt.preprocess()
    pca_opt.fit()

    reduced = pca_opt.transform()

    errors = []
    for k in range(1,11):
        km = KMeans(k)
        km.fit(reduced)
        wcse = km.get_cost(reduced)
        errors.append(wcse)

    ks = list(range(1, 11))


    save_path="./assignments/2/figures/PCA/Cluster_Elbow.png"
    plt.figure(figsize=(8, 6))
    plt.plot(ks, errors, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSE (Within-Cluster Sum of Errors)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig(save_path)

    kmeans3 = 5
    km = KMeans(kmeans3)
    km.fit(reduced)
    clusters = km.predict(reduced)
    save_path="./assignments/2/figures/clusters/clustered_words_kmeans3.txt"
    with open(save_path, 'w') as f:
        for i in range(kmeans3):
            index0 = np.asarray(np.where(clusters == i))
            words = pca_opt.df['words'].iloc[index0[0]] 

            f.write(f"Cluster {i}:\n")
            for word in words:
                f.write(word + "\n")
            f.write("\n") 
    print("All clustered words saved to "+save_path)