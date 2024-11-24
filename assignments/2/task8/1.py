import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from models.GMM.GMM import Data

def generate_dendrogram(embeds, method, metric, save_path):
    link_matrix = linkage(embeds, method, metric)
    plt.figure(figsize=(25, 10))
    plt.title(f'Hierarchical Clustering Dendrogram\nLinkage: {method}, Metric: {metric.capitalize()}')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    dendrogram(
        link_matrix,
        leaf_rotation=90.,
        leaf_font_size=8.,
    )
    plt.savefig(save_path)
    return link_matrix

def save_cluster(link_matrix, dataframe, num_clusters, save_path):
    clusters = fcluster(link_matrix, num_clusters, 'maxclust')
    with open(save_path, 'w') as f:
        for i in range(1, num_clusters + 1):
            index0 = np.asarray(np.where(clusters == i))
            words = dataframe['words'].iloc[index0[0]]
            f.write(f"Cluster {i}:\n")
            for word in words:
                f.write(word + "\n")
            f.write("\n")
    print(f"All clustered words saved to {save_path}")

def run(embeds, dataframe):
    metrics = ['euclidean', 'cosine', 'cityblock']
    methods = ['single', 'complete', 'average']
    
    for metric in metrics:
        for method in methods:
            save_path = f"./assignments/2/figures/hierarchical/clusters_{method}_{metric}.png"
            link_matrix = generate_dendrogram(embeds, method, metric, save_path)

    method, metric = 'ward', 'euclidean'
    save_path = "./assignments/2/figures/hierarchical/clusters_{method}_{metric}.png"
    link_matrix = generate_dendrogram(embeds, method, metric, save_path)

    save_cluster(link_matrix, dataframe, 4, "./assignments/2/figures/clusters/hierarchial_clustering_ward_euclidean_kbest2.txt")
    save_cluster(link_matrix, dataframe, 5, "./assignments/2/figures/clusters/hierarchial_clustering_ward_euclidean_kbest1.txt")

if __name__=="__main__":
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/word-embeddings.feather"
    data_obj = Data()
    data_obj.load_raw_data(path)
    data_obj.preprocess()
    X = data_obj.embeds

    run(X,data_obj.df)