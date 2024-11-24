import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from models.KMeans.km import *
from models.GMM.GMM import *
from models.PCA.PCA import *

if __name__=="__main__":
    k2 = 3
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/word-embeddings.feather"

    km = KMeans(k2)
    km.load_raw_data(path)
    km.preprocess()
    km.fit()
    clusters = km.predict()
    save_path="/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/assignments/2/figures/clusters/clustered_words_k2.txt"
    with open(save_path, 'w') as f:
        for i in range(k2):
            index0 = np.asarray(np.where(clusters == i))
            words = km.df['words'].iloc[index0[0]]  
            
            f.write(f"Cluster {i}:\n")
            for word in words:
                f.write(word + "\n")
            f.write("\n")  

    print("All clustered words saved to "+save_path)