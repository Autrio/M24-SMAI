import numpy as np
import pandas as pd
from tqdm import tqdm
from icecream import ic
import matplotlib.pyplot as plt

from models.KMeans.km import *

if __name__=="__main__":
    wcss = []
    krange = range(1,10)
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/word-embeddings.feather"

    for k in tqdm(krange,desc="optimal K"):
        km = KMeans(k)
        km.load_raw_data(path)
        km.preprocess()
        km.fit()
        wcss.append(km.get_cost())

    plt.plot(krange,wcss,marker='o')
    plt.grid(True)
    plt.show()