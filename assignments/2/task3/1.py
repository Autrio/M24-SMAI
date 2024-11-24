import numpy as np
import pandas as pd
from tqdm import tqdm
from icecream import ic
import matplotlib.pyplot as plt

from models.KMeans.km import *

if __name__ == "__main__":
    km = KMeans(K=3)

    # path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/word-embeddings.feather"


    km.load_raw_data( "/home/autrio/college-linx/SMAI/testing/assignment2/data.csv")  
    km.preprocess2D()

    # km.load_raw_data(path)
    # km.preprocess()

    km.fit()

    cost = km.get_cost()

    visualise2D(km.embeds,km)
