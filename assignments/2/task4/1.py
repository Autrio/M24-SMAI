from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
from icecream import ic
from models.GMM.GMM import *

if __name__ == "__main__":
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(loc=-2, scale=1, size=(100, 2)),
        np.random.normal(loc=3, scale=1, size=(100, 2))
    ])

    gmm = GMM(num_comp=6)
    gmm.fit(X)

    print("GMM Parameters:")
    print(gmm.getParams())

    print("\nMembership Probabilities:")
    print(gmm.getMembership(X)[:5])

    print("\nLikelihood of the dataset:")
    print(gmm.getLikelihood(X))