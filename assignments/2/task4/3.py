from sklearn.mixture import GaussianMixture
from models.GMM.GMM import Data
from icecream import ic
import matplotlib.pyplot as plt
import os

if __name__=="__main__":
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/word-embeddings.feather"

    data = Data()
    data.load_raw_data(path)
    data.preprocess()
    X = data.embeds

    num_comp = 10

    aic = []
    bic = []
    for i in range(1,num_comp):
        gm = GaussianMixture(n_components=i, covariance_type='full', random_state=42,verbose=2)
        gm.fit(X)
        aic.append(gm.aic(X))
        bic.append(gm.bic(X))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1,num_comp), aic, label='AIC', marker='o')
    plt.plot(range(1,num_comp), bic, label='BIC', marker='x')
    plt.title('AIC and BIC Scores for Different Numbers of GMM Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    save_path = '/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/assignments/2/figures/GMM/AICBIC.png'
    plt.savefig(save_path)