from sklearn.mixture import GaussianMixture
from models.GMM.GMM import Data
from icecream import ic

if __name__=="__main__":

    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/word-embeddings.feather"

    data = Data()
    data.load_raw_data(path)
    data.preprocess()
    X = data.embeds

    gm=GaussianMixture(n_components=5,random_state=0)
    gm.fit(X)

    ic("Mean of Data :: ",gm.means_)
    ic("Covariancces of Data :: ",gm.covariances_)
    ic("weights of Data :: ",gm.weights_)

    
