from models.knn.knn import *
from performanceMeasures.macros import *
from performanceMeasures.parameters import *
from models.PCA.PCA import *


if __name__ == "__main__":
    path = "./data/external/spotify.csv"
    knn = KNN()
    knn.load_raw_data(path)
    knn.Preprocess()
    knn.Split(test=0.1,valid=0.1)
    knn.Normalize()

    X = knn.Xtrain

    pca = PCA(2)
    pca.fit(X,sp=True)
    pca.scree_plot(X,name="Spotify")
