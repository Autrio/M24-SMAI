import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from models.KMeans.km import *
from models.knn.knn import *
from performanceMeasures.macros import *
from performanceMeasures.parameters import *
from models.PCA.PCA import *
from icecream import ic
import time

if __name__=="__main__":
    path = "./data/external/spotify.csv"
    knn = KNN()
    knn.load_raw_data(path)
    knn.Preprocess()
    knn.Split(test=0.1,valid=0.1)
    knn.Normalize()

    start_time = time.time()
    predictions = knn.predict(disType="manhattan",k=9,optimized=False)
    end_time = time.time()
    runtime_base = end_time-start_time

    ic("Runtime without PCA :: ",runtime_base) 

    performance = knn.inference()

    report = Macros(performance)
    base_accuracy = report.macros(mode="macro")

    ic("Base performance",report.macros(mode="macro"))

    knn2 = KNN()
    knn2.load_raw_data(path)
    knn2.Preprocess()
    knn2.Split(test=0.1,valid=0.1)
    knn2.Normalize()

    pca = PCA(num_comp=3)
    embeds_train = knn2.Xtrain
    pca.fit(embeds_train)

    reduced_train_data = pca.transform(embeds_train)
    embeds_test = knn2.Xtest
    pca.fit(embeds_test)

    reduced_test_data = pca.transform(embeds_test)

    knn2.Xtrain = reduced_train_data
    knn2.Xtest = reduced_test_data


    start_time = time.time()
    predictions = knn2.predict(disType="manhattan",k=9,optimized=False)
    end_time = time.time()
    runtime_PCA = end_time-start_time

    ic("Runtime with PCA :: ",runtime_PCA) 

    performance2 = knn2.inference()

    report2 = Macros(performance2)
    PCA_accuracy = report2.macros(mode="macro")

    ic("PCA performance",report2.macros(mode="macro"))
    base_acc = base_accuracy["accuracy"]
    pca_acc = PCA_accuracy["accuracy"]

    fig,ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Time Taken (seconds)', color=color)
    ax1.bar(['Without PCA', 'With PCA'], [runtime_base, runtime_PCA], color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(['Without PCA', 'With PCA'], [base_acc,pca_acc], color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Comparison of Time Taken and Accuracy (With and Without PCA)')
    save_path="./assignments/2/figures/PCA/PCA_Time_Taken.png"

    plt.savefig(save_path)

    