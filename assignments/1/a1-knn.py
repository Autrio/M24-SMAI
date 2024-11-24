import argparse
from models.knn.knn import *
from performanceMeasures.macros import *
from performanceMeasures.parameters import *

parser = argparse.ArgumentParser(prog="A1",description="argument parser for KNN")

parser.add_argument("-q","--question",type=int,help="""
                    Choose which subpart of KNN to execute
                    """,default=False)
args  = parser.parse_args()

path = "./data/external/spotify.csv"
knn = KNN()

#* loading the raw data ---> for loading partial dataset use test = True and set percentage(float) 
knn.load_raw_data(path)

#* preprocess data ---> default IQR - 8 and correlation factor - 0.75
knn.Preprocess()

#* splitting data
knn.Split(test=0.1,valid=0.1)

#* z-score normalization
knn.Normalize()

def Q1():
    #!plotting functions
    knn1 = KNN()
    knn1.load_raw_data(path)
    #! histogram before preprocessing
    knn1.histogram()
    knn1.Preprocess()
    knn1.Split()
    #! correlation heatmap after preprocessing
    knn1.CorrelationMatrix()
    knn1.Boxplot()

def Q2():
    #* arbitrary k , distance
    distance_metric = 'euclid'
    k = 1

    #? main prediction function
    knn.predict(distance_metric,k,batch_size=100)

    #* generate performance report
    performance = knn.inference()

    report = Macros(performance)

    print(report.macros(mode="macro"))


def Q3():
    K = [x for x in range(10)[1:]]
    metrics = ['euclid','manhattan','cosine']
    tuner = Tuning(model=knn,K=K,metrics=metrics)
    tuner.tune()
    tuner.disptopN()

def Q4():
    #! finish time  comparision plots
    import matplotlib.pyplot as plt
    #! runtime for full dataset
    knn.RuntimeComp()

    #! evaluating size vs time
    knn3 = KNN()
    splits = [0.1, 0.3, 0.5, 0.7]  
    optimized_times = []
    non_optimized_times = []
    sklearn_times = []

    for ratio in splits:
        knn3.load_raw_data(test=True, percentage=ratio, path=path)
        knn3.Preprocess()
        knn3.Split()
        knn3.Normalize()
        
        optimized_time, non_optimized_time, sklearn_time = knn3.RuntimeComp()
        optimized_times.append(optimized_time)
        non_optimized_times.append(non_optimized_time)
        sklearn_times.append(sklearn_time)
    
    plt.figure(figsize=(12, 8))
    plt.plot(splits, optimized_times, label='Optimized KNN', marker='o')
    plt.plot(splits, non_optimized_times, label='Non-Optimized KNN', marker='x')
    plt.plot(splits, sklearn_times, label='sklearn KNN', marker='^')

    plt.xlabel('Training Data Ratio')
    plt.ylabel('Inference Time (seconds)')
    plt.title('Inference Time vs Training Data Ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./assignments/1/figures/time-{ratio*100}.jpg")
    plt.show()


def Q5():
    trainpath = "./data/external/spotify-2/train.csv"
    testpath = "./data/external/spotify-2/test.csv"
    valpath = "./data/external/spotify-2/validate.csv"
    knn2 = KNN()
    ratio = knn2.load_explict(testPath=testpath,trainPath=trainpath,validatePath=valpath)

    knn2.Preprocess()

    knn2.Split(test=ratio[1],valid=ratio[2])

    knn2.Normalize()

    #! best k, distance metric pair is 9-manhattan

    distance_metric = 'manhattan'
    k = 9

    knn2.predict(distance_metric,k,optimized=True)
    performance = knn2.inference()

    report = Macros(performance)

    print(report.macros(mode="macro"))

    

if __name__ == "__main__":
    if(args.question==1):
        Q1()
    elif(args.question==2):
        Q2()
    elif(args.question==3):
        Q3()
    elif(args.question==4):
        Q4()
    elif(args.question==5):
        Q5()
    else:
        pass