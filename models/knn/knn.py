import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
from tqdm import tqdm
import time

class Data:
    def __init__(self) -> None:
        # self.path = path
        self.labelData = ['track_genre' ,'artists']
        self.dropData = ['track_name', 'album_name', 'Unnamed: 0', 'track_id']

    def load_raw_data(self,path,percentage=1.0,test=False):
        self.path = path
        self.raw_data = pd.read_csv(self.path)
        self.df = pd.DataFrame(self.raw_data)
        
        if(test):
            self.df = self.df[:int(len(self.df)*percentage)]

        print("Number of samples :: ",len(self.df))
        self.attributes = self.df.columns
        print("Data attributes ::",self.attributes.to_list())

    #! loading data for spotify-2
    def load_explict(self,testPath,trainPath,validatePath):
        train = pd.read_csv(trainPath)
        test = pd.read_csv(testPath)
        val = pd.read_csv(validatePath)
        
        train['dataset'] = 'train'
        test['dataset'] = 'test'
        val['dataset'] = 'val'

        self.df = pd.concat([train,test,val])
        self.df.drop(columns=['dataset'],inplace=True)
        return [len(train)/len(self.df),len(test)/len(self.df),len(val)/len(self.df)]


    def Preprocess(self,predict='track_genre',IQRthresh = 5,CF = 0.8):
        #*remove multiple genres
        grouped_genres = self.df.groupby('track_id')['track_genre'].apply(lambda x: sorted(set(x))[0])

        #* Map the resolved genre back to the original DataFrame
        self.df['track_genre'] = self.df['track_id'].map(grouped_genres)

        #* Drop duplicates based on 'track_id'
        self.df.drop_duplicates(subset=['track_id'], inplace=True)

        #* label encoding  
        self.df[self.labelData] = self.df[self.labelData].apply(lambda col: pd.factorize(col)[0])

        #* drop string data
        self.df.drop(columns=self.dropData,inplace=True)

        self.df["explicit"] = self.df["explicit"].astype(int)

        #* obtaining numeric colomns to perform zscore norm
        floatData = self.df.select_dtypes(include=['float64']).columns.tolist()
        floatData.extend([col for col in ['popularity', 'duration_ms'] if col in self.df.columns])

        self.numerical = list(set(floatData) & set(self.df.columns))

        #* removing outliers
        Q1 = self.df[self.numerical].quantile(0.25)
        Q3 = self.df[self.numerical].quantile(0.75)

        lb = Q1 - IQRthresh * (Q3-Q1)
        ub = Q3 + IQRthresh * (Q3-Q1)

        IQRmask = np.all((self.df[self.numerical] >= lb) & (self.df[self.numerical] <= ub), axis=1)

        self.df = self.df[IQRmask]

        subset = self.df[self.numerical].select_dtypes(include=["int", "float"])
        self.corr_mat = subset.corr()

        abs_corr = self.corr_mat.abs()
        CORmask = (abs_corr > CF) & (abs_corr < 1.0)
        high_cor = abs_corr.columns[CORmask.any()]

        self.corl = set(high_cor)

        self.df.drop(columns=self.corl, inplace=True)

    #! LLM generated and refactored for my code structure
    def CorrelationMatrix(self):
        plt.figure(figsize=(20, 15), facecolor='#F2EAC5', edgecolor='black')
        ax = plt.axes()
        ax.set_facecolor('#F2EAC5')

        sns.heatmap(self.corr_mat, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 10},
                    cbar_kws={"shrink": .8},  
                    xticklabels=self.corr_mat.columns,  
                    yticklabels=self.corr_mat.columns)

        
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

        plt.title('Correlation Analysis')
        plt.tight_layout() 
        plt.savefig("./assignments/1/figures/EDA/CM.jpg")
        plt.show()
    

    def Boxplot(self):
        for column in self.df.columns:
            self.df.boxplot(column=column)
            plt.title("boxplot for label :: {}".format(column))
            plt.ylabel("values")
            plt.xlabel("Labels")
            plt.savefig(f"./assignments/1/figures/EDA/{column}-box.jpg")
            plt.show()
    
    #! end of LLM code

    def histogram(self):
        atrib = self.df.columns.difference(["Unnamed: 0",'track_id', 'artists', 'album_name', 'track_name'])
        print(atrib.values)
        for feature in atrib:
            plt.figure(figsize=(8, 6))
            if self.df[feature].dtype in ['int64', 'float64']:
                self.df[feature].hist(bins=90   , edgecolor='black')
                plt.title(f'Histogram of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')
            else:
                self.df[feature].value_counts().plot(kind='bar', edgecolor='black')
                plt.title(f'Bar Plot of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')

            plt.savefig(f'./assignments/1/figures/EDA/{feature}-bar.jpg')
            plt.close()


    def Split(self,test=0.1,valid=0.1,targ='track_genre'):
        self.df = self.df.sample(frac=1, random_state=28).reset_index(drop=True)    

        self.testSize = int((len(self.df)) * test)
        self.validSize = int((len(self.df)) * valid)

        
        self.test_df = self.df[0:self.testSize]
        self.valid_df = self.df[self.testSize:(self.testSize + self.validSize)]
        self.train_df = self.df[(self.testSize + self.validSize):]
        
        self.Xtrain = self.train_df.drop(columns=[targ])
        self.Ytrain = self.train_df[targ]
        self.Xtest = self.test_df.drop(columns=[targ])
        self.Ytest = self.test_df[targ]

        self.Xval = self.valid_df.drop(columns=[targ])
        self.Yval = self.valid_df[targ]    
    
    def split_encoder(self,Y):

        self.Ytest = Y[0:self.testSize]
        self.Yval = Y[self.testSize:(self.testSize + self.validSize)]
        self.Ytrain = Y[(self.testSize + self.validSize):]


    def Normalize(self):
        mean = self.Xtrain.mean()
        std = self.Xtrain.std()

        self.Xtrain = (self.Xtrain - mean) / std
        self.Xtest = (self.Xtest - mean) / std
        self.Xval = (self.Xval - mean) / std

        self.Xtrain = self.Xtrain.to_numpy()
        self.Xtest  = self.Xtest.to_numpy()
        self.Xval   = self.Xval.to_numpy()

        self.Ytrain = self.Ytrain.to_numpy()
        self.Ytest  = self.Ytest.to_numpy()
        self.Yval   = self.Yval.to_numpy()

    
class Metrics(Data):
    def __init__(self) -> None:
        super().__init__()
        self.Ypred = []
        self.metrics = {}
        self.classes = []

    def inference(self):
        self.classes = np.unique(self.Ytest)

        for value in self.classes:
            self.TruePos  = np.sum((self.Ytest == value) & (self.Ypred == value))
            self.FalsePos = np.sum((self.Ytest != value) & (self.Ypred == value))
            self.FalseNeg = np.sum((self.Ytest == value) & (self.Ypred != value))
            self.Actual = np.sum((self.Ytest == value))

            self.metrics[value] = {
                'precision' : self.precision(),
                'recall' : self.recall(),
                'accuracy' : self.accuracy(),
                'F1-score' : self.F1score()
            }
        
        self.TP = np.sum([np.sum((self.Ytest == value) & (self.Ypred == value)) for value in self.classes])
        self.FP = np.sum([np.sum((self.Ytest != value) & (self.Ypred == value)) for value in self.classes])
        self.FN = np.sum([np.sum((self.Ytest == value) & (self.Ypred != value)) for value in self.classes])

        self.metrics["overall"] = self.overall()
        self.metrics["TP"] = self.TP
        self.metrics["FP"] = self.FP
        self.metrics["FN"] = self.FN

        
        return self.metrics

    def accuracy(self):
        self.acc = self.TruePos/self.Actual if self.Actual!=0 else 0
        return self.acc
    
    def overall(self):
        self.ovl = np.sum((self.Ypred==self.Ytest))/len(self.Ytest)
        return self.ovl

    def precision(self):
        self.pres = self.TruePos / (self.TruePos + self.FalsePos) if self.TruePos + self.FalsePos > 0 else 0
        return self.pres

    def recall(self):
        self.rcl = self.TruePos / (self.TruePos + self.FalseNeg) if self.TruePos + self.FalseNeg > 0 else 0
        return self.rcl

    def F1score(self):
        self.F1 = 2 *(self.pres * self.rcl) / (self.pres + self.rcl) if self.pres + self.rcl > 0 else 0
        return self.F1


class KNN(Metrics):
    def __init__(self) -> None:
        super().__init__()
        self.report = ''

    def load_raw_data(self,path,test=False,percentage=0.1):
        return super().load_raw_data(path,percentage,test)
    
    def Preprocess(self, predict='track_genre', IQRthresh=8, CF=0.75):
        return super().Preprocess(predict, IQRthresh, CF)
    
    def Split(self, test=0.1, valid=0.1, targ='track_genre'):
        return super().Split(test, valid, targ)
    
    def Normalize(self):
        return super().Normalize()
    

    def distance(self,disType,x,y,optimised=False):
        if(disType=="euclid"):
            return np.sqrt(np.sum((x - y) ** 2, axis=1)) if not optimised else np.sqrt(np.sum((x - y[:, np.newaxis, :]) ** 2, axis=2)).squeeze(-1)
        
        elif(disType=="manhattan"):
            return np.sum(np.abs(x - y), axis=1) if not optimised else np.sum(np.abs(x - y[:, np.newaxis, :]), axis=2).squeeze(-1)

        elif(disType=="cosine"):
            return 1 - (np.sum(x * y, axis=-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)))
    
    
    def predict_optimized(self,DisType,k=1):
        self.Ypred = []
        for testDP in tqdm(self.Xtest,desc="training",unit="sample"):
            dist = self.distance(DisType,self.Xtrain,testDP)

            Nidx = np.argpartition(dist.ravel(),k)[:k]
            nn = self.Ytrain[Nidx]
            pred = max(set(nn), key=list(nn).count)
            self.Ypred.append(pred)
        return self.Ypred
        
    #! DEPRECIATED SUPPORT
    def predict_base(self, disType='euclid', k=1, batch_size=10000):
        n_test_samples = self.Xtest.shape[0]
        self.Ypred = []
        self.Xtrain = np.expand_dims(self.Xtrain,axis=-1)

        for start in tqdm(range(0, n_test_samples, batch_size), desc="Processing batches"):
            end = min(start + batch_size, n_test_samples)
            Xtest_batch = self.Xtest[start:end]
            Xtest_batch = np.expand_dims(Xtest_batch,axis=-1)

            dists = self.distance(disType,self.Xtrain,Xtest_batch,optimised=True)

            nnIdx = np.argpartition(dists,k)[:,:k]
            # print(np.unique(nnIdx))

            nearest_labels = self.Ytrain[nnIdx]
            print(nearest_labels)
            return
            # print(np.unique(nearest_labels))

            predictions_batch = [Counter(nnl).most_common(1)[0][0] for nnl in nearest_labels]

            self.Ypred.extend(predictions_batch)

        return self.Ypred

    def predict(self,disType,k=1,optimized=False,batch_size=512):
        if(not optimized):
            return self.predict_base(disType,k,batch_size)
        else:
            return self.predict_optimized(disType,k)
        
    def RuntimeComp(self, DisType='manhattan', k=1, batch_size=100): #! hardcoded best metric
        from sklearn.neighbors import KNeighborsClassifier #! used only in this function    
        start_time = time.time()
        self.predict(disType=DisType, k=k, optimized=True)
        self.optimized_time = time.time() - start_time

        start_time = time.time()
        self.predict(disType=DisType, k=k, optimized=False, batch_size=batch_size)
        self.non_optimized_time = time.time() - start_time
    
        temp_Xtrain = self.Xtrain.reshape(self.Xtrain.shape[0], -1)
        temp_Ytrain = self.Ytrain.ravel()

        start_time = time.time()
        knn_sklearn = KNeighborsClassifier(n_neighbors=k, algorithm='auto', metric=DisType)
        knn_sklearn.fit(temp_Xtrain, temp_Ytrain)
        self.sklearn_time = time.time() - start_time

        #! the following formatting and plotting is LLM generated

        print(f"Optimized time: {self.optimized_time:.2f} seconds")
        print(f"Non-optimized time: {self.non_optimized_time:.2f} seconds")
        print(f"sklearn time: {self.sklearn_time:.2f} seconds")

        methods = ['Optimized', 'Non-Optimized', 'sklearn']
        times = [self.optimized_time, self.non_optimized_time, self.sklearn_time]

        plt.figure(figsize=(10, 6))
        plt.bar(methods, times, color=['blue', 'orange', 'green'])
        plt.xlabel('Method')
        plt.ylabel('Time (seconds)')
        plt.title('Comparison of KNN Prediction Times')
        plt.ylim(0, max(times) * 1.1)  # Add some space above the tallest bar
        plt.savefig("./assignments/1/figures/time-comp.jpg")
        plt.show()
        
        #! End of LLM code
        return [self.optimized_time, self.non_optimized_time, self.sklearn_time]
    
        
    def overall(self):
        return super().overall() #! for overall accuracy
    
    def accuracy(self):
        return super().accuracy()
    
    def precision(self):
        return super().precision()
    
    def recall(self):
        return super().recall()
    
    def F1score(self):
        return super().F1score()
    
    def inference(self):
        return super().inference()
    
    def plot_confusion_matrix(self):
        # Generate the confusion matrix
        cm = confusion_matrix(self.Ytest, self.Ypred)
        labels = np.unique(self.Ytest)
        
        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        
        # Save the plot
        plt.savefig("./assignments/1/figures/EDA/confusion_matrix.jpg")
        plt.show()
    