import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from tqdm.auto import tqdm
import wandb
from icecream import ic
import copy

class Data:
    """
    Class for Data Handling
    """
    def __init__(self) -> None:
        pass

    def load_raw_data(self, path):
        """
        Load raw data from a CSV file.

        Parameters:
        path (str): Path to the CSV file.
        """
        self.path = path
        self.df = pd.read_csv(self.path)


    def desc_data(self):
        """
        Display descriptive statistprints of the data.
        """
        summary = {
            'Maximum': [],
            'Minimum': [],
            'Mean': [],
            'Std Dev': []
        }
        
        for column in self.df.columns:
            summary['Maximum'].append(self.df[column].max())
            summary['Minimum'].append(self.df[column].min())
            summary['Mean'].append(np.mean(self.df[column]))
            summary['Std Dev'].append(np.std(self.df[column]))
        
        self.summary_df = pd.DataFrame(summary, index=self.df.columns)
        print(self.summary_df)


    def plot_data(self,task="classification"):
        """
        Plot histograms with KDE for each feature in the dataset.
        """
        for column in self.df.columns:
            plt.figure(figsize=(12,10))
            sns.histplot(data=self.df, x=column, kde=True)
            plt.title(f"Distribution of {column}")
            plt.ylabel('Frequency')
            plt.xlabel(column)
            plt.xticks(rotation=45)
            # plt.show()
            if(task=="classification"):
                plt.savefig(f"/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/assignments/3/figures/2/{column}-wineQT")
            else:
                plt.savefig(f"/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/assignments/3/figures/3/{column}-HousingData")


    def Preprocess(self,drop=["Id"],nan=None,categorical=None,multi_hot=None):
        """
        Preprocess the data by handling missing values and dropping irrelevant columns.
        """
        if self.df is None:
            print("Dataframe is empty. Load data first.")
            return
        
        if nan is not None:
            for col in nan:
                self.df[col] = self.df[col].fillna(self.df[col].mean(), inplace=False)

        if drop is not None:
            self.df.drop(columns=drop, inplace=True)

        if categorical is not None:
            self.categorical_features = categorical
            for feature in self.categorical_features: 
                self.df[feature], _ = pd.factorize(self.df[feature])

        if multi_hot is not None:
            all_labels = set()
            for col in multi_hot:
                for labels in self.df[col]:
                    all_labels.update(labels.split())

            self.all_labels = sorted(list(all_labels))

            for col in multi_hot:
                self.df[col] = self.df[col].apply(
                    lambda x: [1 if label in x.split() else 0 for label in self.all_labels]
    )

        self.numerical_features = self.df.columns
        
        if multi_hot is not None:
            self.numerical_features = list(set(self.numerical_features) - set(multi_hot))
        if categorical is not None:        
            self.numerical_features = list(set(self.numerical_features) - set(self.categorical_features))

        imputer = SimpleImputer(strategy='mean')
        self.df[self.numerical_features] = pd.DataFrame(
            imputer.fit_transform(self.df[self.numerical_features]), 
            columns=self.numerical_features
        )


    def Split(self, test=0.1, valid=0.1, targ='quality',task="single_label"):
        """
        Split the dataset into training, validation, and testing sets.

        Parameters:
        test (float): Proportion of the dataset to include in the test split.
        valid (float): Proportion of the dataset to include in the validation split.
        targ (str): Name of the target variable.
        """
        if self.df is None:
            print("Dataframe is empty. Load and preprocess data first.")
            return

        if targ not in self.df.columns:
            print(f"Target column '{targ}' not found in dataframe.")
            return

        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        total_samples = len(self.df)
        test_size = int(total_samples * test)
        valid_size = int(total_samples * valid)

        self.Xdf = self.df.drop(columns=[targ], axis=1).values
        if task=="multi_label_classification":
            self.Ydf = np.array(self.df[targ].tolist(),dtype=np.float64)
        elif task=="single_label_classification":
            self.Ydf =  pd.get_dummies(self.df['quality'], dtype=np.float64).values
        elif task=="regression":
            self.Ydf = np.array(self.df[targ].tolist(),dtype=np.float64)
        else:
            print("invalid split initialization")
            exit()

        print(self.Ydf.shape)
        print(self.Xdf.shape)

        self.Xtest = self.Xdf[:test_size]
        self.Xval = self.Xdf[test_size:test_size + valid_size]
        self.Xtrain = self.Xdf[test_size + valid_size:]

        self.Ytest = self.Ydf[:test_size]
        self.Yval = self.Ydf[test_size:test_size + valid_size]
        self.Ytrain = self.Ydf[test_size + valid_size:]

        self.num_classes = self.df[targ].nunique() if task=="single_label_classification" else 1 if task=="regression" else self.Ytrain.shape[1]

        print(f"Data split into training ({len(self.Xtrain)} samples), "
              f"validation ({len(self.Xval)} samples), "
              f"and testing ({len(self.Xtest)} samples) sets.")
        print(f"Number of classes: {self.num_classes}")

    def Normalize(self):
        """
        Normalize the feature data using z-score normalization.
        """
        if self.Xtrain is None:
            print("Training data not found. Split data first.")
            return

        mean = self.Xtrain.mean(axis=0)
        std = self.Xtrain.std(axis=0)

        self.Xtrain = (self.Xtrain - mean) / std
        self.Xtest = (self.Xtest - mean) / std
        self.Xval = (self.Xval - mean) / std

        print("Feature data normalized using z-score normalization.")

class Metrics:
    def accuracy_score(self,y_true, y_pred):
        """
        Calculate the accuracy score, which is the ratio of correctly predicted
        labels to the total number of labels.

        Parameters:
        y_true (np.ndarray): Ground truth (correct) labels.
        y_pred (np.ndarray): Predicted labels.

        Returns:
        float: Accuracy score.
        """
        assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match."
        accuracy = np.mean(y_true==y_pred)
        return accuracy
    
    def r2_score(self,y_true, y_pred):
        """
        Calculate the R^2 score (coefficient of determination), which indicates the
        goodness of fit of the predicted values.

        Parameters:
        y_true (np.ndarray): Ground truth (correct) labels.
        y_pred (np.ndarray): Predicted labels.

        Returns:
        float: R^2 score.
        """
        assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match."
        y_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_mean) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)        
        r2 = 1 - (ss_residual / ss_total)
        return r2

    def multilabel_accuracy(self, y_pred, y):
        """
        Calculate accuracy for multilabel classification
        using a threshold of 0.5 for prediction
        """
        predictions = (y_pred > 0.5).astype(int)
        return np.mean(predictions == y)
    
    def precision_score(self, y_true, y_pred, average='macro'):
        """
        Calculate the precision score.
        
        Parameters:
        - y_true (np.ndarray): Ground truth labels.
        - y_pred (np.ndarray): Predicted labels.
        - average (str): Type of averaging performed on the data.
                          'macro' | 'micro' | 'weighted' | 'binary'
        
        Returns:
        - precision (float): Precision score.
        """
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        unique_classes = np.unique(y_true)
        precisions = []

        for cls in unique_classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(precision)

        if average == 'macro':
            return np.mean(precisions)
        elif average == 'weighted':
            weights = np.array([np.sum(y_true == cls) for cls in unique_classes])
            return np.average(precisions, weights=weights)
        elif average == 'binary':
            if len(unique_classes) != 2:
                raise ValueError("Binary precision requires exactly two classes.")
            return precisions[1] 
        else:
            raise ValueError("Unsupported average type. Choose from 'macro', 'micro', 'weighted', 'binary'.")

    def recall_score(self, y_true, y_pred, average='macro'):
        """
        Calculate the recall score.
        """
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        unique_classes = np.unique(y_true)
        recalls = []

        for cls in unique_classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall)

        if average == 'macro':
            return np.mean(recalls)
        elif average == 'weighted':
            weights = np.array([np.sum(y_true == cls) for cls in unique_classes])
            return np.average(recalls, weights=weights)
        elif average == 'binary':
            if len(unique_classes) != 2:
                raise ValueError("Binary recall requires exactly two classes.")
            return recalls[1]  
        else:
            raise ValueError("Unsupported average type. Choose from 'macro', 'micro', 'weighted', 'binary'.")

    def f1_score(self, y_true, y_pred, average='macro'):
        """
        Calculate the F1 score.
        """
        precision = self.precision_score(y_true, y_pred, average=average)
        recall = self.recall_score(y_true, y_pred, average=average)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def hamming_loss(self, y_true, y_pred):
        """
        Calculate the Hamming Loss for multi-label classification.
        
        Parameters:
        - y_true (np.ndarray): Ground truth binary labels (shape: [n_samples, n_labels]).
        - y_pred (np.ndarray): Predicted binary labels (shape: [n_samples, n_labels]).
        
        Returns:
        - float: Hamming Loss.
        """
        assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match."
        mismatches = np.not_equal(y_true, y_pred)
        hamming = mismatches.mean()
        return hamming
    
    def mae(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error (MAE).
        """
        assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match."
        mae = np.mean(np.abs(y_true - y_pred))
        return mae
    
    def mse(self, y_true, y_pred):
        """
        Calculate Mean Squared Error (MSE).
        """
        assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match."
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    def rmse(self, y_true, y_pred):
        """
        Calculate Root Mean Squared Error (RMSE).
        """
        mse = self.mse(y_true, y_pred)
        rmse = np.sqrt(mse)
        return rmse

    def mae(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error (MAE).
        """
        assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match."
        mae = np.mean(np.abs(y_true - y_pred))
        return mae
        
class BatchLoader:
    def __init__(self,X,Y,batch_size,train=True):
        self.X = X
        self.Y = Y
        self.i = 0
        self.batch_size = batch_size
        
    def __iter__(self):
        return self
    
    def __len__(self):
        return int(len(self.X)/self.batch_size)
    
    def __next__(self):
        
        start, end = self.batch_size * self.i, self.batch_size * (self.i + 1)
        if end > len(self.X):
            self.i = 0
            raise StopIteration

        self.i = (self.i + 1)                  
        return self.X[start: end, ], self.Y[start : end, ]

class activations:
    class Relu:
        def __call__(self, x):
            return np.maximum(np.zeros_like(x), x)    
        def __str__(self):
            return "ReLU"
        def grad(self, x):
            return (x >= 0).astype(np.float64)

    class Sigmoid:
        def __call__(self, x):
            out = np.empty_like(x)
            positive_mask = x >= 0
            negative_mask = ~positive_mask
            out[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
            exp_x = np.exp(x[negative_mask])
            out[negative_mask] = exp_x / (1 + exp_x)
            return out

        def __str__(self):
            return "Sigmoid"

        def grad(self, x):
            sig = self.__call__(x)
            return sig * (1 - sig)
        
    class Linear:
        def __call__(self, x):
            return x
        def __str__(self):
            return "Linear"
        def grad(self, x):
            return np.ones_like(x)

    class Tanh:
        def __call__(self, x):
            return np.tanh(x)
                
        def __str__(self):
            return "Tanh"
        
        def grad(self, x):
            dth = self.__call__(x)
            return 1 - dth ** 2
            
    class Softmax:
        def __call__(self, x):
            shiftx = x - np.max(x, axis=1, keepdims=True)
            exps = np.exp(shiftx)
            return exps / np.sum(exps, axis=1, keepdims=True)
        def __str__(self):
            return "Softmax"
        def grad(self, x):
            return np.ones_like(x)
    
class loss:
    class MSELoss:
        def __call__(self, y, y_pred):
            self.y, self.y_pred = y, y_pred
            return np.mean((self.y - self.y_pred) ** 2)
        def grad(self):
            return -2 * (self.y - self.y_pred)
        
    class BCELoss:
        def __call__(self, y, y_pred):
            self.y = y
            self.y_pred = y_pred
            e = 1e-15
            y_pred = np.clip(y_pred, e, 1 - e)
            return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            
        def grad(self):
            e = 1e-15
            y_pred = np.clip(self.y_pred, e, 1 - e)
            return (-(self.y / y_pred) + (1 - self.y) / (1 - y_pred))
            # return self.y_pred - self.y
        
    class CELoss:
        def __call__(self, y, y_pred):
            eps = 1e-15  
            y_pred = np.clip(y_pred, eps, 1 - eps)
            self.y, self.y_pred = y, y_pred
            return np.mean(np.sum(-y * np.log(y_pred), axis=1)) 

        def grad(self):
            return self.y_pred - self.y

class network:
    class Neuron:
        def __init__(self, dim_in, activation):
            self.dzw, self.dzx, self.daz = 0, 0, 0
            self.dim_in = dim_in
            self.activation = activation
        
        def get_grads(self):
            return [self.dzw, self.dzx, self.daz]
            
        def calculate_grad(self, x, z, w, index):
            self.dzw = x
            self.dzx = w[index]
            self.daz = self.activation.grad(z[:, index])        
            return [self.dzw, self.dzx, self.daz]
        
    class Layer:
        def __init__(self, dim_in, dim_out, activation):
            self.dim_in = dim_in
            self.dim_out = dim_out
            self.activation = activation
            
            self.W = np.random.randn(self.dim_out, self.dim_in) * np.sqrt(2 / self.dim_in)
            self.b = np.zeros(self.dim_out)
            
            self.neurons = [network.Neuron(self.dim_in, activation) for _ in range(self.dim_out)]
            
            self.dzw, self.dzx, self.daz = [], [], []
            
        
        def get_grads(self):
            grads = [np.stack(self.dzw, axis=1),
                    np.stack(self.dzx, axis=-1), 
                    np.stack(self.daz, axis=-1)]

            self.dzw.clear()
            self.dzx.clear()
            self.daz.clear()
            return grads
            
        def __str__(self):
            return(f"Layer: [in:{self.dim_in}] [out:{self.dim_out}] [activation:{self.activation}]")
            
        def __call__(self, x):
            '''
                x: (bs, dim_in)
            '''
            
            if x.shape[1] != self.dim_in:
                raise TypeError(f'Input should have dimension {self.dim_in} but found {x.shape[1]}')
            
            z = x @ self.W.T + self.b
            self.a = self.activation(z)

            self.daz.clear()
            self.dzx.clear()
            self.dzw.clear()

            for i, neuron in enumerate(self.neurons):
                dzw, dzx, daz = neuron.calculate_grad(x, z, self.W, i)
                self.dzw.append(dzw)
                self.dzx.append(dzx)
                self.daz.append(daz)
                
            return self.a

class Model(Data):
    def __init__(self,params):
        super().__init__()
        self.loss_fn = params["loss_fn"]
        self.layers = []
        self.lr = params["lr"]
        self.dW, self.dB = [], []
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1': [], 'val_f1': [],
            'train_hamming_loss': [], 'val_hamming_loss': [],
            'train_R2': [], 'val_R2': [],
            'train_mse': [], 'val_mse': [],
            'train_rmse': [], 'val_rmse': [],
            'train_mae': [], 'val_mae': []
        }

        self.logger = params["logger"] if "logger" in params.keys() else wandb
        self.metrics = Metrics()
        self.type = params["type"]
        self.accuracy = self.classification_accuracy if self.type=='single_label_classification' else self.regression_accuracy if self.type=="regression" else self.multilabel_accuracy
        self.batch_size = self.Xtrain.shape[0] if params["optimizer"]=="batch" else (1 if params["optimizer"] == "SGD" else params["batch_size"])
        self.final_act = activations.Sigmoid() if self.type == "multi_label_classification" else activations.Softmax() if self.type=="single_label_classification" else activations.Linear()
        if self.type in ["single_label_classification", "multi_label_classification"]:
            self.precision = self.classification_precision
            self.recall = self.classification_recall
            self.f1 = self.classification_f1
            if self.type == "multi_label_classification":
                self.hamming_loss_fn = self.hamming_loss
        else:
            self.precision = None
            self.recall = None
            self.f1 = None
            self.hamming_loss_fn = None

        self.early_stopping = params.get("early_stopping", False)
        self.patience = params.get("patience", 512)
        self.min_delta = params.get("min_delta", 0.0)
        self.restore_best_weights = params.get("restore_best_weights", True)
        
        self.best_val_loss = np.Inf
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.best_weights = None

    def classification_accuracy(self, y_pred, y):
        return self.metrics.accuracy_score(np.argmax(y, axis=-1), np.argmax(y_pred, axis=-1))
        
    def regression_accuracy(self, y_pred, y):
        return self.metrics.r2_score(y, y_pred)
    
    def multilabel_accuracy(self, y_pred, y):
        return self.metrics.multilabel_accuracy(y_pred=y_pred,y=y)
    
    def classification_precision(self, y_pred, y_true, average='macro'):
        return self.metrics.precision_score(y_true, y_pred, average=average)

    def classification_recall(self, y_pred, y_true, average='macro'):
        return self.metrics.recall_score(y_true, y_pred, average=average)

    def classification_f1(self, y_pred, y_true, average='macro'):
        return self.metrics.f1_score(y_true, y_pred, average=average)
    
    def hamming_loss(self, y_pred, y_true):
        return self.metrics.hamming_loss(y_true, y_pred)
                
    def __str__(self):
        """
        Model Description
        """
        out = ""
        for layer in self.layers:
            out += layer.__str__() + "\n"
        return out

    def add(self, layer):
        self.layers.append(layer)

    def __call__(self, x):  #! this is forward propagation step
        '''
            x: (bs, dim_in)
        '''
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self):
        dLy = self.loss_fn.grad()

        for i in range(len(self.layers)-1, -1, -1):
            dzw, dzx, daz = self.layers[i].get_grads()
            if i != len(self.layers) - 1:
                dLy = dLy @ self.layers[i + 1].W
            dLy = dLy * daz
            dw = dLy[:, :, None] * dzw
            db = dLy[:, :] * 1
            
            self.dW.append(np.mean(dw, axis=0))
            self.dB.append(np.mean(db, axis=0))
            
    def update(self):
        for i, (dw, db) in enumerate(zip(reversed(self.dW), reversed(self.dB))):
            self.layers[i].W += - self.lr * dw
            self.layers[i].b += - self.lr * db
            
        self.dW.clear()
        self.dB.clear()

    def save_weights(self):
        self.best_weights = [ (layer.W.copy(), layer.b.copy()) for layer in self.layers ]

    def load_weights(self):
        if self.best_weights is not None:
            for layer, (W, b) in zip(self.layers, self.best_weights):
                layer.W = W.copy()
                layer.b = b.copy()
        else:
            print("No weights have been saved yet.")
        
    def train_step(self, loader):
        loss, acc, precision, recall, f1 ,hamming , mae= 0, 0, 0, 0, 0, 0, 0
        all_preds = []
        all_labels = []
        for x, y in loader:
            y_pred = self.__call__(x)
            loss += self.loss_fn(y, y_pred)
            acc += self.accuracy(y_pred, y)

            if self.type in ["single_label_classification", "multi_label_classification"]:
                all_preds.append(y_pred)
                all_labels.append(y)
            
            self.backward()
            self.update()

            if np.isnan(loss):
                self.flag=1
            else:
                self.flag=0
        
        loss /= len(loader)
        acc /= len(loader)

        if self.type in ["single_label_classification", "multi_label_classification"]:
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            if self.type == "single_label_classification":
                all_preds = np.argmax(all_preds, axis=1)
                all_labels = np.argmax(all_labels, axis=1)
            elif self.type == "multi_label_classification":
                all_preds = (all_preds > 0.5).astype(int)
                all_labels = all_labels.astype(int)
            
            precision = self.precision(all_preds, all_labels)
            recall = self.recall(all_preds, all_labels)
            f1 = self.f1(all_preds, all_labels)

            if self.type == "multi_label_classification":
                hamming = self.hamming_loss_fn(all_preds, all_labels)
        else:
            precision, recall, f1 ,hamming= None, None, None,None

        if self.type=="regression":
            mae = self.metrics.mae(y,y_pred)
        else:
            mae = None

        return loss, acc, precision, recall, f1, hamming, mae
            
    
    def validate_step(self, loader):
        loss, acc, precision, recall, f1, hamming, mae = 0, 0, 0, 0, 0, 0, 0
        all_preds = []
        all_labels = []
        for x, y in loader:
            y_pred = self.__call__(x)
            loss += self.loss_fn(y, y_pred)
            acc += self.accuracy(y_pred, y)
            
            if self.type in ["single_label_classification", "multi_label_classification"]:
                all_preds.append(y_pred)
                all_labels.append(y)
        
        loss /= len(loader)
        acc /= len(loader)

        if self.type in ["single_label_classification", "multi_label_classification"]:
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            if self.type == "single_label_classification":
                all_preds = np.argmax(all_preds, axis=1)
                all_labels = np.argmax(all_labels, axis=1)
            elif self.type == "multi_label_classification":
                all_preds = (all_preds > 0.5).astype(int)
                all_labels = all_labels.astype(int)
            
            precision = self.precision(all_preds, all_labels)
            recall = self.recall(all_preds, all_labels)
            f1 = self.f1(all_preds, all_labels)

            if self.type == "multi_label_classification":
                hamming = self.hamming_loss_fn(all_preds, all_labels)
        else:
            precision, recall, f1 ,hamming= None, None, None,None

        if self.type=="regression":
            mae = self.metrics.mae(y,y_pred)
        else:
            mae = None

        return loss, acc, precision, recall, f1, hamming, mae

    def train(self, epochs):
        if self.type == "regression":
            train_loader = BatchLoader(X=self.Xtrain, Y=self.Ytrain.reshape(-1,1), train=True, batch_size=self.batch_size)
            val_loader = BatchLoader(X=self.Xval, Y=self.Yval.reshape(-1,1), train=False, batch_size=32)
        else:
            train_loader = BatchLoader(X=self.Xtrain, Y=self.Ytrain, train=True, batch_size=self.batch_size)
            val_loader = BatchLoader(X=self.Xval, Y=self.Yval, train=False, batch_size=32)

        with tqdm(total=epochs, desc="Training", unit="epoch") as pbar:
            for epoch in range(epochs):
                
                train_metrics = self.train_step(train_loader)
                val_metrics = self.validate_step(val_loader)

                train_loss, train_acc = train_metrics[0],train_metrics[1]
                val_loss, val_acc = val_metrics[0],val_metrics[1]

                
                pbar.set_postfix({"Train Acc": train_acc, "Val Acc": val_acc})
                pbar.update(1) 

                if epoch%100 == 0 or epoch == epochs - 1:
                    tqdm.write(f"Epoch: {epoch} \tTrain:[Loss: {train_loss:.4f}, Acc: {train_acc:.4f}] \tVal:[Loss: {val_loss:.4f}, Acc: {val_acc:.4f}]")
                
                if self.early_stopping:
                    if val_loss < self.best_val_loss - self.min_delta:
                        self.best_val_loss = val_loss
                        self.best_epoch = epoch
                        self.epochs_no_improve = 0
                        if self.restore_best_weights:
                            self.save_weights()
                    else:
                        self.epochs_no_improve += 1
                        if self.epochs_no_improve >= self.patience:
                            tqdm.write(f"Early stopping triggered at epoch {epoch+1}.")
                            if self.restore_best_weights and self.best_weights is not None:
                                self.load_weights()
                                tqdm.write(f"Model weights restored to epoch {self.best_epoch+1}.")
                                tqdm.write(f"best validation loss::{self.best_val_loss}" )
                            break

                self.log(train_metrics,val_metrics)

    def log(self,train_metrics,val_metrics):

        if self.flag:
            return

        train_loss, train_acc, train_precision, train_recall, train_f1, train_hamming, train_mae = train_metrics
        val_loss, val_acc, val_precision, val_recall, val_f1, val_hamming, val_mae  = val_metrics

        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        
        if self.type in ["single_label_classification", "multi_label_classification"]:
            self.history['train_precision'].append(train_precision)
            self.history['val_precision'].append(val_precision)
            self.history['train_recall'].append(train_recall)
            self.history['val_recall'].append(val_recall)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            if self.type == "multi_label_classification":
                self.history['train_hamming_loss'].append(train_hamming)
                self.history['val_hamming_loss'].append(val_hamming)
        if self.type=="regression":
            self.history["train_R2"].append(train_acc)
            self.history["val_R2"].append(val_acc)
            self.history["train_mse"].append(train_loss)
            self.history["val_mse"].append(val_loss)
            self.history["train_rmse"].append(np.sqrt(train_loss))
            self.history["val_rmse"].append(np.sqrt(val_loss))
            self.history["train_mae"].append(np.sqrt(train_mae))
            self.history["val_mae"].append(np.sqrt(val_mae))


        if self.logger is not None:
            log_dict = {
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss
            }
            if self.type =="single_label_classification":
                log_dict.update({
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc,'train_f1': [], 'val_f1': [],
                    "val_loss": val_loss,
                    "train_precision": train_precision,
                    "val_precision": val_precision,
                    "train_recall": train_recall,
                    "val_recall": val_recall,
                    "train_f1": train_f1,
                    "val_f1": val_f1
                })
            elif self.type == "multi_label_classification":
                log_dict.update({
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "train_precision": train_precision,
                    "val_precision": val_precision,
                    "train_recall": train_recall,
                    "val_recall": val_recall,
                    "train_f1": train_f1,
                    "val_f1": val_f1,
                    "train_hamming_loss": train_hamming,
                    "val_hamming_loss": val_hamming
                })
            elif self.type == "regression":
                log_dict.update({
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "train_mse": train_loss,
                    "val_mse": val_loss,
                    "train_rmse":np.sqrt(train_loss),
                    "val_rmse":np.sqrt(val_loss),
                    "train_mae":np.sqrt(train_mae),
                    "val_mae":np.sqrt(val_mae)
                })
            self.logger.log(log_dict)

    def predict(self,x,task="classificaion"):
        y = self.__call__(x)
        if task=="regression":
            return y
        return (y>0.5).astype(np.float32)
    
    
    def arch_base(self):
        self.layers.clear()
        self.add(network.Layer(self.Xtrain.shape[1], 16, activations.Tanh()))
        self.add(network.Layer(16, 64, activations.Tanh()))
        self.add(network.Layer(64, 128, activations.Tanh()))
        self.add(network.Layer(128, 256, activations.Tanh()))
        self.add(network.Layer(256, self.num_classes, self.final_act))

    def arch1(self,activation):
        self.layers.clear()
        self.add(network.Layer(self.Xtrain.shape[1], 8, activation))
        self.add(network.Layer(8, 16, activation))
        self.add(network.Layer(16, 16, activation))
        self.add(network.Layer(16, 8, activation))
        self.add(network.Layer(8, self.num_classes, self.final_act))
    
    def arch2(self,activation):
        self.layers.clear()
        self.add(network.Layer(self.Xtrain.shape[1], 16, activation))
        self.add(network.Layer(16, 32, activation))
        self.add(network.Layer(32, 64, activation))
        self.add(network.Layer(64, 32, activation))
        self.add(network.Layer(32, 16, activation))
        self.add(network.Layer(16, self.num_classes, self.final_act))

    def arch3(self,activation):
        self.layers.clear()
        self.add(network.Layer(self.Xtrain.shape[1], 32, activation))
        self.add(network.Layer(32, 128, activation))
        self.add(network.Layer(128, 128, activation))
        self.add(network.Layer(128, 32, activation))
        self.add(network.Layer(32, self.num_classes, self.final_act))

    def arch4(self,activation):
        self.layers.clear()
        self.add(network.Layer(self.Xtrain.shape[1], 16, activation))
        self.add(network.Layer(16, 32, activation))
        self.add(network.Layer(32, 32, activation))
        self.add(network.Layer(32, 16, activation))
        self.add(network.Layer(16, self.num_classes, self.final_act))

    def arch5(self,activation):
        self.layers.clear()
        self.add(network.Layer(self.Xtrain.shape[1], 16, activation))
        self.add(network.Layer(16, 64, activation))
        self.add(network.Layer(64, 128, activation))
        self.add(network.Layer(128, 256, activation))
        self.add(network.Layer(256, self.num_classes, self.final_act))

    def set_arch(self,arch,activation,num_classes = None):
        if num_classes is not None:
            self.num_classes = num_classes 
        return arch(self, activation)
    
    def load_raw_data(self, path):
        return super().load_raw_data(path)
    
    def Preprocess(self, drop=["Id"], nan=None, categorical=None, multi_hot=None):
        return super().Preprocess(drop, nan, categorical, multi_hot)
    
    def Split(self, test=0.1, valid=0.1, targ='quality', task="single_label"):
        return super().Split(test, valid, targ, task)
    
    def Normalize(self):
        return super().Normalize()

class Autoencoder(Model):
    def __init__(self, params, encoding_dim=7): #! 7 is optimal dims from PCA
        """
        Initialize the Autoencoder with specified parameters and encoding dimension.

        Parameters:
        params (dict): Dictionary containing model parameters such as learning rate, optimizer, etc.
        encoding_dim (int): Dimension of the encoded representation.
        """
        super().__init__(params)
        self.encoding_dim = encoding_dim
        self.loss_fn = loss.MSELoss()
        self.activation = params["activation"]

    def set_arch(self):
        """
        Build the encoder-decoder architecture for the autoencoder.
        """
        self.layers.clear()
        
        input_dim = self.Xtrain.shape[1]
        
        self.add(network.Layer(input_dim, 32, self.activation))
        # self.add(network.Layer(64, 32, self.activation))
        self.add(network.Layer(32, 16, self.activation))
        self.add(network.Layer(16, 8, self.activation))
        self.add(network.Layer(8, self.encoding_dim, self.activation))
        
        self.add(network.Layer(self.encoding_dim, 8, self.activation))
        self.add(network.Layer(8, 16, self.activation))
        self.add(network.Layer(16, 32, self.activation))
        # self.add(network.Layer(32, 64, self.activation))
        self.add(network.Layer(32, input_dim, self.activation))  

    def load_raw_data(self, path):
        return super().load_raw_data(path)

    def Preprocess(self,predict='track_genre',IQRthresh = 5,CF = 0.8):

        self.attributes = self.df.columns

        self.labelData = ['track_genre']
        self.dropData = ['track_name', 'album_name', 'Unnamed: 0', 'track_id','artists']

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

    def Split(self, test=0.1, valid=0.1, targ="track_genre", task=None):

        if self.df is None:
            print("Dataframe is empty. Load and preprocess data first.")
            return
        
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        total_samples = len(self.df)
        test_size = int(total_samples * test)
        valid_size = int(total_samples * valid)

        self.Xdf = self.df.values 

        self.Ydf = copy.deepcopy(self.Xdf)

        self.targets = self.df[targ].values

        self.Xtest = self.Xdf[:test_size]
        self.Xval = self.Xdf[test_size:test_size + valid_size]
        self.Xtrain = self.Xdf[test_size + valid_size:]

        self.Ytest = self.Ydf[:test_size]
        self.Yval = self.Ydf[test_size:test_size + valid_size]
        self.Ytrain = self.Ydf[test_size + valid_size:]


        print(f"Data split into training ({len(self.Xtrain)} samples), "
              f"validation ({len(self.Xval)} samples), "
              f"and testing ({len(self.Xtest)} samples) sets.")
        
    def Normalize(self):
        """
        Normalize the feature data using z-score normalization.
        """
        if self.Xtrain is None:
            print("Training data not found. Split data first.")
            return

        mean = self.Xtrain.mean(axis=0)
        std = self.Xtrain.std(axis=0)

        self.Xtrain = (self.Xtrain - mean) / std
        self.Xtest = (self.Xtest - mean) / std
        self.Xval = (self.Xval - mean) / std

        mean = self.Ytrain.mean(axis=0)
        std = self.Ytrain.std(axis=0)

        self.Ytrain = (self.Ytrain - mean) / std
        self.Ytest = (self.Ytest - mean) / std
        self.Yval = (self.Yval - mean) / std

        print("Feature data normalized using z-score normalization.")
            
    def train_step(self, loader):
        total_loss = 0
        for x, y in loader:
            y_pred = self.__call__(x)
            loss = self.loss_fn(y, y_pred)
            total_loss += loss
            self.backward()
            self.update()
        return total_loss / len(loader)
    def validate_step(self, loader):

        total_loss = 0
        for x, y in loader:
            y_pred = self.__call__(x)
            loss = self.loss_fn(y, y_pred)
            total_loss += loss
        return total_loss / len(loader)

    def train(self, epochs):

        train_loader = BatchLoader(X=self.Xtrain, Y=self.Ytrain, batch_size=self.batch_size)
        val_loader = BatchLoader(X=self.Xval, Y=self.Yval, batch_size=self.batch_size)

        with tqdm(total=epochs, desc="Training Autoencoder", unit="epoch") as pbar:
            for epoch in range(epochs):
                train_loss = self.train_step(train_loader)
                val_loss = self.validate_step(val_loader)

                pbar.set_postfix({"Train Loss": f"{train_loss:.4f}", "Val Loss": f"{val_loss:.4f}"})
                pbar.update(1)

                if epoch % 10 == 0:
                    tqdm.write(f"Epoch: {epoch+1}/{epochs} \tTrain Loss: {train_loss:.4f} \tVal Loss: {val_loss:.4f}")

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)

                if self.logger is not None:
                    self.logger.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss
                    })

    def encode(self, x):
        for layer in self.layers[:4]: 
            x = layer(x)
        return x

    def decode(self, encoded):
        for layer in self.layers[4:]:
            encoded = layer(encoded)
        return encoded

    def reconstruct(self, x):
        return self.decode(self.encode(x))
     
    def __call__(self, x):
        return super().__call__(x)
