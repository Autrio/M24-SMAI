import numpy as np
import matplotlib.pyplot as plt

class Data:
    def load_data(self, path):
        self.data = np.genfromtxt(path, delimiter=',', skip_header=1)
        np.random.default_rng().shuffle(self.data)
        self.X, self.Y = self.data[:, :1], self.data[:, 1]

    def split(self,SET=None,test=0.1, valid=0.1):
        Set = SET
        if Set is None:
            Set = self.X
        
        total_size = len(self.X)
        test_size = int(total_size * test)
        valid_size = int(total_size * valid)
        self.Xtrain, self.Ytrain = Set[test_size + valid_size:], self.Y[test_size + valid_size:]
        self.Xtest, self.Ytest = Set[:test_size], self.Y[:test_size]
        self.Xval, self.Yval = Set[test_size:test_size + valid_size], self.Y[test_size:test_size + valid_size]

class LinearRegressor(Data):
    def __init__(self, LR=0.01, epochs=10000):
        super().__init__()
        self.LR = LR
        self.epochs = epochs
        self.reg_type = 0
        self.reg_param = None        

    def fit(self,reg_type=None,reg_param=0,GIF=False):
        self.reg_type = reg_type
        self.reg_param = reg_param
        nsamp, nfeat = self.X.shape
        self.W = np.zeros(nfeat)
        self.B = 0.01

        for i in range(self.epochs):
            self.Ypred = self.X @ self.W + self.B
            self.err = self.Ypred - self.Y
            
            if(self.reg_type=='L2'):
                self.dW = (self.X.T @ self.err + self.reg_param * self.W) / nsamp
            elif(self.reg_type=='L1'):
                self.dW = (self.X.T @ self.err + self.reg_param * np.sign(self.W)) / nsamp
            else:
                self.dW = (self.X.T @ self.err + self.W) / nsamp

            self.dB = 2 * np.mean(self.err)
            self.W -= self.LR * self.dW
            self.B -= self.LR * self.dB

            if(GIF):
                self.plotIter(i)
            
            if self.Xval is not None and self.Yval is not None and i % 500 == 0:
                self.YvalPred = self.Xval @ self.W + self.B

            

    def predict(self, X):
        return np.dot(X, self.W) + self.B

    def MSE(self, x, y):
        return np.mean((x - y) ** 2)

    def STD(self, x):
        return np.std(x)

    def VAR(self, x):
        return np.var(x)

    def BIAS(self, x, y):
        return self.mse(x, y) - self.var(x)
    
    def polynomial(self, X, d):
        return np.hstack([X**n for n in range(d + 1)])
    
    def run(self, maxD):
        for d in range(1, maxD + 1):
            self.temp = self.X
            self.X = self.polynomial(self.X, d)
            self.split()
            self.fit(reg_type="L2",reg_param=5)

            self.Ytest_pred = self.predict(self.Xtest)
            self.CurrMSE = self.MSE(self.Ytest,self.Ytest_pred)
            self.Ytrain_pred = self.predict(self.Xtrain)

            self.train_MSE=self.MSE(self.Ytrain,self.Ytrain_pred)
            self.test_MSE = self.MSE(self.Ytest, self.Ytest_pred)

            self.train_STD=self.STD(self.Ytrain_pred)
            self.test_STD = self.STD(self.Ytest_pred)

            self.train_VAR=self.VAR(self.Ytrain_pred)
            self.test_VAR = self.VAR(self.Ytest_pred)

            self.X = self.temp
            self.minMSE = 1e6

            if self.CurrMSE < self.minMSE:
                self.best = d
                self.minMSE = self.CurrMSE
                with open('./models/linearRegression/best_model.txt', 'w') as f:
                    f.write("# Bias\n")
                    f.write(f'{self.B} \n')
                    f.write("\n# Weights\n"),
                    np.savetxt(f, self.W, delimiter=',', fmt='%d')

            plt.scatter(self.Xtrain[:, 1], self.Ytrain, color='red', label='Train')
            plt.scatter(self.Xval[:, 1], self.Yval, color='green', label='Validation')
            plt.scatter(self.Xtest[:, 1], self.Ytest, color='blue', label='Test')
            idx = np.argsort(self.Xtrain[:, 1])
            plt.plot(self.Xtrain[:, 1][idx], self.Ytrain_pred[idx], color='black', label='Model Prediction')
            plt.title(f'Plot for degree={d}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            textstr = (f'Train MSE: {self.train_MSE:.4f}\n'
            f'Test MSE: {self.test_MSE:.4f}\n'
            f'Train STD: {self.train_STD:.4f}\n'
            f'Test STD: {self.test_STD:.4f}\n'
            f'Train VAR: {self.train_VAR:.4f}\n'
            f'Test VAR: {self.test_VAR:.4f}')

            plt.gcf().text(0.98, 0.02, textstr, fontsize=10, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
            plt.savefig(f"./assignments/1/figures/fits/degree-{d}.jpg") if self.reg_type is None else plt.savefig(f"./assignments/1/figures/regularised/degree-{d}.jpg")
            plt.show()
        print(f'Best fitting model degree={self.best} with val_mse={self.minMSE}')


    def GIFplots(self,d,path):

        self.XK = self.polynomial(self.X,d)
        self.split(SET = self.XK)
        nsamp, nfeat = self.XK.shape
        self.W = np.zeros(nfeat)
        self.B = 0.01

        self.mse_list = []
        self.std_list = []
        self.var_list = []
        self.iterations = []
        # return

        for i in range(self.epochs):
            assert((self.XK @ self.W).shape[0]==nsamp)
            
            self.Ypred = self.XK @ self.W + self.B
            self.err = self.Ypred - self.Y
            
            if(self.reg_type=='L2'):
                self.dW = (self.XK.T @ self.err + self.reg_param * self.W) / nsamp
            elif(self.reg_type=='L1'):
                self.dW = (self.XK.T @ self.err + self.reg_param * np.sign(self.W)) / nsamp
            else:
                self.dW = (self.XK.T @ self.err + self.W) / nsamp

            self.dB = 2 * np.mean(self.err)
            self.W -= self.LR * self.dW
            self.B -= self.LR * self.dB
            if i % 20 == 0 and i < 3000: 
                self.Ytrain_pred = self.predict(self.Xtrain)
                self.train_MSE = self.MSE(self.Ytrain, self.Ytrain_pred)
                self.train_STD = self.STD(self.Ytrain_pred)
                self.train_VAR = self.VAR(self.Ytrain_pred)

                # Plot the original data and the line being fitted
                # Append metrics to lists
                self.mse_list.append(self.train_MSE)
                self.std_list.append(self.train_STD)
                self.var_list.append(self.train_VAR)
                self.iterations.append(i)

                # Create a 2x2 grid of subplots
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                
                # Top-left: Original data and fitting line
                axs[0, 0].scatter(self.Xtrain[:, 1], self.Ytrain, color='yellow', label='Train')
                idx = np.argsort(self.Xtrain[:, 1])
                axs[0, 0].plot(self.Xtrain[:, 1][idx], self.Ytrain_pred[idx], color='black', label=f'Fitting at Iter {i}')
                axs[0, 0].set_title(f'Iteration {i}: Fitting curve')
                axs[0, 0].set_xlabel('x')
                axs[0, 0].set_ylabel('y')
                axs[0, 0].legend()

                # Top-right: MSE vs iteration
                axs[0, 1].plot(self.iterations, self.mse_list, color='red')
                axs[0, 1].set_title('MSE vs Iteration')
                axs[0, 1].set_xlabel('Iteration')
                axs[0, 1].set_ylabel('MSE')

                # Bottom-left: STD vs iteration
                axs[1, 0].plot(self.iterations, self.std_list, color='blue')
                axs[1, 0].set_title('STD vs Iteration')
                axs[1, 0].set_xlabel('Iteration')
                axs[1, 0].set_ylabel('STD')

                # Bottom-right: VAR vs iteration
                axs[1, 1].plot(self.iterations, self.var_list, color='green')
                axs[1, 1].set_title('VAR vs Iteration')
                axs[1, 1].set_xlabel('Iteration')
                axs[1, 1].set_ylabel('VAR')

                # Adjust layout and save
                plt.tight_layout()
                plt.savefig(f'./assignments/1/figures/{d}/plot{i}.jpg', format='jpg')
                plt.clf()
            # plt.show()
                # plt.show()
        
