from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

class Data:
    def __init__(self) -> None:
        pass

    def load_raw_data(self, path):
        self.path = path
        self.df = pd.read_feather(self.path)
        
    def preprocess(self):
        self.words = self.df.iloc[:, 0].values
        self.embeds = np.stack(self.df.iloc[:, 1].values)

class GMM(Data):
    def __init__(self, num_comp=3, max_iter=100, tol=1e-6) -> None:
        super().__init__()
        self.num_comp = num_comp
        self.max_iter = max_iter
        self.tol = tol

    def load_raw_data(self, path):
        return super().load_raw_data(path)
    
    def preprocess(self):
        return super().preprocess()
    
    def step_maximization(self, X):
        sumR = np.sum(self.R, axis=0)
        self.means = np.dot(self.R.T, X) / sumR[:, np.newaxis]
        self.cov = np.zeros((self.num_comp, X.shape[1], X.shape[1]))
        for k in range(self.num_comp):
            diff = X - self.means[k]
            self.cov[k] = np.dot(self.R[:, k] * diff.T, diff) / sumR[k]
        self.W = sumR / self.num_samp

    def step_expectation(self, X):
        self.R = np.zeros((self.num_samp, self.num_comp))
        for k in range(self.num_comp):
            rv = multivariate_normal(mean=self.means[k], cov=self.cov[k])
            self.R[:, k] = self.W[k] * rv.pdf(X)
        sum_resp = np.sum(self.R, axis=1)[:, np.newaxis]
        self.R = self.R / (sum_resp + 1e-10)
        return self.R

    def fit(self, X=None):
        if X is None:
            X = self.embeds
        self.num_samp, self.num_feat = X.shape
        self.means = X[np.random.choice(self.num_samp, self.num_comp, False)]
        self.cov = np.array([np.cov(X.T)] * self.num_comp)
        self.W = np.ones(self.num_comp) / self.num_comp
        self.LLH = -np.inf
        for _ in tqdm(range(self.max_iter), desc="fit"):
            self.R = self.step_expectation(X)
            self.step_maximization(X)
            llh = self.compute_llh(X)
            if np.abs(llh - self.LLH) < self.tol:
                break
            self.LLH = llh

    def predict(self, X=None):
        if X is None:
            X = self.embeds
        self.num_samp, _ = X.shape
        self.R = self.step_expectation(X)
        self.cluster_assignments = np.argmax(self.R, axis=1)
        return self.cluster_assignments

    def compute_llh(self, X):
        log_likelihood = np.zeros(self.num_samp)
        for k in range(self.num_comp):
            rv = multivariate_normal(mean=self.means[k], cov=self.cov[k])
            log_likelihood += self.W[k] * rv.pdf(X)
        return np.sum(np.log(log_likelihood + 1e-10))

    def getParams(self):
        return {
            'means': self.means,
            'covariances': self.cov,
            'weights': self.W
        }

    def getMembership(self, X=None):
        if X is None:
            X = self.embeds
        self.R = self.step_expectation(X)
        return self.R

    def getLikelihood(self, X=None):
        if X is None:
            X = self.embeds
        llh = self.compute_llh(X)
        return np.exp(llh)

    def AIC(self, X=None):
        if X is None:
            X = self.embeds
        self.num_samp, self.num_feat = X.shape
        num_param = self.num_comp * (1 + self.num_feat + self.num_feat * (self.num_feat + 1) / 2)
        llh = self.compute_llh(X)
        return 2 * num_param - 2 * llh

    def BIC(self, X=None):
        if X is None:
            X = self.embeds
        self.num_samp, self.num_feat = X.shape
        num_param = self.num_comp * (1 + self.num_feat + self.num_feat * (self.num_feat + 1) / 2)
        llh = self.compute_llh(X)
        return np.log(self.num_samp) * num_param - 2 * llh
