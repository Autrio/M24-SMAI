import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from hmmlearn import hmm
import pandas as pd
from glob import glob
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import random
import seaborn as sns

class MFCC_Data:
    def __init__(self, path):
        self.dir = path

    def extract_mfcc(self, audio_path, n_mfcc=13, n_fft=512):
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
        return mfcc.T  

    def prepare_data(self, n_mfcc=13, train_ratio=0.8):
        audio_train = {digit: [] for digit in range(10)}
        audio_test = {digit: [] for digit in range(10)}
        
        with tqdm(range(10), desc="Extracting MFCCs", total=10) as pbar:
            for digit in pbar:
                paths = glob(os.path.join(self.dir, f"{digit}_*.wav"))
                random.shuffle(paths)
                
                split_index = int(len(paths) * train_ratio)
                train_paths, test_paths = paths[:split_index], paths[split_index:]
                
                for file_path in train_paths:
                    mfcc = self.extract_mfcc(file_path, n_mfcc=n_mfcc)
                    audio_train[digit].append(mfcc)
                    
                for file_path in test_paths:
                    mfcc = self.extract_mfcc(file_path, n_mfcc=n_mfcc)
                    audio_test[digit].append(mfcc)
                pbar.set_description(f"Processed digit {digit}: {len(train_paths)} train, {len(test_paths)} test")

        return audio_train, audio_test
    
    def visualize_mfcc(self, mfcc, plot="spectrogram"):
        if plot == "spectrogram":
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfcc, x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('MFCC')
            plt.tight_layout()
            plt.show()
        elif plot == "heatmap":
            plt.figure(figsize=(10, 4))
            sns.heatmap(mfcc, cmap="coolwarm")
            plt.xlabel("Time")
            plt.ylabel("MFCC Coefficients")
            plt.show()


class HMM:
    def __init__(self, n_components=5, covariance_type='diag', n_iter=100):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.models = {}

    def train_models(self, audio_train):
        for digit, features in tqdm(audio_train.items(),desc="Training HMMs",position=0,leave=True):
            normalized_features = [StandardScaler().fit_transform(f) for f in features]
            X = np.vstack(normalized_features)  
            lengths = [f.shape[0] for f in normalized_features]  
            
            model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.covariance_type, n_iter=self.n_iter)
            model.fit(X, lengths)
            self.models[digit] = model
            tqdm.write(f"Trained HMM for digit {digit}",end="\n")

    def test_accuracy(self, audio_test):
        correct = 0
        total = 0
        for digit, features in tqdm(audio_test.items(),desc="Testing accuracy"):
            normalized_features = [StandardScaler().fit_transform(f) for f in features]
            for mfcc_features in normalized_features:
                log_likelihoods = {d: model.score(mfcc_features) for d, model in self.models.items()}
                predicted_digit = max(log_likelihoods, key=log_likelihoods.get)
                if predicted_digit == digit:
                    correct += 1
                total += 1
        accuracy = correct / total * 100
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy


