
from models.HMM.hmm import MFCC_Data, HMM
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from hmmlearn import hmm
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    data_path = "./data/interim/5/3/self/self" 
    mfcc_data = MFCC_Data(data_path)
    audio_train, audio_test = mfcc_data.prepare_data(n_mfcc=40)

    hmm_model = HMM(n_components=1)
    hmm_model.train_models(audio_train)
    hmm_model.test_accuracy(audio_test)
