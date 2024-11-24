from models.HMM.hmm import MFCC_Data, HMM
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from hmmlearn import hmm


if __name__ == "__main__":
    data_path = "./data/external/FSDD/recordings" 
    mfcc_data = MFCC_Data(data_path)
    audio_train, audio_test = mfcc_data.prepare_data(n_mfcc=40)

    hmm_model = HMM(n_components=5)
    hmm_model.train_models(audio_train)
    hmm_model.test_accuracy(audio_test)
