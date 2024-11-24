from models.HMM.hmm import MFCC_Data, HMM
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from hmmlearn import hmm
from icecream import ic
import seaborn as sns
import os

if __name__ == "__main__":
    audio_files = np.random.choice(os.listdir(f"/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/FSDD/recordings"), size=25, replace=False)
    fig, axs = plt.subplots(5, 5, figsize=(15, 10))
    fig.suptitle('MFCCs of Audio Files')

    for i, audio_file in enumerate(audio_files):
        audio_file_path = f"/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/FSDD/recordings/{audio_file}"
        y, sr = librosa.load(audio_file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        ax = axs[i // 5, i % 5]
        sns.heatmap(mfccs, cmap='coolwarm', xticklabels=10, yticklabels=10, ax=ax)
        ax.set_title(f'MFCCs of {audio_file}')
        ax.set_xlabel('Time')
        ax.set_ylabel('MFCC Coefficients')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/assignments/5/figures/mfcc.png")
