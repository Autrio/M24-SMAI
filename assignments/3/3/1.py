import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic
import seaborn as sns
import matplotlib.pyplot as plt

from models.MLP.MLP2 import *

if __name__=="__main__":
    path = "/home/autrio/college-linx/SMAI/smai-m24-assignments-Autrio/data/external/HousingData.csv"
    data = Data()
    data.load_raw_data(path)
    data.Preprocess(drop=None,nan=['CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT'])
    data.desc_data()
    data.plot_data(task="regression")