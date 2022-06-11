import pandas as pd 
import numpy as np


def read_data(path):
    WS = pd.read_excel(path)
    return np.array(WS)[:, 1:]
  