from scipy.signal import argrelextrema
import numpy as np

def get_pivots(data, order=5):
    return (
        argrelextrema(data.values, np.greater, order=order)[0],
        argrelextrema(data.values, np.less, order=order)[0],
    )

