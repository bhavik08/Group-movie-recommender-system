import numpy as np


class Aggregators:
    def __init__(self):
        pass

    @staticmethod
    def average(arr):
        return np.average(arr, axis = 0, weights = None)

    @staticmethod
    def average_bf(arr):
        arr[arr == 0] = np.nan
        return np.nanmean(arr, axis=0)
    
    @staticmethod
    def weighted_average(arr, weights):
        return np.average(arr, axis = 0, weights = weights)
