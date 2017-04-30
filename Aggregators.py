
import math
import numpy as np

class Aggregators:
    def __init__(self):
        pass
    
    #pass ratings or factors as input
    @staticmethod
    def average(arr):
        return np.average(arr, axis = 0, weights = None)

    @staticmethod
    def average_bf(arr):
        arr[arr == 0] = np.nan
        arr1 = np.nanmean(arr, axis=0)
        return arr1
    
    @staticmethod
    def weighted_average(arr, weights):
        return np.average(arr, axis = 0, weights = weights)
    
    @staticmethod
    def mode(arr):
        pass
        
    @staticmethod
    def median(arr):
        pass
    
    @staticmethod
    def least_misery(self, arr):
        pass
    
    @staticmethod
    def most_pleasure(self, arr):
        pass
    