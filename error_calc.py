# TODO: error calculation
import numpy as np

def get_mse(feature_data, gt, function, *params):
    prediction = function(feature_data, *params)
    squared_difference = np.power(gt-prediction, 2)
    return squared_difference.sum(0) / len(feature_data)