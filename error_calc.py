# TODO: error calculation
import numpy as np

def get_mse(feature_data, gt, function, *params):
    prediction = function(feature_data, *params)
    squared_difference = np.square(np.subtract(gt, prediction))
    return squared_difference.mean()