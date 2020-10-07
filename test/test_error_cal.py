import math
import unittest

from error_calc import *


class Test_Error_Cal(unittest.TestCase):
    def test_get_mse(self):
        feature_data = [0, 1, 2]
        ground_truth = [2, 3, 4]

        def function(x, a1, a0):
            return np.multiply(x, a1) / a0

        params = (1, 1/3)
        # mse = (2^2 + 2^2) / 3
        mse = get_mse(feature_data, ground_truth, function, *params)
        self.assertEqual(mse, 8/3)

if __name__ == '__main__':
    unittest.main()
