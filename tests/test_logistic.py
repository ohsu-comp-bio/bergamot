import pandas

import unittest

from ..src.logistic import LogisticRegression
from .testdata_logistic import expected_prediction, iris_data

# Currently comparing the results to sklearn logistic regression model.
class LogRegTests(unittest.TestCase):
    def setUp(self):
        self.iris_x = pandas.DataFrame(iris_data)
        
        self.iris_coef = [-0.40731745, -1.46092371,  2.24004724,  1.00841492]
        self.iris_intercept = -0.26048137
        self.lr = LogisticRegression(self.iris_coef, self.iris_intercept)
                
    def tearDown(self):
        self.iris = None
        self.iris_x = None
        self.iris_y = None
        
        self.iris_coef = None
        self.iris_intercept = None
        self.lr = None
        
    def test_predict(self):
        self.assertEqual(self.lr.predict(self.iris_x), expected_prediction)

        
if __name__ == '__main__':
    unittest.main()

