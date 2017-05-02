
import os
import ml_schema.scikit
import unittest
from sklearn.ensemble import RandomForestClassifier
from google.protobuf.json_format import MessageToJson
from sklearn import linear_model

WORK_DIR = os.path.join( os.path.dirname(os.path.dirname(__file__)), "test-work" )

if not os.path.exists(WORK_DIR):
    os.mkdir(WORK_DIR)

class TestSciKitLinear(unittest.TestCase):

    def test_linear(self):
        clf = linear_model.LinearRegression()
        clf.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
        print clf.coef_
        model = ml_schema.scikit.proto_linear(clf, ["val1", "val2"])
        model_file = os.path.join(WORK_DIR, "model.json")
        with open(model_file, "w") as handle:
            handle.write(MessageToJson(model))
        
    def test_ridge(self):
        clf = linear_model.Ridge (alpha = .5)
        clf.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) 
        print clf.coef_
        print clf.intercept_ 
