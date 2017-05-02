

import ml_schema.scikit
import unittest
from sklearn.ensemble import RandomForestClassifier
from google.protobuf.json_format import MessageToJson


class TestRandomForest(unittest.TestCase):

    def test_serialize(self):
        X = [[0, 0], [1, 1]]
        Y = [0, 1]
        clf = RandomForestClassifier(n_estimators=10)
        clf = clf.fit(X, Y)
        
        p = ml_schema.scikit.proto_randomforest(clf)
        print MessageToJson(p)
        
