
import os
import pandas
import ml_schema.scikit
import ml_schema.ml_schema_pb2
import unittest
from sklearn.ensemble import RandomForestClassifier
from google.protobuf.json_format import MessageToJson
from sklearn import linear_model
from sklearn import metrics
WORK_DIR = os.path.join( os.path.dirname(os.path.dirname(__file__)), "test-work" )
DATA_DIR = os.path.join( os.path.dirname(os.path.dirname(__file__)), "examples" )

if not os.path.exists(WORK_DIR):
    os.mkdir(WORK_DIR)

class TestSciKitLinear(unittest.TestCase):

    def test_reader(self):
        model = ml_schema.ml_schema_pb2.Model()
        with open(os.path.join(DATA_DIR, "ElasticNetExample.proto_data"), "rb") as f:
            model.ParseFromString(f.read())
            f.close()
        print MessageToJson(model)

    def test_model(self):
        model = ml_schema.ml_schema_pb2.Model()
        with open(os.path.join(DATA_DIR, "ElasticNetExample.proto_data"), "rb") as f:
            model.ParseFromString(f.read())

        training = pandas.read_csv(os.path.join(DATA_DIR, "TrainingData.txt"), sep="\t", index_col=0)
                
        features = training.loc[:,training.columns[1:]]
        labels = training.loc[:,training.columns[0]]

        sk_model = ml_schema.scikit.from_message(model, features.columns)
        out = sk_model.predict(features)
        fpr, tpr, thresholds = metrics.roc_curve(labels == "sensitive", out)
        print metrics.roc_auc_score(labels == "sensitive", out)