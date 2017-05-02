
import numpy as np
from ml_schema import ml_schema_pb2
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model import LinearRegression

def proto_randomforest(clf):
    forest = ml_schema_pb2.RandomForestData()
    for e in clf.estimators_:
        tree = ml_schema_pb2.DecisionTree()
        for i in xrange(e.tree_.node_count):
            node = ml_schema_pb2.DecisionTree.DecisionNode(
                AboveChild=e.tree_.children_left[i],
                BelowChild=e.tree_.children_right[i],
                SplitVariable=str(e.tree_.feature[i]),
                SplitValue=e.tree_.threshold[i],
                LeafNode=False,
                Label=False
            )
            tree.Nodes.extend([node])
        forest.Forest.extend([tree])
    return forest

def proto_linear(clf, feature_names=None):
    linear = ml_schema_pb2.LinearCoeffData()
    linear.Intercept = clf.intercept_
    for i, coeff in enumerate(clf.coef_):
        name = str(i)
        if feature_names is not None:
            name = feature_names[i]
        coeff = ml_schema_pb2.FeatureCoefficient(
            Feature=name,
            Coeff=coeff
        )
        linear.Coeff.extend([coeff])
    model = ml_schema_pb2.ModelStructure(
        Components=[
            ml_schema_pb2.ModelComponent(Coeff=1,LinearCoeff=linear)
        ]
    )
    return model


class MLModel(LinearModel):
    def __init__(self, msg, feature_names=None):
        self.models = []
        self.coef = []
        for m in msg.Structure.Components:
            s = None
            if m.LinearCoeff:
                s = LinearRegression()
                s.intercept_ = m.LinearCoeff.Intercept
                if feature_names is None:
                    s.coef_ = np.zeros(len(m.LinearCoeff.Coeff))
                else:
                    s.coef_ = np.zeros(len(feature_names))                    
                for i, elem in enumerate(m.LinearCoeff.Coeff):
                    if feature_names is None:
                        s.coef_[i] = elem.Coeff
                    else:
                        l = feature_names.get_loc(elem.Feature)
                        s.coef_[l] = elem.Coeff
            self.models.append(s)
            if m.Coeff:
                self.coef.append(m.Coeff)
            else:
                self.coef.append(1.0)
    
    def fit(self, X, y):
        raise Exception("ReadOnly Model")
    
    def decision_function(self, X):
        out = 0.0
        for i, mod in enumerate(self.models):
            out += self.coef[i] * self.models[0].decision_function(X)
        return out
        
def from_message(msg, feature_names=None):
    return MLModel(msg, feature_names)