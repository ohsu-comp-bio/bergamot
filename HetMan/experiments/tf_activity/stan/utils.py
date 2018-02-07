# Near-duplicate of dream_train's utils.py for now. Change this docstring when they diverge.
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.predict.pipelines import TransferPipe
import HetMan.predict.regressors as regr


# machine learning pipelines for predicting proteomic levels from -omics data
class ElasticNet(regr.ElasticNet, TransferPipe):
    pass

class SVRrbf(regr.SVRrbf, TransferPipe):
    pass

class rForest(regr.rForest, TransferPipe):
    pass

class kNeigh(regr.KNeighbors, TransferPipe):
    pass

class GaussProc(regr.GaussianProcess, TransferPipe):
    pass

class GradBoost(regr.GradientBoosting, TransferPipe):
    pass
