
"""Baseline prediction of TFActivity

Args:
    python fit.py <cohort> <regr_cls> <cv_id> <task_id>

Examples:
    python fit.py BRCA rna ElasticNet 0 0
    python fit.py OV cna SVRrbf 2 11
    python fit.py BRCA rna+cna rForest 4 19
"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../../..')])

from HetMan.features.cohorts import TFActivityCohort
# todo: reminder to check on pipeline situation
from HetMan.predict.pipelines import TransferPipe
import HetMan.predict.regressors as regr

import dill as pickle

# machine learning pipelines for predicting proteomic levels from -omics data
class ElasticNet(regr.ElasticNet, ProteinPipe):
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


def main(argv):
    """Runs the experiment."""

    cohort = argv[0]
    regr_cls = eval(argv[1])
    cv_id = int(argv[2])
    task_id = int(argv[3])

    # creates the directory where the results will be stored, loads the list
    # of genes to predict the proteomic levels of and the learning pipeline
    out_dir = os.path.join(base_dir, 'output', cohort, regr_cls)
    gene_list = pickle.load(
        open(os.path.join(out_dir, 'tmp', 'gene_list.p'), 'rb'))

    # todo: why does something like this happen both here and in run.py in the dream challenge code?
    cdata = TFActivityCohort(cohort=cohort, cv_seed=((cv_id + 3) * 37),
                             cv_prop=0.8)

    # for each gene whose proteomic levels are to be predicted,
    # check if it is to be tested here according to the task ID
    out_rval = {gene: 0 for gene in gene_list}
    for i, gene in enumerate(gene_list):
        if i % 20 == task_id:

            print(gene)
            regr_obj = regr_cls()

            # tunes the regressor hyper-parameters using sub-splits of the
            # training cohort, fits it using the entire training cohort
            regr_obj.tune_coh(cdata, gene,
                              tune_splits=4, test_count=32, parallel_jobs=8)
            regr_obj.fit_coh(cdata, gene)

            #score the performance of the regressor for this gene using the
            # testing cohort, show the tuned hyper-parameters
            print(regr_obj)
            out_rval[gene] = regr_obj.eval_coh(cdata, gene)
            print(out_rval[gene])

        else
            del(out_rval[gene])

    # saves classifier results to file
    out_file = os.path.join(out_dir, 'results',
                            'out__cv-{}_task-{}.p'.format(cv_id, task_id))
    pickle.dump(out_rval, open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])