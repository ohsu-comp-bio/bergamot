"""Saves list of genes to file.

This file will be used to split fitting into parallel tasks in fit.py

Authors: Hannah Manning
         Michal Grzadkowski
"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../../')])

from HetMan.features.expression import get_expr_firehose
from HetMan.features.proteomics import get_rppa_firehose
from HetMan.features.cohorts import TFActivityCohort

import dill as pickle

firehose_dir = '/home/exacloud/lustre1/CompBio/manningh/input-data/firehose'


def main(argv):
    """"""

    cohort = argv[0]
    clf_type = argv[1]

    # get the directory where results will be saved and the name of the
    # TCGA cohort we will be using
    print(argv)
    out_path = os.path.join(base_dir, 'output', cohort, clf_type)

    # load the expression data and rppa data for a given cohort
    expr_data = get_expr_firehose(cohort, firehose_dir)
    rppa_data = get_rppa_firehose(cohort, firehose_dir)
    print("expr_data shape: {}".format(expr_data.shape))
    print("rppa_data shape: {}".format(rppa_data.shape))

    cdata = TFActivityCohort(cohort=cohort, cv_seed=((cv_id * 41) + 1),
                             cv_prop=0.8)

    # save the list of features (genes) from rppa data
    # note that at this point it should match those in train_expr
    pickle.dump(list(sorted(cdata.train_prot.columns)),
                open(os.path.join(out_path, 'tmp/rppa_gene_list.p'), 'wb'))

if __name__ == "__main__":
    main(sys.argv[1:])
