
"""Setting up the TFActivity baseline testing experiment.

Args:

Examples:

"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../../..')])

from HetMan.features.cohorts import TFActivityCohort

import dill as pickle


def main(argv):
    """Runs the experiment."""

    # get the directory where results will be saved and the name of the
    # TCGA cohort we will be using
    print(argv)
    out_path = os.path.join(base_dir, 'output', argv[0], argv[1], argv[2])

    # load the cohort data (rnaseq, rppa, regulatory networks)
    # for the given cohort
    cdata = TFActivityCohort(cohort=cohort, cv_seed=((cv_id * 41) + 1),
                             cv_prop=0.8)

    # todo: when does this get used? also, shouldn't there be a train_genes?
    # save the list of features (genes) from rppa data
    # note that at this point it should match those in train_expr
    pickle.dump(list(sorted(cdata.train_prot.columns)),
                open(os.path.join(out_path, 'tmp/rppa_gene_list.p'), 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

