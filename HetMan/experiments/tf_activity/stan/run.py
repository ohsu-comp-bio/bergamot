
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

import dill as pickle
from glob import glob
from HetMan.features.cohorts import TFActivityCohort
import HetMan.predict.bayesian_pathway.multi_tf as mtfa


# todo: this is a direct copy of dream_train/stan/run.py's load_output. define only once elsewhere.
def load_output(out_dir):

    out_files = [
        glob(os.path.join(out_dir, 'results/out_*_{}.p'.format(cv_id)))
        for cv_id in range(10)
        ]

    return [{fl.split('results/'[-1].split('_')[1]:
        pickle.load(open(fl, 'rb')) for fl in fls}
        for fls in out_files]


def main(argv):

    cohort = argv[0]
    cv_id = int(arg[1])

    # loads the cohort data
    cdata = TFActivityCohort(cohort=cohort, cv_seed=((cv_id * 41) + 1),
                             cv_prop=0.8)

    # initializes the model and fits it using all of the genes in the intersection of the
    # genes in rnaseq, rppa, and regulatory network file
    clf = mtfa.StanDefault()

    # finds the best combination of model hyper-parameters, uses these
    #  parameters to fit to the data
    clf.fit_coh(cdata, pheno='inter')

    out_file = os.path.join(base_dir, 'output', cohort, 'results', 'out_', cv_id)

    # saves the classifier performance and the fitted posterior means of the
    # model variables and their names to file
    pickle.dump({'Eval': clf.eval_coh(cdata, pheno='inter'),
                 'PostMeans': clf.named_steps['fit'].post_means,
                 'VarNames': clf.named_steps['fit'].var_names},
                open(out_file, 'wb'))

if __name__ == "__main__":
    main(sys.argv[1:])
