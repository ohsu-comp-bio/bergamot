
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])
sys.path.extend(['/home/exacloud/lustre1/CompBio/estabroj/bergamot/ophion/client/python/'])

import synapseclient
import dill as pickle
from glob import glob

from HetMan.features.cohorts import TFACohort
import HetMan.predict.bayesian_pathway.multi_protein as mpt


def load_output(out_dir):

    out_files = [
        glob(os.path.join(out_dir, 'results/out_*_{}.p'.format(cv_id)))
        for cv_id in range(10)
        ]
    
    return [{fl.split('results/')[-1].split('_')[1]:
             pickle.load(open(fl, 'rb')) for fl in fls}
            for fls in out_files]

def main(argv):


    # loads the challenge data
    cdata = TFACohort(argv[0], regulator=argv[3], intx_types=[argv[1]],
                                cv_seed=(int(argv[2]) * 41) + 1, cv_prop=0.8)

    # initializes the model and fits it using all of the genes in the
    # `inter`section of the RNA genes, CNA genes, and proteome genes
    clf = mpt.StanTFADefault(argv[1])

    # finds the best combination of model hyper-parameters, uses these
    # parameters to fit to the data
    #clf.tune_coh(cdata, pheno='inter',
    #             tune_splits=4, test_count=4, parallel_jobs=16)
    clf.fit_coh(cdata, pheno='inter') 
    out_dir = '/home/users/estabroj/experiments/predict_TFA/stan'
    out_file = os.path.join(out_dir, 'output', 'intx', argv[0], 'results',
                            'out_{}_{}_{}.p'.format(argv[1], argv[2], argv[3]))
    print(out_file)
    # saves the classifier performance, and the fitted posterior means of the
    # model variables and their names to file
    pickle.dump({'Eval': clf.eval_coh(cdata, pheno='inter'),
                 'PostMeans': clf.named_steps['fit'].post_means,
                 'VarNames': clf.named_steps['fit'].var_names},
                open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

