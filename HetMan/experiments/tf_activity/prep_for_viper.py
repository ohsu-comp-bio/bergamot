"""
Initiates a VIPER run for a specified BMEG cohort.

Author: Hannah Manning <manningh@ohsu.edu>

Example:
    python prep_for_viper.py -c TCGA-BRCA

"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
#sys.path += [base_dir + '/../../../../bergamot']
from HetMan.features.expression import get_expr_bmeg

# the following is for get_sample_type and should be moved with it
from HetMan.features.utils import choose_bmeg_server
import json
from ophion import Ophion

data_dir = base_dir + '/../../data/tf_activity/'


# this should be moved to a script in features/
# takes after get_expr_bmeg()
def get_sample_type(cohort, expr_data):
    """Loads sample_type data for all samples present in an expression matrix
    of interest (expr_data)

    Args:
        cohort (str): The name of an individualCohort vertex in BMEG.
        expr_data (pandas DataFrame of float), shape = [n_samps, n_feats]

    Returns:
        stype_map (pd.Series): index = samps, values = sample_type

    Examples:
        >>> samp_type = get_sample_type('TCGA-BRCA', brca_expr_data)

    """

    oph = Ophion(choose_bmeg_server(verbose=True))

    samp_query = ('oph.query().has("gid", "project:" + cohort)'
                  '.outgoing("hasMember").incoming("biosampleOfIndividual")'
                  '.mark("sample").incoming("expressionForSample")'
                  '.mark("expression").select(["sample"])')

    samp_count = eval(samp_query).count().execute()[0]

    # ensures BMEG is running
    if not samp_count.isnumeric():
        raise IOError("BMEG could not process query, returned error:\n"
                      + samp_count)

    # ensures the query returns data
    if samp_count == '0':
        raise ValueError("No samples found in BMEG for cohort "
                         + cohort + " !")

    # makes an empty pd.Series where keys are tcga sample ids
    # and the values will be filled the sample_type
    stype_map = pd.Series(index=expr_data.index)

    # parses phenotype data and loads it into a list
    # qr = represents one sample's information
    for qr in eval(samp_query).execute():
        dt = json.loads(qr)
        if ('properties' in dt
            and 'sample_type' in dt['properties']
            and 'gid' in dt):
            gid = dt['gid'].split(':')[-1]
            if gid in stype_map.index:
                stype_map[gid] = dt['properties']['sample_type'].strip('["').strip('"]').replace(' ', '_')

    return stype_map


def main():
    # take user-specified names of files
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--cohort",
                        type=str,
                        help="TCGA cohort whose expr will be\n"
                        "accessed by BMEG (i.e. TCGA-BRCA)")
    args = parser.parse_args()
    bmeg_cohort = args.cohort

    # load log-normalized expression data as pd.DataFrame of floats
    print("Getting expression data for " + bmeg_cohort)
    expr = get_expr_bmeg(bmeg_cohort)
    print(bmeg_cohort)
    print("Expression data obtained")

    # load sample type for each sample present in expr
    # (to become phenotype data in Bioconductor's ExpressionSet object in R)
    print("Getting sample type data")
    samp_type = get_sample_type(bmeg_cohort, expr)
    print("Sample type data obtained")

    # transpose expr to take on format expected by VIPER and write out
    # shape = [n_feats, n_samps]
    expr = expr.T
    expr_file = data_dir + 'tmp-' + bmeg_cohort + '-expression.tsv'
    expr.to_csv(expr_file, sep = '\t')

    # write out sample_type data
    samp_type_file = data_dir + 'tmp-' + bmeg_cohort + '-pData.tsv'
    samp_type.to_csv(samp_type_file, sep = '\t')

if __name__ == '__main__':
    main()

