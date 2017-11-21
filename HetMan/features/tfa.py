"""Loading pseudo TFA values generated from pyper.

Author: Joey Estabrook <estabroj@ohsu.edu>

"""

import pandas as pd

def get_tfa_data():
    """Retrieves a particular -omic dataset used to predict TFA
	
    Args:
	#todo currently hard coded for BRCA NES signatures, add utility to pull TFA-NES values for all TCGA-cohorts
    """
    tfa_mat = pd.read_csv('/home/exacloud/lustre1/BioCoders/ProjectCollaborations/PRECEPTS/III/data/BRCA-all-gene-TFA-NES.txt',sep='\t',index=0).transpose()
    return tfa_mat

