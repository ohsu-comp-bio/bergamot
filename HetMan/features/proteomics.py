
"""Loading and processing proteomic datasets.

This module contains functions for retrieving proteomics data
and processing it into formats suitable for use in machine learning pipelines.

Authors: Michal Grzadkowski <grzadkow@ohsu.edu>
         Joey Estabrook <estabroj@ohsu.edu>

"""

import numpy as np
import pandas as pd

import os
import glob

import tarfile
from io import BytesIO


def get_prot_cptac(syn, cohort, source=None):
    """Get the CPTAC proteomic data used in the DREAM challenge.

    Args:
        syn (synapseclient.Synapse): A logged-into Synapse instance.
        cohort (str): A TCGA cohort included in the challenge.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>>
        >>> syn.cache.cache_root_dir = (
        >>>     '/home/exacloud/lustre1/'
        >>>     'share_your_data_here/precepts/synapse'
        >>>     )
        >>> syn.login()
        >>>
        >>> get_prot_cptac(syn, "BRCA")
        >>> get_prot_cptac(syn, "OV", source="PNNL")

    """

    syn_ids = {'BRCA': '11328678',
               'OV': {'JHU': '10514980', 'PNNL': '10514979'}}

    if isinstance(syn_ids[cohort], dict):
        use_id = syn_ids[cohort][source]
    else:
        use_id = syn_ids[cohort]

    prot_data = pd.read_csv(syn.get("syn{}".format(use_id)).path,
                            sep='\t', index_col=0).transpose()

    return prot_data


def member_idx(tar_members, cnv=False):
	"""Retrieve list index location for normalized tar firehose data.
	Args:
		tar_members (tarfile.TarFile): tarfile to be read in as pandas DataFrame
	Returns:
		i (int): Index integer of tar_member file to be read in
	"""
	for i in range(len(tar_members.getmembers())):
		if 'data.txt' in tar_members.getmembers()[i].get_info()['name']:
			return i


def get_rppa_firehose(cohort, data_dir):
	"""Loads RPPA data downloaded from Firehose.
	Firehose data is acquired by downloading the firehose_get from
		https://confluence.broadinstitute.org/display/GDAC/Download
	into `data_dir` and then running
		./firehose_get -o RSEM_genes_normalized data 2016_01_28 brca
	from the command line, assuming firehose_get v0.4.11
	Args:
		cohort (str): The name of a TCGA cohort available in Broad Firehose.
		data_dir (str): The local directory where the Firehose data was
						downloaded.
	Returns:
		expr_data (pandas DataFrame of float), shape = [n_samps, n_feats]
	Examples:
		>>> rppa_data = get_rppa_firehose(
		>>>		'BRCA', '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose')
		>>> rppa_data = get_rppa_firehose('SKCM', '../firehose')
	"""
	expr_tar = tarfile.open(glob.glob(os.path.join(data_dir, "stddata__2016_07_15", cohort, "20160715","*Merge_protein_exp_*protein_normalization*.Level_3*.tar.gz"))[0])

	# finds the file in the tarball that contains the expression data, loads
	# it into a formatted dataframe

	expr_fl = expr_tar.extractfile(expr_tar.getmembers()[member_idx(expr_tar, False)])

	expr_data = pd.read_csv(BytesIO(expr_fl.read()),
							sep='\t', skiprows=[1], index_col=0,
							engine='python').transpose()

	# parses the expression matrix columns to get the gene names, removes the
	# columns that don't correspond to known genes
	expr_data.columns = [gn.split('|')[0] if isinstance(gn, str) else gn
						 for gn in expr_data.columns]
	expr_data = expr_data.iloc[:, expr_data.columns != '?']

	# parses expression matrix rows to get TCGA sample barcodes
	expr_data.index = ["-".join(x[:4])
					   for x in expr_data.index.str.split('-')]
	return expr_data

