"""
Functions to query pre-processed TCGA-PNNL data
"""

import numpy as np
import pandas as pd
import os
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def rename_columns(df, prot=True):
	"""Rename columns to be joined on
	Args:
		DataFrame (pandas DataFrame): Pandas DataFrame with omics data
		prot (bool): Boolean to set slice index for header

	Returns:
		DataFrame (pandas DataFrame of type float) with renamed column headers, shape = [n_feats, n_samps]
	"""
	if prot:
		targ_cols = ['-'.join(x.split('-')[1:4]) for x in df.columns]
		df.columns = targ_cols
	else:
		targ_cols = ['-'.join(x.split('-')[0:3]) for x in df.columns]
		df.columns = targ_cols
	return df


def return_prot(prot, pccs=None):
	"""Select target from proteomics pandas Dataframe.

	Args:
		prot (pandas DataFrame): Pandas DataFrame with Omics data
		pccs (str): String to subset samples on based on protein characterization center i.e 'PNNL' or 'JHU'

	Returns:
		DataFrame (pandas DataFrame of type float), shape = [n_feats, n_samps]
	"""
	if pccs:
		target_prot = prot.loc[:, prot.columns.str.contains(pccs)]
		target_prot = rename_columns(target_prot).transpose()
	else:
		target_prot = rename_columns(prot).transpose()
	return target_prot


def return_copynumber(copynumber):
	"""Select target from copynumber pandas Dataframe.
	Args:
		copynumber (pandas DataFrame): Pandas DataFrame with Omics data

	Returns:
		DataFrame (pandas DataFrame of type float), shape = [n_feats, n_samps]
	"""
	target_copynumber = rename_columns(copynumber, False).transpose()
	target_copynumber.columns = [gn.split('|')[0] if isinstance(gn, str) else gn for gn in target_copynumber.columns]
	target_copynumber = target_copynumber.transpose().iloc[:, 2:].transpose()
	target_copynumber = target_copynumber.loc[:,(target_copynumber != 0).any(axis=0)]
	return target_copynumber


def return_expression(expression):
	"""Select target from expression pandas Dataframe.
	Args:
		expression (pandas DataFrame): Pandas DataFrame with Omics data

	Returns:
		DataFrame (pandas DataFrame of type float): shape = [n_feats, n_samps]
	"""
	index = expression.index.str.split('|')
	idx = pd.DataFrame(np.array(list(index)))
	idx.columns = ['GENESYMB', 'ENTREZID']
	expression.index = idx['GENESYMB']
	target_expression = rename_columns(expression, False).transpose()
	target_expression = target_expression.loc[:, ~target_expression.columns.duplicated()]
	target_expression = target_expression.iloc[:, target_expression.columns != '?']

	return target_expression


def uniq_c(mat_df, column=None):
	"""Selects and returns unique columns to resolve issues with pnnl OV data"""
	if column:
		_, i = np.unique(mat_df.T.columns, return_index=True)
		mat_df = mat_df.T.iloc[:,i].T
	else:
		_, i = np.unique(mat_df.columns, return_index=True)
		mat_df = mat_df.iloc[:,i]

	return mat_df 


def get_pnnl(data_dir, pccs=None):
	"""Loads PNNL-phosphoproteome, proteome, copynumber, and expression data into pandas DataFrame.
	Args:
		data_dir (str): The local directory where the pnnl data was downloaded.
		pccs (str): String to subset TCGA-OV by PNNL samples or JHU

	Returns:
		pnnl_df (pandas DataFrame of type float), shape = [n_feats, n_samps]
	Examples:
		>>> pnnl_df = get_pnnl('/home/users/estabroj/scratch/data/OV')
	"""
	# Read in data

	prot = pd.read_csv(os.path.join(data_dir, 'PNNL-proteome.txt'), sep='\t', index_col=1).drop_duplicates()
	copynumber = pd.read_csv(os.path.join(data_dir, 'copynumber.txt'), sep='\t', index_col=0).drop_duplicates()
	expression = pd.read_csv(os.path.join(data_dir, 'expression.txt'), sep='\t', skiprows=[1], index_col=0).drop_duplicates()

	# Filter indices without identifier i.e. NaN

	prot = prot[~pd.isnull(prot.index)]
	copynumber = copynumber[~pd.isnull(copynumber.index)]
	expression = expression[~pd.isnull(expression.index)]

	# Select target across DataTypes

	target_prot = return_prot(prot, pccs)
	target_copynumber = return_copynumber(copynumber)
	target_expression = return_expression(expression)

	use_samples = list(set(target_prot.index) & set(target_copynumber.index) & set(target_expression.index))
	use_genes = list(set(target_prot.columns) & set(target_copynumber.columns) & set(target_expression.columns))
	
	prot = target_prot.loc[use_samples,use_genes]
	copy = target_copynumber.loc[use_samples,use_genes]
	rna = target_expression.loc[use_samples,use_genes]
	
	prot = uniq_c(prot,True)
	copy = uniq_c(copy,True)
	rna = uniq_c(rna,True)

	prot = uniq_c(prot)
	copy = uniq_c(copy)
	rna = uniq_c(rna)
	

	return prot, copy, rna

