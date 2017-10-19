
"""Loading and processing expression datasets.

This module contains functions for retrieving RNA-seq expression data
and processing it into formats suitable for use in machine learning pipelines.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from .utils import choose_bmeg_server

import numpy as np
import pandas as pd

import os
import glob

import tarfile
from io import BytesIO

import json
from ophion import Ophion


def log_norm_expr(expr):
    """Log-normalizes expression data.

    Puts a matrix of RNA-seq expression values into log-space after adding
    a constant derived from the smallest non-zero value.

    Args:
        expr (array of float), shape = [n_samples, n_features]

    Returns:
        norm_expr (array of float), shape = [n_samples, n_features]

    Examples:
        >>> norm_expr = log_norm_expr(np.array([[1.0, 0], [2.0, 8.0]]))
        >>> print(norm_expr)
                [[ 0.5849625 , -1.],
                 [ 1.32192809,  3.08746284]]

    """
    log_add = np.nanmin(expr[expr > 0]) * 0.5
    norm_expr = np.log2(expr + log_add)

    return norm_expr


def get_expr_bmeg(cohort):
    """Loads RNA-seq gene-level expression data from BMEG.

    Args:
        cohort (str): The name of an individualCohort vertex in BMEG.

    Returns:
        expr_data (pandas DataFrame of float), shape = [n_samps, n_feats]

    Examples:
        >>> expr_data = get_expr_bmeg('TCGA-BRCA')
        >>> expr_data = get_expr_bmeg('TCGA-PCPG')

    """
    oph = Ophion(choose_bmeg_server(verbose=True))
    expr_list = {}

    # builds a query that will traverse the nodes and edges in BMEG to find
    # the samples in the given cohort with expression data, gets the number
    # of samples that this query returns
    expr_query = ('oph.query().has("gid", "cohort:" + cohort)'
                  '.outgoing("hasSample").mark("sample")'
                  '.incoming("expressionFor").mark("expression")'
                  '.select(["sample", "expression"])')
    samp_count = eval(expr_query).count().execute()[0]

    # ensures BMEG is running
    if not samp_count.isnumeric():
        raise IOError("BMEG could not process query, returned error:\n"
                      + samp_count)

    # ensures the query returns data
    if samp_count == '0':
        raise ValueError("No samples found in BMEG for cohort "
                         + cohort + " !")

    # parses expression data and loads it into a list
    for qr in eval(expr_query).execute():
        dt = json.loads(qr)
        if ('expression' in dt and 'properties' in dt['expression']
                and 'expressions' in dt['expression']['properties']):
            expr_list[dt['sample']['gid']] = json.loads(
                dt['expression']['properties']['expressions'])

    # creates a sample x expression matrix and parses the sample names
    expr_data = pd.DataFrame(expr_list).transpose()
    expr_data.index = [x[-1] for x in expr_data.index.str.split(':')]

    return expr_data


def get_expr_firehose(cohort, data_dir):
    """Loads RNA-seq gene-level expression data downloaded from Firehose.

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
        >>> expr_data = get_expr_bmeg(
        >>>     'BRCA', '/home/users/grzadkow/compbio/input-data/firehose')
        >>> expr_data = get_expr_bmeg('SKCM', '../firehose')

    """
    expr_tar = tarfile.open(glob.glob(os.path.join(
        data_dir, "stddata__2016_01_28", cohort, "20160128",
        "*Merge_rnaseqv2_*_RSEM_genes_normalized_*.Level_3*.tar.gz"
        ))[0])

    # finds the file in the tarball that contains the expression data, loads
    # it into a formatted dataframe
    expr_fl = expr_tar.extractfile(expr_tar.getmembers()[0])
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

