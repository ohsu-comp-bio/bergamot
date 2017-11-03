import os
import pandas as pd
import numpy as np
import scipy.stats as st
import random


def aracne_regulon(regul_f):
    """Returns a pandas DataFrame for likelihood and mode of regulation inferred by Aracne-AP, df = [n_feats, n_samps]
    Args:
        regul_f (str): Aracne-AP Regulon file in four column format i.e. Regulator,Target,MoA,likelihood

    Returns:
        lh, mode (pandas DataFrame): sparse DataFrame indicating likelihood and mode or regulation for transcription factors
    """
    regul = pd.read_csv(regul_f, sep='\t')
    targets = regul.Target.unique()
    regulators = regul.Regulator.unique()

    lh = pd.DataFrame(columns=regulators, index=targets)
    mode = pd.DataFrame(columns=regulators, index=targets)

    for r in regulators:
        r_targs = regul.query('Regulator == "{}"'.format(r)).Target.values
        r_lh = regul.query('Regulator == "{}"'.format(r)).likelihood.values
        r_MoA = regul.query('Regulator == "{}"'.format(r)).MoA.values

        lh.loc[r_targs,'{}'.format(r)] = r_lh
        mode.loc[r_targs, '{}'.format(r)] = r_MoA

    return lh, mode


def check_instance(obj, norm=False):
    """Checks instance of obj passed and returns a pandas DataFrame
    Args:
        obj (str | pd.DataFrame): file name or pandas DataFrame
        norm (bool): bool to return Z-score normed data
    Returns:
        obj_norm (pd.DataFrame): pandas DataFrame of shape = [n_feats, n_samps]
    """
    if norm:
        if isinstance(obj, str):
            pd_obj = pd.read_csv(obj, sep='\t', index_col=0)
            obj_norm = pd_obj.sub(pd_obj.mean(axis=1), axis=0).divide(pd_obj.std(axis=1),axis=0)
        elif isinstance(obj, pd.DataFrame):
            obj_norm = obj.sub(obj.mean(axis=1), axis=0).divide(obj.std(axis=1),axis=0)
    else:
        if isinstance(obj, str):
            obj_norm = pd.read_csv(obj, sep='\t', index_col=0)
        elif isinstance(obj, pd.DataFrame):
            obj_norm = obj

    return obj_norm


def generate_complete_nes(lh, mode, expr):
    """Tests for global shift in positions of each regulon genes when projected on the rank-sorted gene expression
    signature.
    Args:
        lh (str | pd.DataFrame): file name or pandas DataFrame that contains likelihood values calculated by Aracne-AP
            of shape [n_feats, n_tfs]
        mode (str | pd.DataFrame): file name or pandas DataFrame that contains mode or regulation values calculated
            by Aracne-AP, of shape [n_feats, n_tfs]
        expr (str | pd.DataFrame): file name or pandas DataFrame that contains mode or expression values
            of shape [n_feats, n_samps]

    Returns:
        nes (pd.DataFrame): pandas DataFrame of normalized enrichment scores of shape [n_tfs, n_samps]
        es (pd.DataFrame): pandas DataFrame of enrichment scores of shape [n_tfs, n_samps]
    """
    mode = check_instance(mode, False)
    lh = check_instance(lh, False)

    # Reorder expression file to be consistent with likelihood and Mode of Regulation
    expression = check_instance(expr, True).loc[mode.index,:]

    mor = mode.fillna(0.0)

    # Generate normalized likelihood weights
    wts = pd.DataFrame(lh / lh.max(),columns=lh.columns,index=lh.index)
    wts = wts.fillna(0.0)

    # Absolute gene expression signature
    nes = pd.DataFrame((wts **2).sum()**.5)
    wts = wts / wts.sum()

    # Pull matching index positions of likelihood frame for expression
    pos = expression.index.get_indexer_for(lh.index)
    t2 = expression.rank() / (expression.shape[0] + 1)

    # (one-tail) absolute value of the gene expression signature i.e. genes are rank-sorted from the less invariant
    # between groups to the most DE
    t1 = abs(t2 -.5)*2
    t1 = t1 + (1 - t1.max())/2

    # Quantile transformed rank positions as enrichment score
    t1 = pd.DataFrame(st.norm.ppf(t1.iloc[pos,], loc=0, scale=1), columns=t1.columns, index=t1.index)
    t2 = pd.DataFrame(st.norm.ppf(t2.iloc[pos,], loc=0, scale=1), columns=t2.columns, index=t2.index)

    # Mode of Requlation by Weights, transpose, perform matrix mult on t1
    sum1 = (mor * wts).T.dot(t2)

    # (two-tail) positions of genes whose expression signatures are repressed by a give TF are inverted prior to
    # enrichment score estimates
    sum2 = ((1 - abs(mor)) * wts).T.dot(t1)
    ss = np.sign(sum1)

    # Integrate one and two tail enrichment scores and weight contribution based on mode of regulation and regulator-target
    # gene interaction confidence

    tmp = (abs(sum1) + sum2 * (sum2>0)) * ss

    es = tmp

    #
    nes = pd.DataFrame(tmp.values*nes.values, columns=tmp.columns, index=tmp.index)

    return nes, es


def generate_null(expression, per=100):
    """Generates a null set of interactions for each regulator through randomly and uniformly permuting samples
    Args:
        expression (pd.DataFrame): file name or pandas DataFrame that contains likelihood values calculated by Aracne-AP
            of shape [n_feats, n_tfs]
        per (int): number of permutations to generate null model

    Returns:
        null (pd.DataFrame): Z-score normalized pandas Dataframe of shape [n_feats, per]
    """
    ncol = expression.shape[1]
    sub = int(expression.shape[1]/2)

    null = np.zeros((expression.shape[0],per))

    # Generate permutations of GES
    per1 = np.array([(lambda x: random.sample(range(ncol), sub))(x) for x in range(100)])
    for i in range(len(per1)):
        pos = per1[i]

        #Zscore normalization
        null[:,i] = (expression.iloc[:,pos].mean(axis=1) - expression.iloc[:,-pos].mean(axis=1))/(expression.iloc[:,pos].var(axis=1) + expression.iloc[:,-pos].var(axis=1))
    null = pd.DataFrame(null,index=expression.index)

    return null
