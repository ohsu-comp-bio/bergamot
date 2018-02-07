"""
Defines classes which fit stan and sklearn models to TCGA expression and rppa data
in order to predict transcription factor activities within a given sample.

Author: Hannah Manning <manningh@ohsu.edu>
        Michal Grzadkowski <grzadkow@ohsu.edu>
"""
from ..pipelines import MultiPipe, TransferPipe, ValuePipe
from .stan_models import *

from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, RegressorMixin

import pystan
import numpy as np
import pandas as pd


class StanTFActivityPredict(BaseEstimator, RegressorMixin):

    def __init__(self, precision=0.01):
        self.precision = precision

        self.use_genes = None
        self.use_path = None

    def fit(self,
            X, y=None,
            n_chains=8, parallel_jobs=24,
            verbose=False, **fit_params):

        # ensures given input data is in the correct format
        if not isinstance(X, dict):
            raise TypeError("X must be a dictionary!")

        # todo: is regulon data supposed to be in X or elsewhere?
        for d in ['rna', 'rppa', 'regulons']:
            if d not in X:
                raise(ValueError("X must have an {} entry!".format(d)))

        regul = X['regulons']
        uniq_in_regul = pd.unique(regul[['Regulator', 'Target']].values.ravel())

        # gets the list of genes we can use in the model
        # problem: this is shadowed in TFActivityCohort
        self.use_genes = sorted(list(
            set(X['rna'].columns) & set(X['rppa'].columns) &
            set(uniq_in_regul)
            ))

        # discards the regulatory relationships we can't use
        regul = regul[regul['Regulator'].isin(self.use_genes)
                      & regul['Target'].isin(self.use_genes)]

        # not sure if needed.
        self.filtered_regul = regul

        if verbose:
            print("\nConsidering {} regulatory relationships between {} genes.\n".format(
                self.filtered_regul.shape[0], len(self.use_genes)))

        x_rna = X['rna'].loc[:, self.use_genes]
        x_rppa = X['rppa'].loc[:, self.use_genes]
        # todo: where does y come from?
        y_use = y.loc[:, self.use_genes]

        # todo: should we construct edges between tfs and themselves?

        # construct tf vector
        # tf name will be converted to its column position in the expr matrix +1
        # stan initializes at 1
        tf = [x_rna.columns.get_loc(r) + 1 for r in regul['Regulator']]
        tg = [x_rna.columns.get_loc(t) + 1 for t in regul['Target']]
        moa = regul['MoA']
        lkly = regul['likelihood']

        sm = pystan.StanModel(model_code=model_code_tf,
                              model_name='TFActivityPredict', verbose=True)

        # todo: how do y and y_use get assigned and fit into here?
        # lists the known data that we will feed into the model
        data_dict = {'N': x_rna.shape[0],
                     'G': x_rna.shape[1],
                     'e': x_rna,
                     'p': x_rppa,
                     'R': len(tf),
                     'tf': tf,
                     'tg': tg,
                     'moa': moa,
                     'lkly': lkly,
                     'UTF': len(np.unique(tf)),
                     'UTG': len(np.unique(tg)),
                     'uniqtf': np.unique(tf),
                     'uniqtg': np.unique(tg)
                     }

        # fits the model given known data, initial values, and priors
        # todo: provide init? or seed?
        self.fit_obj = sm.sampling(
            chains=n_chains, n_jobs=parallel_jobs, iter=75,
            data=data_dict, verbose=True
        )

        if verbose:
            print("Fitting has finished!")

        # get the names of the variables in the model and their posterior
        # means, find the chain with the best log-posterior
        self.var_names = self.fit_obj.flatnames
        self.post_means = self.fit_obj.get_posterior_mean()
        self.best_chain = self.post_means[-1, :].argmax()

    def predict(self, X, verbose=False, **fit_params):

        if self.fit_obj is None:
            raise ValueError("Model has not been fit yet!")

        if verbose:
            print("Creating predictions...")

        # reorders the given RNA and CNA input to match the data
        # on which the model was trained
        # todo: should we just move the objects into this function instead of repeating ourselves?
        x_rna = X['rna'].loc[:, self.use_genes]
        x_rppa = X['rppa'].loc[:, self.use_genes]


        # todo: can i get the fitted tfa_matrix values?
        tf_activities = self.fit_obj.summary(pars='tfa_matrix')['summary'][:, 0]

        # todo: complete this function

        return tf_activities








