
"""Consolidating datasets from The Cancer Genome Atlas for use in learning.

Authors: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from .mut import *
from .mut_freq import *

from ..data.expression import get_expr_firehose, get_expr_bmeg, get_expr_toil
from ..data.variants import get_variants_mc3, get_variants_firehose
from ..data.copies import get_copies_firehose

from ..data.annot import get_gencode
from ..cohorts.utils import log_norm

from functools import reduce
from operator import and_
from itertools import cycle, combinations

import os
from glob import glob
from sklearn.preprocessing import scale


def match_tcga_samples(*samples):
    """Finds the tumour samples common between lists of TCGA barcodes.

    Args:
        samples (:obj:`list` of :obj:`str`)

    Returns:
        samps_match (:obj:`list` of :obj:`tuple`)

    """
    samp_lists = [sorted(set(samps)) for samps in samples]

    # parse the sample barcodes into their constituent parts
    parsed_samps = [[samp.split('-') for samp in samps]
                    for samps in samp_lists]

    # get the names of the individuals associated with each sample
    partics = [['-'.join(prs[:3]) for prs in parsed]
               for parsed in parsed_samps]

    # get the type of each sample (tumour, control, etc.) and the
    # vial it was tested in
    types = [np.array([int(prs[3][:2]) for prs in parsed])
             for parsed in parsed_samps]
    vials = [[prs[3][-1] for prs in parsed] for parsed in parsed_samps]

    # find the individuals with primary tumour samples in both lists
    partic_use = sorted(reduce(
        and_,
        [set(prt for prt, tp in zip(partic, typ) if tp < 10)
         for partic, typ in zip(partics, types)]
        ))

    # find the positions of the samples associated with these shared
    # individuals in the original lists of samples
    partics_indx = [[[i for i, prt in enumerate(partic) if prt == use_prt]
                     for partic in partics]
                    for use_prt in partic_use]

    # match the samples of the individuals with only one sample in each list
    samps_match = [
        (partics[0][indx[0][0]],
         tuple(samp_list[ix[0]] for samp_list, ix in zip(samp_lists, indx)))
         for indx in partics_indx if all(len(ix) == 1 for ix in indx)
        ]

    # for individuals with more than one sample in at least one of the two
    # lists, find the sample in each list closest to the primary tumour type
    if len(partics_indx) > len(samps_match):
        choose_indx = [
            tuple(ix[np.argmin(typ)] for ix, typ in zip(indx, types))
            for indx in partics_indx if any(len(ix) > 1 for ix in indx)
            ]

        samps_match += [
            (partics[0][chs[0]],
             tuple(samp_list[ix] for samp_list, ix in zip(samp_lists, chs)))
            for chs in choose_indx
            ]

    match_dict = [{old_samps[i]: new_samp
                   for new_samp, old_samps in samps_match}
                  for i in range(len(samples))]

    return match_dict


def get_expr_data(cohort, expr_source, **expr_args):
    """Loads RNA-seq expression data for a given TCGA cohort and data source.

    Args:
        cohort (str): A cohort in TCGA with expression data.
        expr_source (str): A repository which contains TCGA datasets.

    """

    if expr_source == 'BMEG':
        expr_mat = get_expr_bmeg(cohort)
        expr = log_norm(expr_mat.fillna(0.0))

    elif expr_source == 'Firehose':
        expr_mat = get_expr_firehose(cohort, expr_args['expr_dir'])
        expr = log_norm(expr_mat.fillna(0.0))

    elif expr_source == 'toil':
        expr = get_expr_toil(cohort, expr_args['expr_dir'],
                             expr_args['collapse_txs'])

    else:
        raise ValueError("Unrecognized source of expression data!")

    return expr


def add_variant_data(cohort, var_source, expr, gene_annot, add_cna=False,
                     **var_args):
    """Adds variant calls to loaded TCGA expression data.

    Args:
        cohort (str): A cohort in TCGA with variant call data.
        var_source (str): A repository which contains TCGA datasets.
        expr (:obj:`pd.DataFrame`, shape = [n_samps, n_genes])
        gene_annot (dict): Annotation for each of the expressed genes.
        add_cna (bool, optional): Whether to add discrete copy number calls.
                                  Default is to only use point mutations.

    """

    if var_source == 'mc3':
        variants = get_variants_mc3(var_args['syn'])
    
    elif var_source == 'Firehose':
        variants = get_variants_firehose(cohort, var_args['var_dir'])
    
    else:
        raise ValueError("Unrecognized source of variant data!")

    if add_cna:
        copy_data = get_copies_firehose(cohort, var_args['copy_dir'],
                                        discrete=True)

        if 'Gene' in var_args['mut_levels']:
            copy_lvl = var_args['mut_levels'][
                var_args['mut_levels'].index('Gene') + 1]

        else:
            copy_lvl = var_args['mut_levels'][0]

        # reshapes the matrix of CNA values into the same long format
        # mutation data is represented in
        copy_lvl = copy_lvl.split('_')[0]
        copy_df = pd.DataFrame(copy_data.stack())
        copy_df = copy_df.reset_index(level=copy_df.index.names)

        copy_df.columns = ['Sample', 'Gene', copy_lvl]
        matched_samps = match_tcga_samples(
            expr.index, variants['Sample'], copy_df['Sample'])

        copy_df = copy_df.loc[
            copy_df.index.isin(matched_samps[2]),
            copy_df.columns.isin(list(gene_annot))
        ]
        copy_df.index = [matched_samps[2][old_samp]
                         for old_samp in copy_df.index]

        # removes CNA values corresponding to an absence of a variant
        copy_df = copy_df.loc[copy_df[copy_lvl] != 0, :]

        # maps CNA integer values to their descriptions, appends
        # CNA data to the mutation data
        copy_df[copy_lvl] = copy_df[copy_lvl].map(
            {-2: 'HomDel', -1: 'HetDel', 1: 'HetGain', 2: 'HomGain'})
        variants = pd.concat([variants, copy_df])

    else:
        matched_samps = match_tcga_samples(expr.index, variants['Sample'])

    return variants, matched_samps


def get_copy_data(cohort, copy_source, **copy_args):
    """Loads discrete CNA calls for a given TCGA cohort and data source.

    Args:
        cohort (str): A cohort in TCGA with expression data.
        copy_source (str): A repository which contains TCGA datasets.

    """

    if copy_source == 'Firehose':
        copy_data = get_copies_firehose(cohort, copy_args['copy_dir'],
                                        discrete=False)

    elif copy_source == 'BMEG':
        pass

    else:
        raise ValueError("Unrecognized source of CNA data!")

    return copy_data


def list_cohorts(data_source, **data_args):
    """Finds all the TCGA cohorts available in a given data repository.

    Args:
        data_source (str): A repository which contains TCGA datasets.

    """

    if data_source == 'BMEG':
        pass

    elif data_source == 'Firehose':
        cohorts = {pth.split('/')[-1]
                   for pth in glob(os.path.join(data_args['data_dir'],
                                                'stddata__*/*'))}

    elif data_source == 'toil':
        pass

    else:
        raise ValueError("Unrecognized source of expression data!")

    return cohorts


class MutationCohort(BaseMutationCohort):
    """An expression dataset used to predict genes' mutations (variants).

    Args:
        cohort (str): The label for a cohort of samples.
        mut_genes (:obj:`list` of :obj:`str`)
            Which genes' variants to include.
        mut_levels (:obj:`list` of :obj:`str`)
            What variant annotation levels to consider.
        expr_source (str): Where to load the expression data from.
        var_source (str): Where to load the variant call data from.
        copy_source (:obj:`str`, optional)
            Where to load continuous copy number alteration data from. The
            default is to not use continuous CNA scores.
        top_genes (:obj:`int`, optional)
            How many of the genes in the cohort ordered by descending
            mutation frequency to load mutation data for.
        cv_prop (float): Proportion of samples to use for cross-validation.
        cv_seed (int): The random seed to use for cross-validation sampling.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>>
        >>> cdata = MutationCohort(
        >>>     cohort='BRCA', mut_genes=['TP53', 'PIK3CA'],
        >>>     mut_levels=['Gene', 'Form', 'Exon'], syn=syn
        >>>     )
        >>>
        >>> cdata2 = MutationCohort(
        >>>     cohort='PAAD', mut_genes=['KRAS'],
        >>>     mut_levels=['Form_base', 'Exon', 'Location'],
        >>>     expr_source='Firehose', expr_dir='../input-data/firehose',
        >>>     cv_seed=98, cv_prop=0.8, syn=syn
        >>>     )
        >>>
        >>> cdata3 = MutationCohort(
        >>>     cohort='LUSC', mut_genes=None, mut_levels=['Gene', 'Form'],
        >>>     expr_source='Firehose', var_source='mc3',
        >>>     copy_source='Firehose', expr_dir='../input-data/firehose',
        >>>     copy_dir='../input-data/firehose', samp_cutoff = 0.2,
        >>>     syn=syn, cv_prop=0.75, cv_seed=5601
        >>>     )
        >>>
    """

    def __init__(self,
                 cohort, mut_genes, mut_levels=('Gene', 'Form'),
                 expr_source='BMEG', var_source='mc3', copy_source=None,
                 top_genes=100, samp_cutoff=None, cv_prop=2.0/3, cv_seed=None,
                 **coh_args):
        self.cohort = cohort

        # use expression source for variants unless otherwise specified
        if var_source is None:
            var_source = expr_source

        # loads TCGA expression data, gets annotation for each gene for which
        # expression was measured
        expr = get_expr_data(cohort, expr_source, **coh_args)
        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in get_gencode().items()
                           if at['gene_name'] in expr.columns}

        variants, matched_samps = add_variant_data(
            cohort, var_source, expr, self.gene_annot, **coh_args)

        # load copy number alteration data from the given source if necessary
        if copy_source is not None:
            copy_data = get_copy_data(cohort, copy_source, **coh_args)

            matched_samps = match_tcga_samples(
                expr.index, variants['Sample'], copy_data.index)

            copy_data = copy_data.loc[
                copy_data.index.isin(matched_samps[2]),
                copy_data.columns.isin(list(self.gene_annot))
                ]
            copy_data.index = [matched_samps[2][old_samp]
                               for old_samp in copy_data.index]

        else:
            copy_data = None

        expr = expr.loc[expr.index.isin(matched_samps[0]),
                        expr.columns.isin(list(self.gene_annot))]
        expr.index = [matched_samps[0][old_samp] for old_samp in expr.index]

        variants = variants.loc[variants['Sample'].isin(matched_samps[1]), :]
        variants['Sample'] = [matched_samps[1][old_samp]
                              for old_samp in variants['Sample']]

        super().__init__(expr, variants, copy_data, mut_genes, mut_levels,
                         top_genes, samp_cutoff, cv_prop, cv_seed)


class MutFreqCohort(BaseMutFreqCohort):
    """An expression dataset used to predict genes' mutations (variants).

    Args:
        cohort (str): The label for a cohort of samples.
        expr_source (str): Where to load the expression data from.
        var_source (str): Where to load the variant data from.
        copy_source (:obj:`str`, optional)
            Where to load the copy number alteration data from. The default
            is to not use any CNA data.
        cv_prop (float): Proportion of samples to use for cross-validation.
        cv_seed (int): The random seed to use for cross-validation sampling.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>>
        >>> cdata = MutationCohort(
        >>>     cohort='TCGA-BRCA', mut_genes=['TP53', 'PIK3CA'],
        >>>     mut_levels=['Gene', 'Form', 'Exon'], syn=syn
        >>>     )

    """

    def __init__(self,
                 cohort, expr_source='BMEG', var_source='mc3',
                 cv_prop=2.0/3, cv_seed=None, **coh_args):
        self.cohort = cohort

        if var_source is None:
            var_source = expr_source

        expr = get_expr_data(cohort, expr_source, **coh_args)
        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in get_gencode().items()
                           if at['gene_name'] in expr.columns}

        variants, matched_samps = add_variant_data(
            cohort, var_source, expr, self.gene_annot, **coh_args)

        expr = expr.loc[expr.index.isin(matched_samps[0]),
                        expr.columns.isin(list(self.gene_annot))]
        expr.index = [matched_samps[0][old_samp] for old_samp in expr.index]

        variants = variants.loc[variants['Sample'].isin(matched_samps[1]), :]
        variants['Sample'] = [matched_samps[1][old_samp]
                              for old_samp in variants['Sample']]

        super().__init__(expr, variants, cv_prop, cv_seed)


class PanCancerMutCohort(BaseMutationCohort):

    def __init__(self, 
                 mut_genes, mut_levels=('Gene', 'Form'),
                 expr_source='BMEG', var_source='mc3', copy_source=None,
                 top_genes=100, samp_cutoff=None, cv_prop=2.0/3, cv_seed=None,
                 **coh_args):
        expr_args = dict()
        if 'expr_dir' in coh_args:
            expr_args['data_dir'] = coh_args['expr_dir']
        expr_cohorts = list_cohorts(expr_source, **expr_args)

        expr_dict = dict()
        for cohort in expr_cohorts:
            try:
                expr_data = get_expr_data(cohort, expr_source, **coh_args)

                expr_dict[cohort] = pd.DataFrame(scale(expr_data),
                                                 index=expr_data.index,
                                                 columns=expr_data.columns)

            except:
                print('no expression found for {}'.format(cohort))

        # removes samples that appear in more than one cohort
        for coh1, coh2 in combinations(expr_dict, 2):
            if len(expr_dict[coh1].index & expr_dict[coh2].index):
                ovlp1 = expr_dict[coh1].index.isin(expr_dict[coh2].index)
                ovlp2 = expr_dict[coh2].index.isin(expr_dict[coh1].index)

                if np.all(ovlp1):
                    expr_dict[coh2] = expr_dict[coh2].loc[~ovlp2, :]
                elif np.all(ovlp2) or np.sum(~ovlp1) >= np.sum(~ovlp2):
                    expr_dict[coh1] = expr_dict[coh1].loc[~ovlp1, :]
                else:
                    expr_dict[coh2] = expr_dict[coh2].loc[~ovlp2, :]

        expr = pd.concat(list(expr_dict.values()))
        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in get_gencode().items()
                           if at['gene_name'] in expr.columns}

        if var_source == 'mc3':
            variants, matched_samps = add_variant_data(
                cohort=None, var_source='mc3', expr=expr,
                gene_annot=self.gene_annot, **coh_args
                )

        else:
            if var_source is None:
                var_source = expr_source
                var_cohorts = expr_cohorts

            else:
                var_args = dict()
                if 'var_dir' in coh_args:
                    var_args['data_dir'] = coh_args['var_dir']
                if 'syn' in coh_args:
                    var_args['syn'] = coh_args['syn']

                var_cohorts = list_cohorts(var_source, **var_args)
                var_cohorts &= expr_dict

            var_dict = dict()
            for cohort in var_cohorts:
                try:
                    var_dict[cohort] = add_variant_data(cohort, expr_source,
                                                        **coh_args)

                    if copy_source is not None:
                        var_dict[cohort] = pd.concat([
                            var_dict[cohort],
                            get_copy_data(cohort, copy_source, **coh_args)
                            ])

                except:
                    print('no variants found for {}'.format(cohort))
            
            variants = pd.concat(list(var_dict.values()))

        expr = expr.loc[expr.index.isin(matched_samps[0]),
                        expr.columns.isin(list(self.gene_annot))]
        expr.index = [matched_samps[0][old_samp] for old_samp in expr.index]

        variants = variants.loc[variants['Sample'].isin(matched_samps[1]), :]
        variants['Sample'] = [matched_samps[1][old_samp]
                              for old_samp in variants['Sample']]

        # for each expression cohort, find the samples that were matched to
        # a sample in the mutation call data
        cohort_samps = {
            cohort: set(matched_samps[0][samp]
                        for samp in expr_df.index & matched_samps[0])
            for cohort, expr_df in expr_dict.items()
            }

        # save a list of matched samples for each cohort as an attribute
        self.cohort_samps = {cohort: samps
                             for cohort, samps in cohort_samps.items()
                             if samps}
        copy_data = None
        super().__init__(expr, variants, copy_data, mut_genes, mut_levels,
                         top_genes, samp_cutoff, cv_prop, cv_seed)


class TransferMutationCohort(BaseTransferMutationCohort):

    def __init__(self,
                 cohorts, mut_genes, mut_levels=('Gene', 'Form'),
                 expr_sources='BMEG', var_sources='mc3', copy_sources=None,
                 top_genes=250, samp_cutoff=None, cv_prop=2.0/3, cv_seed=None,
                 **coh_args):

        if isinstance(expr_sources, str):
            expr_sources = [expr_sources]

        if var_sources is None:
            var_sources = expr_sources

        if isinstance(var_sources, str):
            var_sources = [var_sources]

        if 'mc3' in var_sources:
            mc3_vars = get_variants_mc3(coh_args['syn'])
        else:
            mc3_vars = None

        if isinstance(copy_sources, str):
            copy_sources = [copy_sources]

        expr_dict = dict()
        for coh, expr_source in zip(cohorts, cycle(expr_sources)):
            if expr_source == 'BMEG':
                expr_mat = get_expr_bmeg(coh)
                expr_dict[coh] = log_norm(expr_mat.fillna(0.0))

            elif expr_source == 'Firehose':
                expr_mat = get_expr_firehose(coh, coh_args['expr_dir'])
                expr_dict[coh] = log_norm(expr_mat.fillna(0.0))

            elif expr_source == 'toil':
                expr_dict[coh] = get_expr_toil(coh, coh_args['expr_dir'],
                                               coh_args['collapse_txs'])

            else:
                raise ValueError("Unrecognized source of expression data!")

        var_dict = dict()
        for coh, var_source in zip(cohorts, cycle(var_sources)):

            if var_source == 'mc3':
                var_dict[coh] = mc3_vars.copy()

            elif var_source == 'Firehose':
                var_dict[coh] = get_variants_firehose(
                    coh, coh_args['var_dir'])

            else:
                raise ValueError("Unrecognized source of variant data!")

        matched_samps = {coh: match_tcga_samples(expr_dict[coh].index,
                                                 var_dict[coh]['Sample'])
                         for coh in cohorts}

        if copy_sources is not None:
            for coh, copy_source in zip(cohorts, cycle(copy_sources)):
                if copy_source == 'Firehose':
                    copy_data = get_copies_firehose(coh, coh_args['copy_dir'])

                    # reshapes the matrix of CNA values into the same long
                    # format mutation data is represented in
                    copy_df = pd.DataFrame(copy_data.stack())
                    copy_df = copy_df.reset_index(level=copy_df.index.names)
                    copy_df.columns = ['Sample', 'Gene', 'Form']
                    copy_df = copy_df.loc[copy_df['Form'] != 0, :]

                    # maps CNA integer values to their descriptions, appends
                    # CNA data to the mutation data
                    copy_df['Form'] = copy_df['Form'].map(
                        {-2: 'HomDel', -1: 'HetDel', 1: 'HetGain',
                         2: 'HomGain'})
                    var_dict[coh] = pd.concat([var_dict[coh], copy_df])

        annot_data = get_gencode()
        gene_annot = {coh: {at['gene_name']: {**{'Ens': ens}, **at}
                            for ens, at in annot_data.items()
                            if at['gene_name'] in expr_dict[coh].columns}
                      for coh in cohorts}
        self.cohorts = cohorts

        super().__init__(
            expr_dict, var_dict, matched_samps, gene_annot, mut_genes,
            mut_levels, top_genes, samp_cutoff, cv_prop, cv_seed
            )

