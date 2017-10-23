"""Loading data for predicting TF Activity via regulon expression.

Author: Hannah Manning <manningh@ohsu.edu>

"""

import pandas as pd
import os

def load_regul_from_file(cohort, id_style="entrez"):
    """Loads (from file) an ARACNE-generated regulon derived from
    the specified TCGA cohort.

    Args:
        cohort (str): TCGA cohort
        id_style (str): Gene ID format ('entrez' or 'ensembl')

    Examples:
        >>> reg_obj = load_regul_from_file("TCGA-BRCA", "entrez")
        >>> reg_obj = load_regul_from_file("TCGA-LAAD", "ensembl")

    """

    fdir = os.path.dirname(__file__)
    reg_file = os.path.join(fdir, "/../data/tf_activity/tmp-" + id_style + "-" + cohort + "-reg.adj")

    # todo: make sure this exits appropriately
    if not os.path.isfile(reg_file):
        except IOError as e:
            print("Regulon file " + reg_file + "does not exist.")

    # todo: this probably doesn't ever close the file...
    reg_obj = pd.read_csv(reg_file, sep='\t')

    return reg_obj
