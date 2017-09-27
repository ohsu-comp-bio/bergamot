#!/bin/sh

# Produces VIPER activity results for a given TCGA cohort

# Usage: source ./run_viper.sh [-c bmeg_cohort]
# Example: source ./run_viper.sh -c TCGA-BRCA

source activate HetMan

OPTIND=1

while getopts "c:a:p:" opt; do
    case "$opt" in
    c)  cohort=$OPTARG;;
    esac
done

shift $((OPTIND-1))
echo "Preparing to run VIPER on $cohort"

echo "$cohort expression and sample_type (i.e. Tumor v. Normal) will be "
echo "temporarily saved in ../../data/tf_activity/ as:"
echo "$exprfile and $pfile"

# get and write out the expression file
# capture the names of that file as expr
echo "Running prep_for_viper.py -c $cohort"
python prep_for_viper.py -c $cohort

echo "Loading and saving ARACNe regulatory network derived from $cohort context"
Rscript write_aracne_regs_joey_v.R $cohort

echo "Mapping regulatory network's ENTREZ IDs to ENSEMBL IDs"
Rscript translate_regulon.R $cohort

# Run VIPER using the expression data and regulon relationships
echo "Running run_viper.R $cohort"
Rscript run_viper.R $cohort

echo "Finished executing run_viper.sh"
