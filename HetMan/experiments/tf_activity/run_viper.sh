#!/bin/sh

# Calls a python script that queries BMEG for expression data and
# writes expression data to file.
# Calls an R script that uses this expression data and a regulon file
# to run VIPER.

# Usage: source ./run_viper.sh [-c bmeg_cohort] [-a adj_file] [-p pheno_file]
# Example: source ./run_viper.sh -c TCGA-BRCA -a adj_file

OPTIND=1

while getopts "c:a:p:" opt; do
    case "$opt" in
    c)  cohort=$OPTARG;;
    esac
done

shift $((OPTIND-1))
echo "Preparing to run VIPER on $cohort"

exprfile=tmp-$cohort-expression.tsv
pfile=tmp-$cohort-pData.tsv

echo "$cohort expression and sample_type (i.e. Tumor v. Normal) will be "
echo "temporarily saved in ../../data/tf_activity/ as:"
echo "$exprfile and $pfile"

# get and write out the expression file
# capture the names of that file as expr
echo "Running prep_for_viper.py -c $cohort"
python prep_for_viper.py -c $cohort

echo "Loading and saving ARACNe regulatory network derived from $cohort context"
Rscript write_aracne_regs.R $cohort

echo "Mapping regulatory network's entrez IDs to ensembl IDs"
python joeys_translator.py -c $cohort

# Run VIPER using the expression data and regulon relationships
echo "Running run_viper.R $exprfile $pfile $cohort"
Rscript run_viper.R $exprfile $pfile $cohort

echo "Mission accomplished"