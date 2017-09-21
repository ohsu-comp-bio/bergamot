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
echo "cohort=$cohort, adjfile=$adjfile"

exprfile=tmp-$cohort-expression.tsv
pfile=tmp-$cohort-pData.tsv

echo "exprfile=$exprfile, pfile=$pfile"

# get and write out the expression file
# capture the names of that file as expr
python prep_for_viper.py -c $cohort

# Run VIPER using the expression data and regulon relationships
Rscript run_viper.R $exprfile $pfile $cohort