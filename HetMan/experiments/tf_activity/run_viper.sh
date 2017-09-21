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

# get and write out the expression file
# capture the names of that file as expr
expr=$(prep_for_viper.py -c $cohort 2>&1)
echo "expr=$expr"