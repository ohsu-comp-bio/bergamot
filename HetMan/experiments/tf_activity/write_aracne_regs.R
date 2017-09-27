# A script for loading and writing a regulon file in preparation for running VIPER.
# Writes out ENTREZID labeled regulon to HetMan/data/tf_activity/tmp-{cohort}-reg.adj
#
# Run from command line: Rscript run_viper.R --args arg1
# where arg1=bmeg_cohort (i.e. TCGA-BRCA)
#
# a bash script calls this one: 
# /Users/manningh/PycharmProjects/bergamot/experiments/tf_activity/run_viper.sh
#
# Author: Joey Estabrook
# Date: September 22, 2017

suppressMessages(library(aracne.networks))

basedir <- "."
datadir <- paste(basedir,"/../../data/tf_activity/", sep="/")

main = function(){
  setwd(datadir)
  args = commandArgs(trailingOnly = TRUE)
  cohort = (args[1])
  id = tolower(strsplit(cohort,'-')[[1]][2])
  regulon = paste('regulon', id, sep='')
  tmp_regulon = paste('tmp-entrez-TCGA-', toupper(id), '-reg.adj',sep='')
  print(paste("Writing", cohort, "regulatory network with ENTREZ IDs to", tmp_regulon))
  write.regulon(get(regulon),file=tmp_regulon)
}

main()