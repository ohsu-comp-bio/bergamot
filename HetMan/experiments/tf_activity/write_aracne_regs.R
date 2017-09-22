# A script for loading and writing a regulon file in preparation for running VIPER.
#
# Run from command line: Rscript run_viper.R --args arg1
# where arg1=bmeg_cohort (i.e. TCGA-BRCA)
#
# a bash script calls this one: 
# /Users/manningh/PycharmProjects/bergamot/experiments/tf_activity/run_viper.sh
#
# Author: Hannah Manning
# Date: September 21, 2017

basedir <- "/Users/manningh/PycharmProjects/bergamot/HetMan"
datadir <- paste(basedir,"data/tf_activity/", sep="/")

main <- function() {
  
  library("Biobase")
  library(aracne.networks)
  
  # get oriented
  setwd(datadir)
  
  # allow for command line arguments (cohort = 'TCGA-BRCA' or similar)
  args <- commandArgs(trailingOnly = TRUE)
  cohort <- args[1]
  
  # build regulon selecting structure
  aracne_regs <- c("regulonblca", "regulonbrca", "reguloncesc", 
                   "reguloncoad", "regulonesca", "regulongbm",
                   "regulonhnsc", "regulonkirc", "regulonkirp", 
                   "regulonlaml", "regulonlihc", "regulonluad", 
                   "regulonlusc", "regulonov", "regulonpaad",
                   "regulonpcpg", "regulonprad", "regulonread", 
                   "regulonsarc", "regulonstad", "regulontgct", 
                   "regulonthca", "regulonthym", "regulonucec")
  
  names(aracne_regs) <- c("TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC",
                          "TCGA-COAD", "TCGA-ESCA", "TCGA-GBM",
                          "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP",
                          "TCGA-LAML", "TCGA-LIHC", "TCGA-LUAD",
                          "TCGA-LUSC", "TCGA-ONOV", "TCGA-PAAD",
                          "TCGA-PCPG", "TCGA-PRAD", "TCGA-READ",
                          "TCGA-SARC", "TCGA-STAD", "TCGA-TCGT",
                          "TCGA-THCA", "TCGA-THYM", "TCGA-UCEC")
  
  # select aracne regulatory network name for cohort of interest
  print("Selecting contextually appropriate ARACNe regulatory network")
  regname <- aracne_regs[[cohort]]
  
  # load the regulatory network
  print("Loading the regulatory network")
  reg_network = get(regname)
  
  # write out the regulatory network because aracne2regulon expects a file.
  reg_fl=paste('./', 'tmp-entrez-', cohort, '-reg.adj', sep="")
  print("Writing the regulatory network with entrez IDs to file in ../../data/tf_activity/:")
  print(gsub("./", "", reg_fl))
  write.regulon(reg_network, file=reg_fl)
  
}

main()