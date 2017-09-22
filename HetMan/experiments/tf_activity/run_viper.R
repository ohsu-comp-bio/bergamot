# a script for running VIPER (Califano et al. 2016) to assign activity scores to
# transcription factors based on the expression profiles of their regulons.
# VIPER usage: http://127.0.0.1:23290/library/viper/doc/viper.pdf
# Run from command line: Rscript run_viper.R --args arg1 
# where arg1=bmeg_cohort (i.e. TCGA-BRCA)
# a bash script calls this one: 
# /Users/manningh/PycharmProjects/bergamot/experiments/tf_activity/run_viper.sh

# Author: Hannah Manning
# Date: Aug 4, 2017

# see here for example files:
# "/Library/Frameworks/R.framework/Versions/3.3/Resources/library/bcellViper"
# TODO: make basedir a relative path
basedir <- "."
datadir <- paste(basedir,"/../../data/tf_activity/", sep="/")

main <- function() {
  
  # get oriented
  setwd(datadir)
  # allow for command line arguments
  args <- commandArgs(trailingOnly = TRUE)
  # expr_fl should be the filename (not path) for matrix
  # with samples as cols and features as rows
  cohort <- args[1]
  expr_fl <- paste("./tmp-", cohort, "-expression.tsv", sep="")
  phen_fl <- paste("./tmp-", cohort, "-pData.tsv", sep="")
  
  suppressMessages(library("viper"))
  suppressMessages(library("Biobase"))
  suppressMessages(library(aracne.networks))

  print("Loading expression data in R")
  # exprDataPath <- file.path(datadir, expr_fl)
  expr <- as.matrix(read.table(expr_fl, row.names=1,
                               header=TRUE, sep="\t",
                               as.is=TRUE, check.names=FALSE))
  
  print("Loading phenotype data in R")
  print(paste("phen_fl:", phen_fl))
  # load phenotyptic data (will be used to separate expr into experimental groups)
  pdat <- read.table(phen_fl, row.names=1, 
                      header=FALSE, sep="\t")
  names(pdat) <- "samp_type"

  # a makeshift assert statement. should have a work around.
  stopifnot(all(rownames(pdat)==colnames(expr)))
  
  # turn it into an AnnotatedDataFrame as ExpressionSet requires
  pData <- AnnotatedDataFrame(data=pdat)
  
  print("Generating Biobase ExpressionSet")
  # build a Biobase ExpressionSet
  xset <- ExpressionSet(assayData=expr, phenoData=pData)
  
  # specify name of regulon file with ENSEMBL ids
  reg_fl=paste('./', 'tmp-ensembl-', cohort, '-reg.adj', sep="")
  print("Regulatory network with ensembl IDs will be loaded from:")
  print(gsub('./', '', reg_fl))
  
  # UNCOMMENT THIS WHEN ENTREZ-ENSEMBL MAPPING IS COMPLETE
  # load the regulatory network into regulon object
  # regul <- aracne2regulon(reg_fl, xset, verbose=FALSE)
  
  print("Calculating signature")
  # TODO: ADAPT SAMPLE TYPES TO BE REPRESENTATIVE OF ALL IN TCGA COHORTS
  # calculate signatures (vector of t statistics, then transform to Z scores)
  signature <- rowTtest(xset, "samp_type", c("Primary_Tumor", "Metastatic"), 
                        "Solid_Tissue_Normal")
  signature <- (qnorm(signature$p.value/2, lower.tail = FALSE) *
                  sign(signature$statistic))[,1]
  
  # generate a null model for the signature to be compared against
  print("Generating null model")
  nullmodel <- ttestNull(xset, "samp_type", c("Primary_Tumor", "Metastatic"), 
                          "Solid_Tissue_Normal", per = 1000,
                          repos = TRUE, verbose = FALSE)
   
  # generate activity scores based on multiple samples
  print("Generating activity scores based on multiple samples (msviper)")
  mrs <- msviper(signature, regul, nullmodel, verbose = FALSE)
  
  # view results:
  # summary(mrs)
  # plot(mrs, cex=0.7)
  
  # generate activity scores based on a *single* sample
  # vpres <- viper(dset, regul, verbose = FALSE)
  # get t statistic and p value for difference between the categorizations of B cells
  # tmp <- rowTtest(vpres, "Group", c(cohort, "Normal))
  
  print("Reached the end of run_viper.R")
}

main()