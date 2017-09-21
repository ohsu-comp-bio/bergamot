# a script for running VIPER (Califano et al. 2016) to assign activity scores to
# transcription factors based on the expression profiles of their regulons.
# VIPER usage: http://127.0.0.1:23290/library/viper/doc/viper.pdf
# Run from command line: Rscript run_viper.R --args arg1 arg2 arg3
# where arg1=expr.tsv, arg2=phenotype.tsv, arg3=bmeg_cohort (i.e. TCGA-BRCA)
# a bash script calls this one: 
# /Users/manningh/PycharmProjects/bergamot/experiments/tf_activity/run_viper.sh

# Author: Hannah Manning
# Date: Aug 4, 2017

# see here for example files:
# "/Library/Frameworks/R.framework/Versions/3.3/Resources/library/bcellViper"
# TODO: make basedir a relative path
basedir <- "/Users/manningh/PycharmProjects/bergamot/HetMan"
datadir <- paste(basedir,"data/tf_activity/", sep="/")

main <- function() {
  
  # get oriented
  setwd(datadir)
  # allow for command line arguments
  args <- commandArgs(trailingOnly = TRUE)
  # expr_fl should be the filename (not path) for matrix
  # with samples as cols and features as rows
  expr_fl <- paste('./', args[2])
  phen_fl <- paste('./', args[3])
  cohort <- args[4]
  library("viper")
  library("Biobase")
  library(aracne.networks)

  # load expression data
  # exprDataPath <- file.path(datadir, expr_fl)
  expr <- as.matrix(read.table(expr_fl, row.names=1,
                               header=TRUE, sep="\t",
                               as.is=TRUE, check.names=FALSE))
  
  # load phenotyptic data (will be used to separate expr into experimental groups)
  pdat <- read.table(phen_fl, row.names=1, 
                      header=FALSE, sep="\t")
  names(pdat) <- "samp_type"

  # a makeshift assert statement. should have a work around.
  stopifnot(all(rownames(pdat)==colnames(expr)))
  
  # turn it into an AnnotatedDataFrame as ExpressionSet requires
  pData <- AnnotatedDataFrame(data=pdat)
  
  # build a Biobase ExpressionSet
  xset <- ExpressionSet(assayData=expr, phenoData=pData)
  
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
  regname <- aracne_regs[[cohort]]
  
  # load the regulatory network
  reg_network = get(regname)
  
  # write out the regulatory network because aracne2regulon expects a file.
  reg_fl=gsub('pData.tsv', 'reg.adj', phen_fl)
  write.regulon(reg_network, file=reg_fl)
  
  # load the regulatory network into regulon object
  # TODO: map our ensembl ids to their entrez ones
  regul <- aracne2regulon(reg_fl, xset, verbose=FALSE)
  
  # TODO: ADAPT SAMPLE TYPES TO BE REPRESENTATIVE OF ALL IN TCGA COHORTS
  # calculate signatures (vector of t statistics, then transform to Z scores)
  signature <- rowTtest(xset, "samp_type", c("Primary_Tumor", "Metastatic"), 
                        "Solid_Tissue_Normal")
  signature <- (qnorm(signature$p.value/2, lower.tail = FALSE) *
                  sign(signature$statistic))[,1]
  
  # generate a null model for the signature to be compared against
  nullmodel <- ttestNull(xset, "samp_type", c("Primary_Tumor", "Metastatic"), 
                         "Solid_Tissue_Normal", per = 1000,
                         repos = TRUE, verbose = FALSE)
  
  # generate activity scores based on multiple samples
  mrs <- msviper(signature, regul, nullmodel, verbose = FALSE)
  
  # view results:
  # summary(mrs)
  # plot(mrs, cex=0.7)
  
  # generate activity scores based on a *single* sample
  # vpres <- viper(dset, regul, verbose = FALSE)
  # get t statistic and p value for difference between the categorizations of B cells
  # tmp <- rowTtest(vpres, "Group", c(cohort, "Normal))
}