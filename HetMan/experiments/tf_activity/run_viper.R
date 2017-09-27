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
basedir <- "."
#datadir <- paste(basedir,"/../../data/tf_activity/", sep="/")

datadir <- paste("/home/exacloud/lustre1/BioCoders/ProjectCollaborations/PRECEPTS/bergamot/HetMan/data/tf_activity")

main <- function() {
  
  # get oriented
  setwd(datadir)
  # allow for command line arguments
  args <- commandArgs(trailingOnly = TRUE)
  # expr_fl should be the filename (not path) for matrix
  # with samples as cols and features as rows
  expr_fl <- paste(args[1], sep="")
  phen_fl <- paste(args[2], sep="")
  print(phen_fl)
  print(expr_fl)
  print(getwd())
  cohort <- args[3]
  suppressMessages(library(viper))
  suppressMessages(library(Biobase))
  suppressMessages(library(aracne.networks))

  print("Loading expression data in R")
  # exprDataPath <- file.path(datadir, expr_fl)
  expr <- as.matrix(read.table(expr_fl, row.names=1,
                               header=TRUE, sep="\t",
                               as.is=TRUE, check.names=FALSE))
  
  print("Loading phenotype data in R")
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
  id = strsplit(cohort,'-')[[1]][2]
  reg_fl=paste('./', 'tmp-ensembl-', id, '.adj', sep='')
  print("Regulatory network with ensembl IDs will be loaded from:")
  print(gsub('./', '', reg_fl))
  
  # UNCOMMENT THIS WHEN ENTREZ-ENSEMBL MAPPING IS COMPLETE
  # load the regulatory network into regulon object
  regul <- aracne2regulon(reg_fl, xset, verbose=FALSE)
  
  print("Calculating signature")
  # TODO: ADAPT SAMPLE TYPES TO BE REPRESENTATIVE OF ALL IN TCGA COHORTS
  # calculate signatures (vector of t statistics, then transform to Z scores)
  signature <- rowTtest(xset, "samp_type", c("Primary_Tumor", "Metastatic"), 
                        "Solid_Tissue_Normal")
  signature <- (qnorm(signature$p.value/2, lower.tail = FALSE) *
                  sign(signature$statistic))[,1]
  
  # UNCOMMENT THIS SECTION WHEN ENTREZ-ENSEMBL MAPPING IS COMPLETE 
  # generate a null model for the signature to be compared against
   nullmodel <- ttestNull(xset, "samp_type", c("Primary_Tumor", "Metastatic"), 
                          "Solid_Tissue_Normal", per = 1000,
                          repos = TRUE, verbose = FALSE)
   
  # # generate activity scores based on multiple samples
   mrs <- msviper(signature, regul, nullmodel, verbose = FALSE)
  
   # Leading-edge analysis
   # Identify genes driving enrichment with GSEA (Subramanian et al)

   mrs <- ledge(mrs)

   mrshadow <- shadow(mrs, regulators = 25, verbose = FALSE)
   save(mrshadow, file=paste(id,'shadow','RData',sep='.'))

   # Synergy analysis
   # Compute enrichment of all co-regulons for top regulons
   # TODO update # of regulators with var

   mrs <- msviperCombinatorial(mrs, regulators = 25, verbose = FALSE)

   # Compare enrichment of co-regulon versus union of corresponding regulons

   mrs <- msviperSynergy(mrs, verbose = FALSE)
   
   # save results:

   n = summary(mrs,length(mrs$signature)) 
   sink(file=paste(id,'top',sum(n$p.value < 0.05,na.rm = T),'values.txt',sep='_'))
   summary(mrs,sum(x$p.value < 0.05,na.rm = T))
   sink()
   
   # save R object 
   save(mrs,file=paste(id,'mrs','RData',sep='.'))

   pdf(filename = paste(id,'top_10.pdf'))
   plot(mrs, cex=0.7)
   dev.off()  

 
   # generate activity scores based on a *single* sample
   vpres <- viper(xset, regul, verbose = FALSE)

   # get t statistic and p value for difference between the categorizations of B cells
   tmp <- rowTtest(vpres, "Group", c(cohort, "Normal"))
   df = data.frame(Gene = rownames(tmp$p.value), t = round(tmp$statistic, 2), "p-value" = signif(tmp$p.value, 3))[order(tmp$p.value),] 
   
   # Save    
   write.table(df,paste(id,'single_sample',sep=''),quote=F,sep='\t')

}

main()
