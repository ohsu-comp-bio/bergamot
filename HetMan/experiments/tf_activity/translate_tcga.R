# a script for translating regulons with ENTREZID identifiers to ENSEMBL. Regulons
# are generated from aracne.networks package (v.1.2.0).
# aracne.networks usage https://bioconductor.org/packages/release/data/experiment/manuals/aracne.networks/man/aracne.networks.pdf
# Run from command line: Rscript translate_regulon.R --args args1
# where arg1= TCGA cohort (i.e. TCGA-BRCA)
# Title     : Generate TCGA NCBI geneID to ENTREZID
# Objective : Convert NCBI geneID labeled TCGA expression data to ENTREZID
# Created by: Joey Estabrook
# Created on: 9/21/17




suppressMessages(require(AnnotationDbi))
suppressMessages(require(org.Hs.eg.db))
suppressMessages(require(data.table))

getMatrixWithSelectedIds = function(df, columns=list('V1'), db='org.Hs.eg.db', type='ENTREZID', keytype='SYMBOL'){
    df2 = df
    for (column in columns){
        stopifnot( inherits( get(db), "AnnotationDb" ) )
        df2[[column]] = suppressWarnings(mapIds(get(db), keys=as.character(df[[column]]),column=type, keytype=keytype, multiVals='first'))
    }
    return(df2)
}

basedir <- "."

datadir <- paste("/home/exacloud/lustre1/BioCoders/ProjectCollaborations/PRECEPTS/bergamot/HetMan/data/tf_activity")

main = function(){
    setwd(datadir)
    args = commandArgs(trailingOnly = TRUE)
    cohort = (args[1])
    cohort_table = fread(paste('tmp-', cohort,'-expression.tsv',sep=''))
    mappedIDs = getMatrixWithSelectedIds(cohort_table)
    # filter NA gene ids
    filt_mapped = mappedIDs[complete.cases(mappedIDs),]
    write.table(filt_mapped,paste('tmp-entrez-',cohort,'-expression.tsv',sep=''),sep='\t',row.names=F,quote=F)
}

main()
