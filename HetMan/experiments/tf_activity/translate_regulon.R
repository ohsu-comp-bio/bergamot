# a script for translating regulons with ENTREZID identifiers to ENSEMBL. Regulons
# are generated from aracne.networks package (v.1.2.0).
# aracne.networks usage https://bioconductor.org/packages/release/data/experiment/manuals/aracne.networks/man/aracne.networks.pdf
# Run from command line: Rscript translate_regulon.R --args args1
# where arg1= TCGA cohort (i.e. TCGA-BRCA)
# Title     : Generate TCGA ENSEMBL regulons
# Objective : Convert ENTREZID labeled VIPER regulons to ENSEMBL
# Created by: Joey Estabrook
# Created on: 9/21/17

suppressMessages(require(AnnotationDbi))
suppressMessages(require(org.Hs.eg.db))
suppressMessages(require(data.table))

getMatrixWithSelectedIds = function(df, columns=list('Regulator','Target'), db='org.Hs.eg.db', type='ENSEMBL', keytype='ENTREZID'){
    df2 = df
    for (column in columns){
        stopifnot( inherits( get(db), "AnnotationDb" ) )
        df2[[column]] = suppressWarnings(mapIds(get(db), keys=as.character(df[[column]]),column=type, keytype=keytype, multiVals='first'))
    }
    return(df2)
}

print("TEST")
basedir <- "."
datadir <- paste(basedir,"/../../data/tf_activity/", sep="/")

main = function(){
    setwd(datadir)
    print(paste("Basedir:", basedir))
    print(paste("Datadir:", datadir))
    args = commandArgs(trailingOnly = TRUE)
    cohort = (args[1])
    print(paste("Cohort:", cohort))s
    id = strsplit(cohort,'-')[[1]][2]
    cohort_table = fread(paste('tmp-entrez-TCGA-', id,'-reg.adj',sep=''))
    mappedIDs = getMatrixWithSelectedIds(cohort_table)
    write.table(mappedIDs,paste('tmp-ensembl-',id,'.adj',sep=''),sep='\t',row.names=F,quote=F)
}

main()