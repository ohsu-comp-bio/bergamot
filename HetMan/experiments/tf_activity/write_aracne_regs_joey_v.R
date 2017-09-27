# Title     : Generate TCGA cohort regulon
# Objective : Writes out ENTREZID labeled regulon to .data/tmp-{cohort}.adj
# Created by: estabroj
# Created on: 9/22/17

suppressMessages(library(aracne.networks))

basedir <- "."
datadir <- paste("/home/exacloud/lustre1/BioCoders/ProjectCollaborations/PRECEPTS/bergamot/HetMan/data/tf_activity")

main = function(){
    setwd(datadir)
    args = commandArgs(trailingOnly = TRUE)
    cohort = (args[1])
    id = tolower(strsplit(cohort,'-')[[1]][2])
    regulon = paste('regulon', id, sep='')
    tmp_regulon = paste('tmp-', toupper(id), '.adj',sep='')
    write.regulon(get(regulon),file=tmp_regulon)
}

main()
