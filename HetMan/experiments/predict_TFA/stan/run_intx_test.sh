#!/bin/bash

#SBATCH --job-name=TFA-stan-intx
#SBATCH --partition=exacloud
#SBATCH --time=20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --output=/home/users/estabroj/scratch/slurm/log-files/TFA-stan-intx_%j.out
#SBATCH --error=/home/users/estabroj/scratch/slurm/log-files/TFA-stan-intx_%j.err
#SBATCH --verbose


cd /home/users/estabroj/scratch/bergamot
source activate paulie

# finds the name of the TCGA cohort to use
if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to BRCA"
	export cohort="BRCA"
fi

# gets the output directory where results will be saved,
# removing it if it already exists
TEMPDIR=HetMan/experiments/predict_TFA/stan/output/intx/$cohort
echo $TEMPDIR
rm -rf $TEMPDIR

# creates the output directory and sub-directories
mkdir -p $TEMPDIR/slurm
mkdir -p $TEMPDIR/tmp
mkdir -p $TEMPDIR/results

intx_list=('controls-expression-of' 'controls-phosphorylation-of' 'controls-state-change-of')
for intx_use in "${intx_list[@]}";
	do export intx=$intx_use;
	sbatch HetMan/experiments/predict_TFA/stan/run_cv.sh;
done;

