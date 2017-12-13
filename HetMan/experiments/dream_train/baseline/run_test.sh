#!/bin/bash

#SBATCH --job-name=TFA-train
#SBATCH --partition=exacloud
#SBATCH --mem=2000
#SBATCH --time=20

#SBATCH --output=/home/users/estabroj/scratch/slurm/log-files/TFA-train_%j.out
#SBATCH --error=/home/users/estabroj/scratch/slurm/log-files/TFA-train_%j.err
#SBATCH --verbose


cd /home/users/estabroj/scratch/bergamot
source activate visions

# finds the name of the TCGA cohort to use
if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to BRCA"
	export cohort="BRCA"
fi

# finds the name of the -omics dataset(s) to use as prediction input
if [ -z ${input+x} ]
then
	echo "no input defined, defaulting to rna"
	export input="rna"
fi

# finds the name of the regressor to use for prediction
if [ -z ${classif+x} ]
then
	echo "no classifier, defaulting to ElasticNet"
	export classif="ElasticNet"
fi

# gets the output directory where results will be saved,
# removing it if it already exists
TEMPDIR=HetMan/experiments/predict_TFA/baseline/output/$cohort/$input/$classif
echo $TEMPDIR
rm -rf $TEMPDIR

mkdir -p $TEMPDIR/slurm
mkdir -p $TEMPDIR/tmp
mkdir -p $TEMPDIR/results

srun -p=exacloud \
	--output=$TEMPDIR/slurm/setup.txt --error=$TEMPDIR/slurm/setup.err \
	python HetMan/experiments/predict_TFA/baseline/setup.py \
	$cohort $input $classif

sbatch HetMan/experiments/predict_TFA/baseline/fit.sh

