#!/bin/bash

#SBATCH --job-name=tfactivity-stan-cv
#SBATCH --partition=exacloud

#SBATCH --array=0-9
#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16000

#SBATCH --output=/home/exacloud/lustre1/CompBio/manningh/slurm/log-files/tfactivity-stan-cv_out-%A.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/manningh/slurm/log-files/tfactivity-stan-cv_err-%A.txt
#SBATCH --verbose

cd ~/PycharmProjects/bergamot
source activate paulie

# finds the name of the TCGA cohort to use
if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to BRCA"
	export cohort="BRCA"
fi

# finds the name of classifier to use
if [ -z ${classif+x} ]
then
	echo "no classifier, defaulting to Lasso"
	export classif="Lasso"
fi

# gets the output directory where results will be saved, removing it if it already exists
TEMPDIR=HetMan/experiments/tf_activity/stan/output/$cohort/$classif
echo $TEMPDIR
rm -rf $TEMPDIR

# creates the output directory and sub-directories
mkdir -p $TEMPDIR/slurm
mkdir -p $TEMPDIR/tmp
mkdir -p $TEMPDIR/results

# creates the output directory and sub-directories
mkdir -p $TEMPDIR/slurm
mkdir -p $TEMPDIR/tmp
mkdir -p $TEMPDIR/results

# submits the script that pickles training gene names for execution in real time.
srun -p=exacloud \
	--output=$TEMPDIR/slurm/setup.txt --error=$TEMPDIR/slurm/setup.err \
	python HetMan/experiments/tf_activity/stan/setup.py $cohort $classif

# submits the script that fits the model for execution at a later time.
sbatch HetMan/experiments/tf_activity/stan/fit.sh
