#!/bin/bash

#SBATCH --job-name=dream-train_fit
#SBATCH --partition=exacloud

#SBATCH --array=0-99
#SBATCH --time=2000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000

#SBATCH --output=/home/users/estabroj/scratch/slurm/log-files/TFA-fit_out-%A.txt
#SBATCH --error=/home/users/estabroj/scratch/slurm/log-files/TFA-fit_err-%A.txt
#SBATCH --verbose


# move to the working directory, find where to place output
cd /home/users/estabroj/scratch/bergamot
source activate visions

TEMPDIR=HetMan/experiments/predict_TFA/baseline/output/$cohort/$input/$classif

# disable threading on each CPU, print info about this array job
export OMP_NUM_THREADS=1
echo $cohort
echo $input
echo $classif

# get the cross-validation ID and sub-variant sub-task ID defined by this
# job's SLURM array ID
cv_id=$(($SLURM_ARRAY_TASK_ID / 20));
task_id=$(($SLURM_ARRAY_TASK_ID % 20));

# find the expression effects for the training/testing cohort split defined
# by the cross-validation ID and the sub-variant subset defined by the sub-task ID
srun -p=exacloud \
	--output=$TEMPDIR/slurm/fit-${cv_id}_${task_id}.txt \
	--error=$TEMPDIR/slurm/fit-${cv_id}_${task_id}.err \
	python HetMan/experiments/predict_TFA/baseline/fit.py \
	$cohort $input $classif $cv_id $task_id

