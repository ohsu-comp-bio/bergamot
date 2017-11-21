#!/bin/bash

#SBATCH --job-name=TFA-stan_cv
#SBATCH --partition=exacloud

#SBATCH --array=0-9
#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16000

#SBATCH --output=/home/users/estabroj/scratch/slurm/log-files/TFA-stan-cv_out-%A.txt
#SBATCH --error=/home/users/estabroj/scratch/slurm/log-files/TFA-stan-cv_err-%A.txt
#SBATCH --verbose


# move to the working directory, find where to place output
cd /home/users/estabroj/scratch/bergamot
TEMPDIR=HetMan/experiments/predict_TFA/stan/output/intx/$cohort

# get the cross-validation ID defined by this job's SLURM array ID
cv_id=$(($SLURM_ARRAY_TASK_ID % 10));

# disable threading on each CPU, print info about this array job
export OMP_NUM_THREADS=1
echo $cohort
echo $intx
echo $cv_id

# pause between starting array jobs to allow Synapse to recover
sleep $(($SLURM_ARRAY_TASK_ID * 7));

srun --output=$TEMPDIR/slurm/cv_${intx}_${cv_id}.txt \
	--error=$TEMPDIR/slurm/cv_${intx}_${cv_id}.txt \
	python HetMan/experiments/predict_TFA/stan/run.py $cohort $intx $cv_id

