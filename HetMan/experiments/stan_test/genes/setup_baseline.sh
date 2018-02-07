#!/bin/bash

#SBATCH --job-name=stan-baseline
#SBATCH --partition=exacloud

#SBATCH --array=0-19
#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/stan-base_out-%A.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/stan-base_err-%A.txt
#SBATCH --verbose


# move to the working directory, find where to place output
cd ~/compbio/bergamot
source activate precepts

OUTDIR=HetMan/experiments/stan_test/genes/setup/baseline_perf
mkdir -p $OUTDIR/slurm

# disable threading on each CPU, print info about this array job
export OMP_NUM_THREADS=1
echo $out_tag
echo $cohort
echo $baseline

# pause between starting array jobs to ease load when downloading -omic datasets
sleep $(($SLURM_ARRAY_TASK_ID * 11));

# get the cross-validation ID and sub-variant sub-task ID defined by this
# job's SLURM array ID
cv_id=$(($SLURM_ARRAY_TASK_ID % 5));
task_id=$(($SLURM_ARRAY_TASK_ID / 5));

# find the expression effects for the training/testing cohort split defined
# by the cross-validation ID and the sub-variant subset defined by the sub-task ID
srun -p=exacloud \
	--output=$OUTDIR/slurm/fit-${cv_id}_${task_id}.txt \
	--error=$OUTDIR/slurm/fit-${cv_id}_${task_id}.err \
	python HetMan/experiments/stan_test/genes/setup_baseline.py \
	$cohort $baseline $cv_id $task_id -v
