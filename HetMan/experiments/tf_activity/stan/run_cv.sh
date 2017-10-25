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


# move to the working directory, specify output location
# todo: where do you feed it $cohort & $cv_id?
cd ~/lustrehome/PyCharmProjects/bergamot
TEMPDIR=HetMan/experiments/tf_activity/stan/output/$cohort

# get the cross-validation ID defined by this job's SLURM array ID
cv_id=$(($SLURM_ARRAY_TASK_ID % 10));

# disable threading on each CPU, print info about this array job
export OMP_NUM_THREADS=1
echo $cohort
echo $intx
echo $cv_id

# may need to pause between tasks to allow firehose/bmeg to recover?

srun --output=$TEMPDIR/slurm/cv_${cv_id}.txt \
    --error=$TEMPDIR/slurm/cv_${cv_id}.txt \
    python HetMan/experiments/tf_activity/stan/run.py $cohort $cv_id
