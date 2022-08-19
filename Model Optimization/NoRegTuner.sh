#! /bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name="No-Reg Tuner"
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=187GB
#SBATCH -e NoRegTuner-%j.err
#SBATCH -o NoRegTuner-%j.out

echo "=========="
echo "Start Time : $(date)"
echo "Submit Dir : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Num Tasks : $SLURM _NTASKS total [$SLURM_NNODES nodes @ $SLURM_CPUS_ON_NODE CPUs/node]"
echo "=========="
echo ""

module load tensorflow

python3 NoRegTuner.py

echo ""
echo "=========="
echo "End Time : $(date)"
echo "=========="
