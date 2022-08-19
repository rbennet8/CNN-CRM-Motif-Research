#! /bin/bash

#SBATCH --partition=Orion
#SBATCH --job-name="Human CRM Validation"
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10GB
#SBATCH -e HumanVal-%j.err
#SBATCH -o HumanVal-%j.out

echo "=========="
echo "Start Time : $(date)"
echo "Submit Dir : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Num Tasks : $SLURM _NTASKS total [$SLURM_NNODES nodes @ $SLURM_CPUS_ON_NODE CPUs/node]"
echo "=========="
echo ""

module load tensorflow

python3 HumanValidation.py

echo ""
echo "=========="
echo "End Time : $(date)"
echo "=========="
