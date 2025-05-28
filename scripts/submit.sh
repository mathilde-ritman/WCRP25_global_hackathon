#!/bin/bash
#SBATCH --time=10:00
#SBATCH --mem=32GB
#SBATCH --account=workshop
#SBATCH --partition=standard
#SBATCH --qos=workshop
#SBATCH -e /home/users/train045/Documents/WCRP25_hackathon/errors/err-%j.txt
#SBATCH -o /home/users/train045/Documents/WCRP25_hackathon/outputs/out-%j.txt


source activate wcrp_hackathon
# python /home/users/train045/Documents/WCRP25_hackathon/feb_aug_extreme_w-icon.py $SLURM_ARRAY_TASK_ID
python /home/users/train045/Documents/WCRP25_hackathon/feb_aug_extreme_w.py $SLURM_ARRAY_TASK_ID
# python /home/users/train045/Documents/WCRP25_hackathon/w99pi.py $SLURM_ARRAY_TASK_ID
# python /home/users/train045/Documents/WCRP25_hackathon/cmf_integrated.py $SLURM_ARRAY_TASK_ID
