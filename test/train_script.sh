#!/bin/bash
#SBATCH --job-name=Genesis_GO2_Dog_Training_FRiemen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task =8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
python train.py