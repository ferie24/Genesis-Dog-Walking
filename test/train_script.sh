#!/bin/bash
#SBATCH --job-name=genesis_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7:00:00
#SBATCH --output=logs/%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genesis
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd ~/working_dir/Genesis-Dog-Walking/test
python3 train.py -o run4