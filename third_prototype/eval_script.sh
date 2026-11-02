#!/bin/bash
#SBATCH --job-name=genesis_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=rema
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd ~/working_dir/Genesis-Dog-Walking/third_prototype
python eval_video.py \
  --model-path logs/go2_genesis_rsl_rl_config_B_v037/model_9999.pt \
  --commands 0 0.4 0.8 \
  --y-range 0 0 \
  --yaw-commands -0.4 0 0.4 \
  --tolerance 0.1 \
  --yaw-tolerance 0.1 \
  --duration 30 \
  --warmup 2 \
  --repeats 3 \
  --seed 7 \
  --video-dir video/go2_genesis_rsl_rl_config_B_v037
