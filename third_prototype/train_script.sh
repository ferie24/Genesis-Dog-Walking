#!/bin/bash
#SBATCH --job-name=genesis_resume_B36
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/friemen/working_dir/Genesis-Dog-Walking/third_prototype/logs/slurm/%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd ~/working_dir/Genesis-Dog-Walking/third_prototype


# Standard: B36 ab 6999 für 2000 WEITERE Iterationen (7000–8999) fortsetzen.
# Der Runner verwendet die gespeicherte Run-Konfiguration und einen neuen Logordner.
# Änderungen z.B. über ADDITIONAL_ITERATIONS=1000 oder --iterations 1000.
# Mit --dry-run nur den Plan prüfen. Finale Videos verwenden den neuen Checkpoint.


python runner.py \
  --resume-from logs/go2_genesis_rsl_rl_config_B_v036/model_6999.pt\
  --iterations 3000 \
  --video-interval 0 \
  "$@"
