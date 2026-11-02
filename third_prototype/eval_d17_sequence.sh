#!/bin/bash
#SBATCH --job-name=eval_B37_sequence
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=rema
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/friemen/working_dir/Genesis-Dog-Walking/third_prototype/logs/slurm/%j.out

set -euo pipefail

source /home/friemen/miniconda3/etc/profile.d/conda.sh
conda activate test
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

cd /home/friemen/working_dir/Genesis-Dog-Walking/third_prototype

# Standard: eine Folge, 2 s Anlauf + 71 s Kommandosequenz (73 s insgesamt).
# Enthält 0.7/0.8 m/s, schnelle Kurven in beide Richtungen und Stopp aus 0.8.
# Für die Statistik z.B. --repeats 50 --no-video übergeben.
# Standard: aktuelles Terrain; mit --terrain flat auch auf ebener Fläche testbar.
# Videos, Zeitreihen (CSV), Kurven (PNG) und metrics.json werden gemeinsam gespeichert.
printf 'Sequenzauswertung auf %s (Slurm-Job %s)\n' "$(hostname)" "${SLURM_JOB_ID:-lokal}"
python -u -X faulthandler sequence_eval.py \
  --model-path logs/go2_genesis_rsl_rl_config_B_v037/model_9999.pt \
  --repeats 1 \
  --seed 100 \
  "$@"
