#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J tool_add_control
#SBATCH -p gp4d
#SBATCH -e tool_add_control.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 542016
python tool_add_control.py ./models/v1-5-pruned-ZERO.ckpt ./models/control_sd15_ini_ZERO.ckpt
