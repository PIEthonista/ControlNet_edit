#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J tutorial_train_fill50k
#SBATCH -p gp4d
#SBATCH -e train_tutorial_train_fill50k.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 527841
python tutorial_train.py