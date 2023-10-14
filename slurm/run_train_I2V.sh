#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J I2V_training_20232014
#SBATCH -p gp4d
#SBATCH -e train_I2V_training_20232014.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 527841
python custom_train.py 
--train_dataset_cond_dir
--train_dataset_target_dir
--val_dataset_cond_dir
--val_dataset_target_dir
--batch_size 4
--val_batch_size 4
--num_workers 8
--image_size 512 512
--resume_path
--cldm_model_config
--sd_locked True
--only_mid_control False
--max_epochs 1000