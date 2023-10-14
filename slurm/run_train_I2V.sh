#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J I2V_training_20232014
#SBATCH -p gp4d
#SBATCH -e train_I2V_training_20232014.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 528322
python custom_train.py --train_dataset_cond_dir /work/u5832291/datasets/LLVIP/infrared/train --train_dataset_target_dir /work/u5832291/datasets/LLVIP/visible/train --val_dataset_cond_dir /work/u5832291/datasets/LLVIP/infrared/test --val_dataset_target_dir /work/u5832291/datasets/LLVIP/visible/test --batch_size 4 --val_batch_size 4 --num_workers 4 --image_size 256 256 --resume_path /work/u5832291/yixian/ControlNet_edit/models/control_sd15_ini.ckpt --cldm_model_config /work/u5832291/yixian/ControlNet_edit/models/cldm_v15.yaml --sd_locked True --only_mid_control False --max_epochs 1000

# python custom_train.py 
# --train_dataset_cond_dir /work/u5832291/datasets/LLVIP/infrared/train
# --train_dataset_target_dir /work/u5832291/datasets/LLVIP/visible/train
# --val_dataset_cond_dir /work/u5832291/datasets/LLVIP/infrared/test
# --val_dataset_target_dir /work/u5832291/datasets/LLVIP/visible/test
# --batch_size 4
# --val_batch_size 4
# --num_workers 4
# --image_size 256 256
# --resume_path /work/u5832291/yixian/ControlNet_edit/models/control_sd15_ini.ckpt
# --cldm_model_config /work/u5832291/yixian/ControlNet_edit/models/cldm_v15.yaml
# --sd_locked True
# --only_mid_control False
# --max_epochs 1000