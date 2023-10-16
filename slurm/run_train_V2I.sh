#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J V2I_training_20232016
#SBATCH -p gp4d
#SBATCH -e train_V2I_training_20232016.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 528323 (failed)
# 528508 (failed)
# 528950
python custom_train.py  --train_dataset_cond_dir /work/u5832291/datasets/LLVIP/visible/train --train_dataset_target_dir /work/u5832291/datasets/LLVIP/infrared/train --val_dataset_cond_dir /work/u5832291/datasets/LLVIP/visible/test --val_dataset_target_dir /work/u5832291/datasets/LLVIP/infrared/test --batch_size 4 --val_batch_size 4 --num_workers 4 --image_size 256 256 --resume_path /work/u5832291/yixian/ControlNet_edit/models/control_sd15_ini.ckpt --cldm_model_config /work/u5832291/yixian/ControlNet_edit/models/cldm_v15.yaml --sd_locked True --only_mid_control False --max_epochs 5 --save_ckpt_every_n_epoch 1 --save_ckpt_dir /work/u5832291/yixian/ControlNet_edit/experiments/20231016_V2I --save_ckpt_filename 20231016_V2I

# python custom_train.py 
# --train_dataset_cond_dir /work/u5832291/datasets/LLVIP/visible/train
# --train_dataset_target_dir /work/u5832291/datasets/LLVIP/infrared/train
# --val_dataset_cond_dir /work/u5832291/datasets/LLVIP/visible/test
# --val_dataset_target_dir /work/u5832291/datasets/LLVIP/infrared/test
# --batch_size 4
# --val_batch_size 4
# --num_workers 4
# --image_size 256 256
# --resume_path /work/u5832291/yixian/ControlNet_edit/models/control_sd15_ini.ckpt
# --cldm_model_config /work/u5832291/yixian/ControlNet_edit/models/cldm_v15.yaml
# --sd_locked True
# --only_mid_control False
# --max_epochs 5
# --save_ckpt_every_n_epoch 1
# --save_ckpt_dir /work/u5832291/yixian/ControlNet_edit/experiments/20231016_V2I
# --save_ckpt_filename 20231016_V2I