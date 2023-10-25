#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J I2V_training_20231024_sd_unlocked_epoch_10_w_txt_prompt
#SBATCH -p gp4d
#SBATCH -e train_I2V_training_20231024_sd_unlocked_epoch_10_w_txt_prompt.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 528322 (failed)
# 528509 (failed)
# 528958 (done, epoch 5)
# 531204 (done, epoch 7)
# 531484 (done, epoch 10, sd unlocked)
# 534013 (running, epoch 10, sd unlocked, ori code)
# python custom_train.py  --train_dataset_cond_dir /work/u5832291/datasets/LLVIP/infrared/train --train_dataset_target_dir /work/u5832291/datasets/LLVIP/visible/train --val_dataset_cond_dir /work/u5832291/datasets/LLVIP/infrared/test --val_dataset_target_dir /work/u5832291/datasets/LLVIP/visible/test --batch_size 4 --val_batch_size 4 --num_workers 4 --image_size 256 256 --resume_path /work/u5832291/yixian/ControlNet_edit/models/control_sd15_ini.ckpt --cldm_model_config /work/u5832291/yixian/ControlNet_edit/models/cldm_v15.yaml --sd_locked False --only_mid_control False --max_epochs 10 --save_ckpt_every_n_epoch 1 --save_ckpt_dir /work/u5832291/yixian/ControlNet_edit/experiments/20231024_I2V_sd_unlocked_epoch_10_ORI_CODE --save_ckpt_filename 20231024_I2V  --wandb_project ControlNet-edit --wandb_run_name I2V_training_20231024

# 534015 (running, epoch 10, sd unlocked, ori code, txt prompt)
python custom_train.py  --train_dataset_cond_dir /work/u5832291/datasets/LLVIP/infrared/train --train_dataset_target_dir /work/u5832291/datasets/LLVIP/visible/train --val_dataset_cond_dir /work/u5832291/datasets/LLVIP/infrared/test --val_dataset_target_dir /work/u5832291/datasets/LLVIP/visible/test --batch_size 4 --val_batch_size 4 --num_workers 4 --image_size 256 256 --resume_path /work/u5832291/yixian/ControlNet_edit/models/control_sd15_ini.ckpt --cldm_model_config /work/u5832291/yixian/ControlNet_edit/models/cldm_v15.yaml --sd_locked False --only_mid_control False --max_epochs 10 --save_ckpt_every_n_epoch 1 --save_ckpt_dir /work/u5832291/yixian/ControlNet_edit/experiments/20231024_I2V_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt --save_ckpt_filename 20231024_I2V  --wandb_project ControlNet-edit --wandb_run_name I2V_training_20231024 --input_text_prompt "an rgb image of streets at night as viewed from the angle of a security camera"
