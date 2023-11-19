#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J colorization_training_20231119_sd_unlocked_epoch_20_w_txt_prompt_modified
#SBATCH -p gp4d
#SBATCH -e train_colorization_training_20231119_sd_unlocked_epoch_20_w_txt_prompt_modified.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 541867
python custom_train.py --train_dataset_cond_dir /work/u5832291/datasets/LLVIP/visible_grayscale_3_channels_0_255/train --train_dataset_target_dir /work/u5832291/datasets/LLVIP/visible/train --val_dataset_cond_dir /work/u5832291/datasets/LLVIP/visible_grayscale_3_channels_0_255/test --val_dataset_target_dir /work/u5832291/datasets/LLVIP/visible/test --batch_size 4 --val_batch_size 4 --num_workers 4 --image_size 256 256 --resume_path /work/u5832291/yixian/ControlNet_edit/models/control_sd15_ini.ckpt --cldm_model_config /work/u5832291/yixian/ControlNet_edit/models/cldm_v15.yaml --sd_locked False --only_mid_control False --max_epochs 20 --save_ckpt_every_n_epoch 1 --save_ckpt_dir /work/u5832291/yixian/ControlNet_edit/experiments/20231119_colorization_sd_unlocked_epoch_20_ORI_CODE_w_txtprompt_modified --save_ckpt_filename 20231119_colorization --wandb_project ControlNet-edit --wandb_run_name colorization_training_20231119  --input_text_prompt "an rgb image of streets at night as viewed from the angle of a security camera, with cars, bikes, pedestrians, and trees" --gpus 0

