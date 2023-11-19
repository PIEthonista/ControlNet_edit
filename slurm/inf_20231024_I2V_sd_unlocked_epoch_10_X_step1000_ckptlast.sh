#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J test_20231024_I2V_sd_unlocked_epoch_10_X_step1000_ckptlast
#SBATCH -p gp4d
#SBATCH -e test_20231024_I2V_sd_unlocked_epoch_10_X_step1000_ckptlast.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 536192
python custom_inference.py --input_image_dir /work/u5832291/datasets/LLVIP/infrared/test --output_image_dir /work/u5832291/yixian/ControlNet_edit/experiments/20231024_I2V_sd_unlocked_epoch_10_ORI_CODE/inf_outputs_step_1000_ckpt_last_X --model_ckpt /work/u5832291/yixian/ControlNet_edit/experiments/20231024_I2V_sd_unlocked_epoch_10_ORI_CODE/last.ckpt --cldm_model_config /work/u5832291/yixian/ControlNet_edit/models/cldm_v15.yaml --ddim_steps 1000