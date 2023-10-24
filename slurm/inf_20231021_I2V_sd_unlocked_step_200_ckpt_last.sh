#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J test_20231021_I2V_sd_unlocked_step_200_ckpt_last
#SBATCH -p gp4d
#SBATCH -e test_20231021_I2V_sd_unlocked_step_200_ckpt_last.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 532070
python custom_inference.py --input_image_dir /work/u5832291/datasets/LLVIP/infrared/test --output_image_dir /work/u5832291/yixian/ControlNet_edit/experiments/20231021_I2V_sd_unlocked_epoch_10/inf_outputs_step_200_ckpt_last --model_ckpt /work/u5832291/yixian/ControlNet_edit/experiments/20231021_I2V_sd_unlocked_epoch_10/last.ckpt --cldm_model_config /work/u5832291/yixian/ControlNet_edit/models/cldm_v15.yaml --ddim_steps 200