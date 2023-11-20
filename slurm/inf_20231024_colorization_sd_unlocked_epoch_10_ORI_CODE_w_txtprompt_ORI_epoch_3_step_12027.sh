#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J test_20231024_colorization_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt_ORI_epoch_3_step_12027
#SBATCH -p gp4d
#SBATCH -e test_20231024_colorization_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt_ORI_epoch_3_step_12027.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 542054
python custom_inference.py --input_image_dir /work/u5832291/datasets/LLVIP/visible_grayscale_3_channels_0_255/test --output_image_dir /work/u5832291/yixian/ControlNet_edit/experiments/20231119_colorization_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt_ori/inf_outputs_step_200_epoch_3_step_12027 --model_ckpt /work/u5832291/yixian/ControlNet_edit/experiments/20231119_colorization_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt_ori/20231119_colorization-epoch=3-step=12027.ckpt --cldm_model_config /work/u5832291/yixian/ControlNet_edit/models/cldm_v15.yaml --ddim_steps 200 --input_text_prompt "an rgb image of streets at night as viewed from the angle of a security camera"