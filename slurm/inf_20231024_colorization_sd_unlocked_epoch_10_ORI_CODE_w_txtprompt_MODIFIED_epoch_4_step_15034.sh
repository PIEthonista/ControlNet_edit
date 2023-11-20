#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J test_20231024_colorization_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt_MODIFIED_epoch_4_step_15034
#SBATCH -p gp4d
#SBATCH -e test_20231024_colorization_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt_MODIFIED_epoch_4_step_15034.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 541966
python custom_inference.py --input_image_dir /work/u5832291/datasets/LLVIP/visible_grayscale_3_channels_0_255/test --output_image_dir /work/u5832291/yixian/ControlNet_edit/experiments/20231119_colorization_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt_modified/inf_outputs_step_200_epoch_4_step_15034 --model_ckpt /work/u5832291/yixian/ControlNet_edit/experiments/20231119_colorization_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt_modified/20231119_colorization-epoch=4-step=15034.ckpt --cldm_model_config /work/u5832291/yixian/ControlNet_edit/models/cldm_v15.yaml --ddim_steps 200 --input_text_prompt "an rgb image of streets at night as viewed from the angle of a security camera, with cars, bikes, pedestrians, and trees"