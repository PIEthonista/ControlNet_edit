#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J test_20231024_I2V_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt_step200_ckptlast
#SBATCH -p gp4d
#SBATCH -e test_20231024_I2V_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt_step200_ckptlast.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 535003
python custom_inference.py --input_image_dir /media/nine/HD_1/HD_1_from_seven/data/LLVIP/visible_grayscale_3_channels_0_255/test --output_image_dir /media/nine/HD_1/HD_1_from_seven/code/ControlNet_edit/experiments/20231118_colorization_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt/inf_outputs_step_200_ckpt_last --model_ckpt /media/nine/HD_1/HD_1_from_seven/code/ControlNet_edit/experiments/20231118_colorization_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt/last.ckpt --cldm_model_config /media/nine/HD_1/HD_1_from_seven/code/ControlNet_edit/models/cldm_v15.yaml --ddim_steps 200 --input_text_prompt "an rgb image of streets at night as viewed from the angle of a security camera"