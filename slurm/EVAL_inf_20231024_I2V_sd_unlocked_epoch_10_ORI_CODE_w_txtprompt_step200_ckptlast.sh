#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J eval_std_fid
#SBATCH -p gp4d
#SBATCH -e EVAL_inf_20231024_I2V_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt_step200_ckptlast.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 535625
srun python /work/u5832291/yixian/palette_scene2scene_rec/eval_std_fid.py -s /work/u5832291/datasets/LLVIP/visible/test -d /work/u5832291/yixian/ControlNet_edit/experiments/20231024_I2V_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt/inf_outputs_step_200_ckpt_last

# compute FID between two folders
# Found 3463 images in the folder /work/u5832291/datasets/LLVIP/visible/test
# Found 3463 images in the folder /work/u5832291/yixian/ControlNet_edit/experiments/20231024_I2V_sd_unlocked_epoch_10_ORI_CODE_w_txtprompt/inf_outputs_step_200_ckpt_last
# make_dataset
#   Std FID: 166.78429235858331
#       FID: 175.38762324167106
# IS (mean): 2.9241946684275333
#  IS (std): 0.2811907749163208
