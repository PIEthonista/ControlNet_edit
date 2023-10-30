#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J eval_std_fid
#SBATCH -p gp4d
#SBATCH -e EVAL_inf_20231024_I2V_sd_unlocked_epoch_10_ORI_CODE_step200_ckptlast.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 535624
srun python /work/u5832291/yixian/palette_scene2scene_rec/eval_std_fid.py -s /work/u5832291/datasets/LLVIP/visible/test -d /work/u5832291/yixian/ControlNet_edit/experiments/20231024_I2V_sd_unlocked_epoch_10_ORI_CODE/inf_outputs_step_200_ckpt_last

# compute FID between two folders
# Found 3463 images in the folder /work/u5832291/datasets/LLVIP/visible/test
# Found 3463 images in the folder /work/u5832291/yixian/ControlNet_edit/experiments/20231024_I2V_sd_unlocked_epoch_10_ORI_CODE/inf_outputs_step_200_ckpt_last
# make_dataset
#   Std FID: 203.87812114740024
#       FID: 225.02173942597102
# IS (mean): 3.998450300798132
#  IS (std): 0.7616541350184131
