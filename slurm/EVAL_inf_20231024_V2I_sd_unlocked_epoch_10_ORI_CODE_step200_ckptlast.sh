#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J eval_std_fid
#SBATCH -p gp4d
#SBATCH -e EVAL_inf_20231024_V2I_sd_unlocked_epoch_10_ORI_CODE_step200_ckptlast.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 535626
srun python /work/u5832291/yixian/palette_scene2scene_rec/eval_std_fid.py -s /work/u5832291/datasets/LLVIP/infrared/test -d /work/u5832291/yixian/ControlNet_edit/experiments/20231024_V2I_sd_unlocked_epoch_10_ORI_CODE/inf_outputs_step_200_ckpt_last

# compute FID between two folders
# Found 3463 images in the folder /work/u5832291/datasets/LLVIP/infrared/test
# Found 3463 images in the folder /work/u5832291/yixian/ControlNet_edit/experiments/20231024_V2I_sd_unlocked_epoch_10_ORI_CODE/inf_outputs_step_200_ckpt_last
# make_dataset
#   Std FID: 258.4732924716652
#       FID: 249.25082167001818
# IS (mean): 3.214054308276566
#  IS (std): 0.5130091089383366
