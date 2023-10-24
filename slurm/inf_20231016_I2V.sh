#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J test_20231016_I2V
#SBATCH -p gp4d
#SBATCH -e test_20231016_I2V.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 529153
python custom_inference.py --input_image_dir /work/u5832291/datasets/LLVIP/infrared/test --output_image_dir /work/u5832291/yixian/ControlNet_edit/experiments/20231016_I2V/inf_outputs --model_ckpt /work/u5832291/yixian/ControlNet_edit/experiments/20231016_I2V/20231016_I2V-epoch=4-step=15034.ckpt --cldm_model_config /work/u5832291/yixian/ControlNet_edit/models/cldm_v15.yaml