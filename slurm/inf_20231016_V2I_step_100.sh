#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J test_20231016_V2I_step_100
#SBATCH -p gp4d
#SBATCH -e test_20231016_V2I_step_100.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 529154
python custom_inference.py --input_image_dir /work/u5832291/datasets/LLVIP/visible/test --output_image_dir /work/u5832291/yixian/ControlNet_edit/experiments/20231016_V2I/inf_outputs_steps_100 --model_ckpt /work/u5832291/yixian/ControlNet_edit/experiments/20231016_V2I/20231016_V2I-epoch=4-step=15034.ckpt --cldm_model_config /work/u5832291/yixian/ControlNet_edit/models/cldm_v15.yaml --ddim_steps 100