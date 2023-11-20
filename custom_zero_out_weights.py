import os
import torch
import argparse
from tqdm import tqdm

def zero_out_weights(input_ckpt_path, output_ckpt_path, mode='zero'):
    # Load the pretrained weights
    state_dict_full = torch.load(input_ckpt_path)
    state_dict_key = 'state_dict'
    state_dict = state_dict_full[state_dict_key]

    # Zero out all the weights in the state dictionary
    print("Altering weights with mode: ", mode)
    for key in tqdm(state_dict):
        if mode == 'zero':
            state_dict[key] = torch.zeros_like(state_dict[key])
        elif mode == 'random':
            random_values = torch.randn_like(state_dict[key])
            state_dict[key] = random_values
    
    # Save the modified state dictionary to a new file
    state_dict_full[state_dict_key] = state_dict
    torch.save(state_dict_full, output_ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='controlnet_edited_inferencing')
    parser.add_argument('--input_ckpt_path', type=str, default='', help='')
    parser.add_argument('--output_ckpt_path', type=str, default='', help='')
    parser.add_argument('--mode', type=str, default='zero', help='zero | random')
    opt = parser.parse_args()

    zero_out_weights(opt.input_ckpt_path, opt.output_ckpt_path, opt.mode)
    print("DONE")


# python custom_zero_out_weights.py --input_ckpt_path /work/u5832291/yixian/ControlNet_edit/models/v1-5-pruned.ckpt --output_ckpt_path /work/u5832291/yixian/ControlNet_edit/models/v1-5-pruned-ZERO.ckpt --mode zero