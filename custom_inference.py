# built based on gradio_canny2image.py
import os
import argparse
from PIL import Image
from tqdm import tqdm

from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')  # HxWx3
    return img


def save_img(image_tensor, filename):
    image_pil = Image.fromarray(image_tensor)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


def process(model, ddim_sampler, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        control = torch.from_numpy(img.copy()).float().cuda() / 255.0 # 0.-1.
        control = torch.stack([control for _ in range(num_samples)], dim=0) # batch
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples) # -1. - +1.
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)] # list of HxWx3, 0-255, np.uint8

    return results
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='controlnet_edited_inferencing')
    parser.add_argument('--input_image_dir', type=str, default='', help='directory of input images (condition)')
    parser.add_argument('--output_image_dir', type=str, default='', help='directory for output images')
    parser.add_argument('--image_size', type=int, default=512, help='resize image to [n, n]. Min=256 Max=768 step=64')
    parser.add_argument('--input_text_prompt', type=str, default='', help='fixed input text prompt for sd model')
    parser.add_argument('--model_ckpt', type=str, default='', help='path to trained model checkpoint')
    parser.add_argument('--cldm_model_config', type=str, default='./models/cldm_v15.yaml', help='config file for the cldm v15 / v21 model')
    parser.add_argument('--control_strength', type=float, default=1.0, help='Control Strength. Min=0.0 Max=2.0')
    parser.add_argument('--ddim_steps', type=int, default=20, help='No. of diffusion steps. Min=1 Max=200')
    parser.add_argument('--guidance_scale', type=float, default=9.0, help='Guidance scale. Min=0.1 Max=30.0 step=0.1')
    parser.add_argument('--seed', type=int, default=-1, help='Seed for random noise. -1 for auto random. Min=-1 Max=2147483647')
    opt = parser.parse_args()
    
    # hard coded configs
    num_samples = 1 # fixed to only 1 output per image (max 12)
    guess_mode = False
    eta = 0. # eta (DDIM)
    a_prompt = '' # additional prompt, concat behind original prompt
    n_prompt = '' # negative prompt
    
    # load model
    print("==> Loading Model")
    model = create_model(opt.cldm_model_config).cpu()
    model.load_state_dict(load_state_dict(opt.model_ckpt, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    
    print("==> Reading Images")
    all_files = os.listdir(opt.input_image_dir)
    all_images = sorted([f for f in tqdm(all_files, total=len(all_files)) if is_image_file(f)])
    print(f"Total: {len(all_images)} images")
    
    print("==> Starting Inferencing")
    for f in tqdm(all_images, total=len(all_images)):
        image_path_name = os.path.join(opt.input_image_dir, f)
        input_image = np.array(load_img(image_path_name)) # HxWx3, 0-255, np.uint8
        output_img_list = process(model=model, 
                                  ddim_sampler=ddim_sampler, 
                                  input_image=input_image, 
                                  prompt=opt.input_text_prompt, 
                                  a_prompt=a_prompt, 
                                  n_prompt=n_prompt, 
                                  num_samples=num_samples, 
                                  image_resolution=opt.image_size, 
                                  ddim_steps=opt.ddim_steps, 
                                  guess_mode=guess_mode, 
                                  strength=opt.control_strength, 
                                  scale=opt.guidance_scale, 
                                  seed=opt.seed, 
                                  eta=eta)
        print(f"Got {len(output_img_list)} result")
        output_image_path = os.path.join(opt.output_image_dir, f)
        save_img(output_img_list[0], output_image_path)
    
    print("\nALL DONE")
