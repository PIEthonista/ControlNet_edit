import numpy as np
from PIL import Image
from tutorial_dataset import MyDataset

# dataset = MyDataset()
# print(len(dataset))

# item = dataset[1234]
# target_img = item['target_img']
# txt = item['txt']
# cond_img = item['cond_img']
# print(txt)
# print(target_img.shape)
# print(cond_img.shape)
# print(target_img.__class__)
# print(cond_img.__class__)
# print(target_img.dtype)
# print(cond_img.dtype)
# print(np.mean(target_img), np.min(target_img), np.max(target_img))
# print(np.mean(cond_img), np.min(cond_img), np.max(cond_img))

from custom_datasets import LLVIPSameTextPromptDataset

dataset = LLVIPSameTextPromptDataset(image_dir_cond='/Users/gohyixian/Desktop/LLVIP/infrared/test', 
                                image_dir_target='/Users/gohyixian/Desktop/LLVIP/visible/test', 
                                image_size=[512, 512], 
                                input_text_prompt="")
print("----------")
print(len(dataset))

item = dataset[100]
target_img = item['jpg']
cond_img = item['hint']
txt = item['txt']

print(txt)
print(target_img.shape)
print(cond_img.shape)
print(target_img.__class__)
print(cond_img.__class__)
print(target_img.dtype)
print(cond_img.dtype)
print(np.sum(np.isnan(target_img).astype(int)))
print(np.sum(np.isnan(cond_img).astype(int)))
print(np.mean(target_img), np.min(target_img), np.max(target_img))
print(np.mean(cond_img), np.min(cond_img), np.max(cond_img))

target_img = (target_img + 1.0) * 127.5
target_img = target_img.astype(np.uint8)

cond_img = cond_img * 255.0
cond_img = cond_img.astype(np.uint8)

target_img = Image.fromarray(target_img)
cond_img = Image.fromarray(cond_img)

target_img.save('/Users/gohyixian/Desktop/LLVIP/target_img.png')
cond_img.save('/Users/gohyixian/Desktop/LLVIP/cond_img.png')


# 50000
# burly wood circle with orange background
# (512, 512, 3)
# (512, 512, 3)
# <class 'numpy.ndarray'>
# <class 'numpy.ndarray'>
# float32
# float32
# ----------
# 12025

# (512, 512, 3)
# (512, 512, 3)
# <class 'numpy.ndarray'>
# <class 'numpy.ndarray'>
# float32
# float32

