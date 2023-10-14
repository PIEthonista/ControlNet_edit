import numpy as np
from tutorial_dataset import MyDataset

dataset = MyDataset()
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
print(jpg.__class__)
print(hint.__class__)
print(jpg.dtype)
print(hint.dtype)
print(np.mean(jpg), np.min(jpg), np.max(jpg))
print(np.mean(hint), np.min(hint), np.max(hint))

from custom_datasets import SameTextPromptDataset

dataset = SameTextPromptDataset(image_dir_cond='/work/u5832291/datasets/LLVIP/infrared/train', 
                                image_dir_target='/work/u5832291/datasets/LLVIP/visible/train', 
                                image_size=[512, 512], 
                                input_text_prompt="")
print("----------")
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
print(jpg.__class__)
print(hint.__class__)
print(jpg.dtype)
print(hint.dtype)
print(np.sum(np.isnan(jpg).astype(int)))
print(np.sum(np.isnan(hint).astype(int)))
print(np.mean(jpg), np.min(jpg), np.max(jpg))
print(np.mean(hint), np.min(hint), np.max(hint))


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

