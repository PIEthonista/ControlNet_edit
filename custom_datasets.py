import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SameTextPromptDataset(Dataset):
    def __init__(self, image_dir_cond, image_dir_target, image_size=[512, 512], input_text_prompt=""):
        self.cond_path = image_dir_cond
        self.target_path = image_dir_target
        self.image_filenames = sorted([x for x in os.listdir(self.cond_path)])
        self.image_size = image_size
        self.input_text_prompt = input_text_prompt

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((int(image_size[0]), int(image_size[1]))),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        cond_img = cv2.imread(os.path.join(self.cond_path, self.image_filenames[idx]))
        target_img = cv2.imread(os.path.join(self.target_path, self.image_filenames[idx]))

        # Do not forget that OpenCV read images in BGR order.
        cond_img = cv2.cvtColor(cond_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        
        cond_img = self.transform(cond_img)
        target_img = self.transform(target_img)

        # original code needs (h, w, 3)
        cond_img = np.transpose(cond_img.numpy(), (1,2,0))
        target_img = np.transpose(target_img.numpy(), (1,2,0))

        # Normalize cond_img images to [0, 1].
        cond_img = cond_img.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target_img = (target_img.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target_img, txt=self.input_text_prompt, hint=cond_img)
