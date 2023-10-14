from share import *

import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from custom_datasets import SameTextPromptDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='controlnet_edited')
    # dataset config
    parser.add_argument('--train_dataset_cond_dir', type=str, default='', help='')
    parser.add_argument('--train_dataset_target_dir', type=str, default='', help='')
    parser.add_argument('--val_dataset_cond_dir', type=str, default='', help='')
    parser.add_argument('--val_dataset_target_dir', type=str, default='', help='')
    parser.add_argument('--batch_size', type=int, default=0, help='')
    parser.add_argument('--val_batch_size', type=int, default=4, help='')
    parser.add_argument('--num_workers', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--image_size', default=[512, 512], nargs='+', help='resize image to [h, w]') # --image_size 64 64
    parser.add_argument('--input_text_prompt', type=str, default='', help='fixed input text prompt for sd model')
    
    # training params
    parser.add_argument('--resume_path', type=str, default='./models/control_sd15_ini.ckpt', # this is a smaller model (~5GB) compared to sd21 (~7GB)
                        help='path to untrained controlnet weights built from tool_add_control.py')
    parser.add_argument('--cldm_model_config', type=str, default='./models/cldm_v15.yaml', 
                        help='config file for the cldm v15 / v21 model')
    parser.add_argument('--sd_locked', type=bool, default=True, help='lock sd decoder part from training')
    parser.add_argument('--only_mid_control', type=bool, default=False, help='apply controlnet to sd bottleneck only')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='')
    
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='Check val every n train epochs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulates grads every k batches before backprop or as set up in the dict')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='If true enables cudnn.benchmark')
    parser.add_argument('--detect_anomaly', type=bool, default=True, help='Enable anomaly detection for the autograd engine.')
    parser.add_argument('--gpus', default=[0], nargs='+', help='Which GPUs to train on (list or str) applied per node')
    parser.add_argument('--precision', type=int, default=32, help='Double precision (64), full precision (32), half precision (16)')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Stop training once this number of epochs is reached. To enable infinite training, set max_epochs to -1')
    parser.add_argument('--max_steps', type=int, default=-1, help='Stop training after this number of steps. Disabled by default (-1)')
    
    # logger config
    parser.add_argument('--logger_freq', type=int, default=300, help='')
    opt = parser.parse_args()
    
    # fix nargs='+' from list of str to list of int
    for i in range(len(opt.image_size)):
        opt.image_size[i] = int(opt.image_size[i])
    for i in range(len(opt.gpus)):
        opt.gpus[i] = int(opt.gpus[i])


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(opt.cldm_model_config).cpu()
    model.load_state_dict(load_state_dict(opt.resume_path, location='cpu'))
    model.learning_rate = opt.learning_rate
    model.sd_locked = opt.sd_locked
    model.only_mid_control = opt.only_mid_control


    train_dataset = SameTextPromptDataset(image_dir_cond=opt.train_dataset_cond_dir,
                                          image_dir_target=opt.train_dataset_target_dir,
                                          image_size=opt.image_size,
                                          input_text_prompt=opt.input_text_prompt)
    val_dataset = SameTextPromptDataset(image_dir_cond=opt.val_dataset_cond_dir,
                                        image_dir_target=opt.val_dataset_target_dir,
                                        image_size=opt.image_size,
                                        input_text_prompt=opt.input_text_prompt)
    train_dataloader = DataLoader(train_dataset, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=opt.num_workers, batch_size=opt.val_batch_size, shuffle=False)
    
    logger = ImageLogger(batch_frequency=opt.logger_freq)
    trainer = pl.Trainer(accumulate_grad_batches=opt.accumulate_grad_batches, 
                         benchmark=opt.cudnn_benchmark,
                         detect_anomaly=opt.detect_anomaly,
                         gpus=opt.gpus, 
                         precision=opt.precision, 
                         max_epochs=opt.max_epochs,
                         max_steps=opt.max_steps,
                         check_val_every_n_epoch=opt.check_val_every_n_epoch,
                         callbacks=[logger])
    
    # Train!
    # fit(model, train_dataloaders=None, val_dataloaders=None, datamodule=None, train_dataloader=None, ckpt_path=None)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print("DONE.")
    