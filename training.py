"""Import libraries"""
import os, glob, yaml
from datetime import datetime
import numpy as np
from easydict import EasyDict
import pytz

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import *

from dataloaders.bcic4a import BCICompet2aIV, BCICompet2aIV_TEST
from model.deepconvnet import DeepConvNet
# from dataloaders.preprocessing import preprocessing_vhdr, prepare_label

from utils.train import train


""" Config setting"""
CONFIG_PATH = f"{os.getcwd()}/configs"
filename = "config.yaml"

with open(f"{CONFIG_PATH}/{filename}") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    args = EasyDict(config)
    

# Set Device
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_NUM

cudnn.benchmark = True
cudnn.fastest = True
cudnn.deteministic = True

args.lr = float(args.lr)
args.weight_decay = float(args.weight_decay)

# Set SEED
torch.manual_seed(args.SEED)



def main():
    model = DeepConvNet().to(device=args.gpu)

    args.train_mode = 'train'
    train_data = BCICompet2aIV(args)
    args.train_mode = 'validation'
    validation_data = BCICompet2aIV(args)
    args.train_mode = 'test'
    test_data = BCICompet2aIV_TEST(args)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=256)
    valid_loader = DataLoader(validation_data, shuffle=False, batch_size=1)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)

    train(train_loader, valid_loader, test_loader, model, args)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
