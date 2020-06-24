import os
import torch
from hyper import *
from PIL import Image
import torchvision.transforms.functional as F


def load_single_sp_gt(sp_name, gt_name):
    audio = Image.open(sp_name).convert('RGB')
    audio_data = F.to_tensor(audio)
    audio_data = F.normalize(audio_data, MEAN, STD)

    gt = Image.open(gt_name).convert('L')
    gt_data = F.to_tensor(gt)

    return audio_data, gt_data
