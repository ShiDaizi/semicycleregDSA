from PIL import Image
import os
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
import h5py
import glob
from scipy import signal
import numpy as np
import pydicom
import kornia.morphology as mm
import config
import matplotlib.pyplot as plt


def data_normalize(data, min_value=0, max_value=1):
    return (data - data.min()) / (data.max() - data.min()) * (max_value - min_value) + min_value


def calc_edge2d(data, grad=False):
    device = data.device
    data = data.float()
    ndim = len(data.shape)
    if ndim == 2:
        data = data.unsqueeze(0).unsqueeze(0)
    elif ndim == 3:
        data = data.unsqueeze(0)

    sobel_x = torch.tensor(
        data=[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]],
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0).to(device)

    sobel_y = torch.tensor(
        data=[[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]],
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0).to(device)

    grad_x = F.conv2d(F.pad(data, pad=(1, 1, 1, 1), mode='replicate'), sobel_x, padding=0)
    grad_y = F.conv2d(F.pad(data, pad=(1, 1, 1, 1), mode='replicate'), sobel_y, padding=0)
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    if grad:
        return magnitude, grad_x, grad_y
    else:
        return magnitude

