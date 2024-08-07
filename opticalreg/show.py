import torch
import sys
import os
import cv2
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
import Net
import layers
import losses
import discriminator
import generator
from dataset import DSADataset
import matplotlib.pyplot as plt
import numpy as np

def minmaxscaler(data):
    min = data.min()
    max = data.max()
    return (data - min)/(max-min)

def show():
    vm = Net.VxmDense(inshape=config.inshape, nb_unet_features=config.nb_unet_features).to(config.DEVICE)
    opt_vm = optim.Adam(
        vm.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    config.load_checkpoint(
        config.CHECKPOINT_VM,
        vm,
        opt_vm,
        config.LEARNING_RATE,
    )

    dataset = DSADataset(path=config.TEST_PATH, transform=config.transform)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKS, pin_memory=True)

    MSE = losses.MSE()
    Grad = losses.Grad()

    vm.eval()

    F, M, F_edge, M_edge = next(iter(loader))
    F = F[0].to(config.DEVICE).unsqueeze(0)
    M = M[0].to(config.DEVICE).unsqueeze(0)

    Fr, flow = vm(F, M)
    MASK = (M - Fr > -1e-2).clone().detach().to(config.DEVICE)
    F = F.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    M = M.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    Fr = Fr.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    MASK = MASK.squeeze(0).permute(1, 2, 0).data.cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(F, cmap='gray', vmin=0, vmax=1)
    plt.title('F')
    plt.subplot(2, 3, 2)
    plt.imshow(M, cmap='gray', vmin=0, vmax=1)
    plt.title('M')
    plt.subplot(2, 3, 3)
    plt.imshow(M - F, cmap='gray', vmin=-0.3, vmax=0.3)
    plt.title('Subtracted')
    plt.subplot(2, 3, 4)
    plt.imshow(Fr, cmap='gray')
    plt.title('Fr')
    plt.subplot(2, 3, 5)
    plt.imshow(MASK, cmap='gray')
    plt.title('Mask')
    plt.subplot(2, 3, 6)
    plt.imshow((M - Fr), cmap='gray', vmin=-0.3, vmax=0.3)
    plt.title('M - Fr')
    plt.show()

if __name__ == '__main__':
    show()