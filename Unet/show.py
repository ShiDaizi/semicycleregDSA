import torch
import sys
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
import Net
import losses
from dataset import DSADataset
import matplotlib.pyplot as plt

def show():
    gen = Net.VxmDense(inshape=config.inshape, nb_unet_features=config.nb_unet_features)
    opt_gen = optim.Adam(
        gen.parameters(),
        lr=config.LEARNING_RATE_G,
        betas=(0.5, 0.999),
    )

    config.load_checkpoint(
        config.CHECKPOINT_GEN,
        gen,
        opt_gen,
        config.LEARNING_RATE_G,
    )

    dataset = DSADataset(config.ROOT_F, config.ROOT_M, transform=config.transform, edge_require=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.NUM_WORKS, pin_memory=True)

    MSE = losses.MSE()
    Grad = losses.Grad()

    gen.eval()

    F, M, F_edge, M_edge = next(iter(loader))
    Fg, flow = gen(F, M)
    Fg, flow = gen(Fg, M)
    F = F.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    M = M.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    F_edge = F_edge.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    M_edge = M_edge.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    Fg= Fg.squeeze(0).permute(1, 2, 0).data.cpu().numpy()



    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(F, cmap='gray')
    plt.title('F')
    plt.subplot(2, 3, 2)
    plt.imshow(M, cmap='gray')
    plt.title('M')
    plt.subplot(2, 3, 3)
    plt.imshow(M_edge, cmap='gray')
    plt.title('M_edge')
    plt.subplot(2, 3, 4)
    plt.imshow(M - F, cmap='gray')
    plt.title('Original Subtracted')
    plt.subplot(2, 3, 5)
    plt.imshow(M - Fg, cmap='gray')
    plt.title('Registed Subtracted')
    plt.subplot(2, 3, 6)
    plt.imshow(Fg, cmap='gray')
    plt.title('Fg')
    plt.show()

if __name__ == '__main__':
    show()