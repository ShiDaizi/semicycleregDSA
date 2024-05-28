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
import generator
from dataset import DSADataset
import matplotlib.pyplot as plt

def show():
    gen = Net.Extnet(inshape=config.inshape, infeats=1, nb_gen_features=config.nb_gen_features)
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
    Mg = gen(M)
    F = F.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    M = M.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    F_edge = F_edge.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    M_edge = M_edge.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    Mg= Mg.squeeze(0).permute(1, 2, 0).data.cpu().numpy()



    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(F, cmap='gray')
    plt.title('F')
    plt.subplot(2, 3, 2)
    plt.imshow(M, cmap='gray')
    plt.title('M')
    plt.subplot(2, 3, 3)
    plt.imshow(M - F, cmap='gray')
    plt.title('Subtracted')
    plt.subplot(2, 3, 4)
    plt.imshow(F_edge, cmap='gray')
    plt.title('F_edge')
    plt.subplot(2, 3, 5)
    plt.imshow(M_edge, cmap='gray')
    plt.title('M_edge')
    plt.subplot(2, 3, 6)
    plt.imshow(Mg, cmap='gray', vmin=-1, vmax=0)
    plt.title('Mg')
    plt.show()

if __name__ == '__main__':
    show()