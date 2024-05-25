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
import layers
import losses
import discriminator
import generator
from dataset import DSADataset
import matplotlib.pyplot as plt

def show():
    gen_R = Net.VxmDense(inshape=config.inshape, nb_unet_features=config.nb_unet_features, bidir=True)
    gen_B = Net.Unet(inshape=config.inshape, infeats=1, nb_features=config.nb_gen_features)
    disc_R = discriminator.Discriminator(in_channels=2, features=config.nb_disc_features)
    disc_B = discriminator.Discriminator(in_channels=1, features=config.nb_disc_features)
    opt_gen = optim.Adam(
        list(gen_R.parameters()) + list(gen_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_disc = optim.Adam(
        list(disc_R.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    config.load_checkpoint(
        config.CHECKPOINT_GEN_R,
        gen_R,
        opt_gen,
        config.LEARNING_RATE,
    )
    config.load_checkpoint(
        config.CHECKPOINT_GEN_B,
        gen_B,
        opt_gen,
        config.LEARNING_RATE,
    )
    config.load_checkpoint(
        config.CHECKPOINT_DISC_R,
        disc_R,
        opt_disc,
        config.LEARNING_RATE,
    )
    config.load_checkpoint(
        config.CHECKPOINT_DISC_B,
        disc_B,
        opt_disc,
        config.LEARNING_RATE,
    )

    dataset = DSADataset(config.ROOT_F, config.ROOT_M, transform=config.transform, edge_require=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.NUM_WORKS, pin_memory=True)

    MSE = losses.MSE()
    Grad = losses.Grad()

    F, M, F_edge, M_edge = next(iter(loader))
    F = transforms.RandomRotation(degrees=2)(F)
    Mr, Fr, flowr = gen_R(M, F)
    Mrb = gen_B(Mr)
    Frb = gen_B(Fr)
    Mb = gen_B(M)
    Fb = gen_B(F)
    Mbr, Fbr, flowbr = gen_R(Mb, Fb)


    F = F.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    M = M.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    F_edge = F_edge.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    M_edge = M_edge.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    Fr= Fr.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    Mr = Mr.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    Frb = Frb.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    Mrb = Mrb.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    Fb = Fb.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    Mb = Mb.squeeze(0).permute(1, 2, 0).data.cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(F, cmap='gray', vmin=-1, vmax=1)
    plt.title('F')
    plt.subplot(2, 3, 2)
    plt.imshow(M, cmap='gray', vmin=-1, vmax=1)
    plt.title('M')
    plt.subplot(2, 3, 3)
    plt.imshow(M - F, cmap='gray')
    plt.title('Subtracted')
    plt.subplot(2, 3, 4)
    plt.imshow(Fr, cmap='gray')
    plt.title('Fr')
    plt.subplot(2, 3, 5)
    plt.imshow(Mb, cmap='gray')
    plt.title('Mb')
    plt.subplot(2, 3, 6)
    plt.imshow(M - Mb, cmap='gray')
    plt.title('M - Mb')
    plt.show()

if __name__ == '__main__':
    show()