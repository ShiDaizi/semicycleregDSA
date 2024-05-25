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


def train_fn(gen_R, gen_B, disc_R, disc_B, loader, opt_gen, opt_disc, epoch):
    loop = tqdm(loader, desc=f"Training (Epoch {epoch}) | Generator Loss: {0.0:.4f} | Discriminator Loss: {0.0:.4f}", leave=True)
    MSE = losses.MSE()
    Grad = losses.Grad()

    for idx, img in enumerate(loop):
        F, M, F_edge, M_edge = img
        F = F.to(config.DEVICE)
        M = M.to(config.DEVICE)
        F_edge = F_edge.to(config.DEVICE)
        M_edge = M_edge.to(config.DEVICE)

        Mr, Fr, flowr = gen_R(M, F)
        Mrb = gen_B(Mr)
        Frb = gen_B(Fr)

        Mb = gen_B(M)
        Fb = gen_B(F)
        Mbr, Fbr, flowbr = gen_R(Mb, Fb)

        #Generator
        flow_loss = MSE.loss(flowr, flowbr)
        id_loss = (MSE.loss(Fr, Frb) + MSE.loss(F, Fb) + MSE.loss(Fr, Mb) + MSE.loss(F, Mrb) + MSE.loss(F, Mbr))
        Grad_loss = Grad.loss(flowr, flowr) + Grad.loss(flowbr, flowbr)
        D_B_fake_Mb = disc_B(Mb)
        D_B_fake_Fb = disc_B(Fb)
        D_R_fake_r = disc_R(torch.cat((Mr, F), dim=1))
        D_R_fake_b = disc_R(torch.cat((M, Mb), dim=1))
        D_R_fake_rb = disc_R(torch.cat((M, Mrb), dim=1))  #?
        D_R_fake_br = disc_R(torch.cat((M, Mbr), dim=1))  #?
        G_D_loss = (MSE.loss(D_B_fake_Mb, torch.ones_like(D_B_fake_Mb))
                    + MSE.loss(D_B_fake_Fb, torch.zeros_like(D_B_fake_Fb))
                    + MSE.loss(D_R_fake_r, torch.ones_like(D_R_fake_r))
                    + MSE.loss(D_R_fake_b, torch.ones_like(D_R_fake_b))
                    #+ MSE.loss(D_R_fake_rb, torch.ones_like(D_R_fake_rb))
                    #+ MSE.loss(D_R_fake_br, torch.ones_like(D_R_fake_br))
                    )
        gen_loss = (G_D_loss * config.LAMBDA_DISC
                    + id_loss * config.LAMBDA_ID
                    + flow_loss * config.LAMBDA_FLOW
                    + Grad_loss * config.LAMBDA_GRAD
                    )

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()


        #Discrimator
        D_B_real = disc_B(F)
        D_B_fake_Mb = disc_B(Mb.detach())
        D_B_fake_Fb = disc_B(Fb.detach())  #?
        D_B_fake_Mrb = disc_B(Mrb.detach())  #?
        D_B_fake_Frb = disc_B(Frb.detach())  #?
        D_R_real = disc_R(torch.cat((M, F), dim=1))
        D_R_fake_r = disc_R(torch.cat((Mr, F), dim=1).detach())
        D_R_fake_b = disc_R(torch.cat((M, Mb), dim=1).detach())
        D_R_fake_rb = disc_R(torch.cat((M, Mrb), dim=1).detach())  #?
        D_R_fake_br = disc_R(torch.cat((M, Mbr), dim=1).detach())  #?
        D_B_loss = (MSE.loss(D_B_real, torch.ones_like(D_B_real))
                    + MSE.loss(D_B_fake_Mb, torch.zeros_like(D_B_fake_Mb))
                    + MSE.loss(D_B_fake_Fb, torch.zeros_like(D_B_fake_Fb))
                    #+ MSE.loss(D_B_fake_Mrb, torch.zeros_like(D_B_fake_Mrb))
                    #+ MSE.loss(D_B_fake_Frb, torch.zeros_like(D_B_fake_Frb))
                    )
        D_R_loss = (MSE.loss(D_R_real, torch.ones_like(D_R_real))
                    + MSE.loss(D_R_fake_r, torch.zeros_like(D_R_fake_r))
                    + MSE.loss(D_R_fake_b, torch.ones_like(D_R_fake_b))
                    #+ MSE.loss(D_R_fake_rb, torch.zeros_like(D_R_fake_rb))
                    #+ MSE.loss(D_R_fake_br, torch.zeros_like(D_R_fake_br))
                    )
        disc_loss = D_B_loss + D_R_loss

        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()
        loop.set_description(f"Training (Epoch {epoch}) | Generator Loss: {gen_loss.item():.4f} | Discriminator Loss: {disc_loss:.4f}")


def main():
    gen_R = Net.VxmDense(inshape=config.inshape, nb_unet_features=config.nb_unet_features, bidir=True).to(config.DEVICE)
    gen_B = Net.Unet(inshape=config.inshape, infeats=1, nb_features=config.nb_gen_features).to(config.DEVICE)
    disc_R = discriminator.Discriminator(in_channels=2, features=config.nb_disc_features).to(config.DEVICE)
    disc_B = discriminator.Discriminator(in_channels=1, features=config.nb_disc_features).to(config.DEVICE)
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

    if config.LOAD_MODEL:
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
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKS, pin_memory=True)


    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            gen_R,
            gen_B,
            disc_R,
            disc_B,
            loader,
            opt_gen,
            opt_disc,
            epoch
        )

        if config.SAVE_MODEL:
            config.save_checkpoint(gen_R, opt_gen, filename=config.CHECKPOINT_GEN_R)
            config.save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
            config.save_checkpoint(disc_R, opt_disc, filename=config.CHECKPOINT_DISC_R)
            config.save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_DISC_B)



if __name__ == "__main__":
    main()