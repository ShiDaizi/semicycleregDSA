import torch
import sys
import os
from torch.optim.lr_scheduler import LambdaLR
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


def train_fn(gen, loader, opt_gen, epoch):
    loop = tqdm(loader, desc=f"Training (Epoch {epoch}) | Generator Loss: {0.0:.4f} | Discriminator Loss: {0.0:.4f}", leave=True)
    L2 = losses.MSE()
    Grad = losses.Grad()
    L1 = losses.L1Loss()
    BCE = losses.CrossEntropyLoss()


    for idx, img in enumerate(loop):
        F, M = img
        F = F.to(config.DEVICE)
        M = M.to(config.DEVICE)
        Mg = gen(M)
        sim_loss = L2.loss(Mg, M - F)
        opt_gen.zero_grad()
        sim_loss.backward()
        opt_gen.step()

        loop.set_description(f"Training (Epoch {epoch}) | Generator Loss: {sim_loss.item():.4f}")

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
def main():
    gen = Net.Extnet(inshape=config.inshape, infeats=1, nb_gen_features=config.nb_gen_features).to(config.DEVICE)
    gen.apply(init_weights)
    opt_gen = optim.Adam(
        gen.parameters(),
        lr=config.LEARNING_RATE_G,
        betas=(0.5, 0.999),
    )

    if config.LOAD_MODEL:
        config.load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE_G,
        )

    dataset = DSADataset(config.ROOT_F, config.ROOT_M, transform=config.transform, edge_require=False)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKS, pin_memory=True)


    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            gen,
            loader,
            opt_gen,
            epoch
        )

        if config.SAVE_MODEL:
            config.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)



if __name__ == "__main__":
    main()