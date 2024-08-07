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


def train_fn(vm, loader, opt_vm, epoch):
    loop = tqdm(loader, desc=f"Training (Epoch {epoch}) | Loss: {0.0:.4f}", leave=True)
    L2 = losses.MSE()
    # NCC = losses.NCC(win=[7, 7]) # 7_31
    NCC = losses.NCC(win=[7, 7])
    Grad = losses.Grad()
    Grad_1 = losses.SmoothnessLoss(smoothness_order=1, smoothness_const=config.SMOOTHNESS_CONSTANT)
    Grad_2 = losses.SmoothnessLoss(smoothness_order=2, smoothness_const=config.SMOOTHNESS_CONSTANT)
    Pho = losses.PhotometricLoss()

    for idx, img in enumerate(loop):
        F, M, _, _ = img
        F = F.to(config.DEVICE)
        M = M.to(config.DEVICE)
        MASK = (M - F > -1e-2).clone().detach().to(config.DEVICE)
        Fr, flow = vm(F, M)
        # reg_loss = Grad.loss(flow, flow)
        reg_loss = Grad_1(flow[:, 0, ...], F) + Grad_1(flow[:, 1, ...], F)
        if epoch > config.EPOCH_MASK:
            MASK = MASK + (M - Fr > 1e-2).clone().detach().to(config.DEVICE)
        pho_loss = Pho(Fr, M, MASK)
        if epoch > config.EPOCH_NCC:
            pho_loss += config.NCC_LAMBDA * NCC.loss(Fr, M)
        loss = pho_loss + config.LAMBDA * reg_loss
        opt_vm.zero_grad()
        loss.backward()
        opt_vm.step()

        loop.set_description(f"Training (Epoch {epoch}) | Loss: {loss.item():.4f}")

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.1)
def main():
    config.seed_everything()
    vm = Net.VxmDense(inshape=config.inshape, nb_unet_features=config.nb_unet_features).to(config.DEVICE)
    vm.apply(init_weights)
    opt_vm = optim.Adam(
        vm.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    scheduler = optim.lr_scheduler.StepLR(opt_vm, step_size=config.STEP_SIZE, gamma=config.GAMMA)

    if config.LOAD_MODEL:
        config.load_checkpoint(
            config.CHECKPOINT_VM,
            vm,
            opt_vm,
            config.LEARNING_RATE,
        )

    dataset = DSADataset(path=config.PATH, transform=config.transform)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKS, pin_memory=True)


    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            vm,
            loader,
            opt_vm,
            epoch,
        )
        scheduler.step()
        if config.SAVE_MODEL:
            config.save_checkpoint(vm, opt_vm, filename=config.CHECKPOINT_VM)


if __name__ == "__main__":
    main()