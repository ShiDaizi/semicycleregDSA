from PIL import Image
import os
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
import numpy as np
import pydicom
import kornia.morphology as mm
import config
import matplotlib.pyplot as plt


class DSADataset(Dataset):
    def __init__(self, root_F, root_M, transform=None, edge_require=False):
        self.root_F = root_F
        self.root_M = root_M
        self.transform = transform
        self.supported = ['.dcm', '.png', '.PNG', '.jpg', '.JPG']
        self.edge_require = edge_require
        self.F_imgs = os.listdir(self.root_F)
        self.M_imgs = os.listdir(self.root_M)
        self.imgs = list(set(self.F_imgs) & set(self.M_imgs))
        self.imgs = [i for i in self.imgs if os.path.splitext(i)[-1] in self.supported]
        self.length = len(self.imgs)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        F_img = self.imgs[index % self.length]
        M_img = self.imgs[index % self.length]
        F_path = os.path.join(self.root_F, F_img)
        M_path = os.path.join(self.root_M, M_img)


        if os.path.splitext(F_img)[-1] == '.dcm':
            F_img = np.uint8(pydicom.dcmread(F_path).pixel_array)
            M_img = np.uint8(pydicom.dcmread(M_path).pixel_array)
        else:
            F_img = np.uint8(Image.open(F_path).convert('L'))
            M_img = np.uint8(Image.open(M_path).convert('L'))

        #F_img = cv2.bilateralFilter(F_img, d=15, sigmaColor=75, sigmaSpace=75)
        #M_img = cv2.bilateralFilter(M_img, d=15, sigmaColor=75, sigmaSpace=75)

        if self.transform:
            # F_img = A.Rotate(limit=1.0, interpolation=cv2.INTER_LINEAR, p=0.5)(image=F_img)['image']
            augmentations = self.transform(image=F_img, image0=M_img)
            F_img = augmentations['image']
            M_img = augmentations['image0']


        img = []
        img += [F_img] + [M_img]
        if self.edge_require:
            kernel = torch.zeros(3, 3)
            tmp = F_img.unsqueeze(0) * 0.5 + 0.5
            tmp = mm.opening(mm.closing(tmp, kernel), kernel)
            F_edge = mm.dilation(mm.closing(tmp, kernel), kernel) - mm.closing(tmp, kernel)
            tmp = M_img.unsqueeze(0) * 0.5 + 0.5
            tmp = mm.opening(mm.closing(tmp, kernel), kernel)
            M_edge = mm.dilation(mm.closing(tmp, kernel), kernel) - mm.closing(tmp, kernel)
            F_edge = ((F_edge - F_edge.min()) / (F_edge.max() - F_edge.min())).float()
            M_edge = ((M_edge - M_edge.min()) / (M_edge.max() - M_edge.min())).float()
            #F_edge = (F_edge > 0.04).float()
            #M_edge = (M_edge > 0.04).float()


            img += [F_edge.squeeze(0)] + [M_edge.squeeze(0)]

        return img

def test():
    root_F = './data/F_1'
    root_M = './data/M_1'
    dataset = DSADataset(root_F, root_M, transform=config.transform, edge_require=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.NUM_WORKS, pin_memory=True)
    F, M, F_edge, M_edge = next(iter(loader))
    F = F.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    M = M.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    F_edge = F_edge.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    M_edge = M_edge.squeeze(0).permute(1, 2, 0).data.cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(F, cmap='gray', vmin=-1, vmax=1)
    plt.title('Fixed')
    plt.subplot(2, 3, 2)
    plt.imshow(M, cmap='gray', vmin=-1, vmax=1)
    plt.title('Moved')
    plt.subplot(2, 3, 3)
    plt.imshow(M - F, cmap='gray')
    plt.title('Subtracted')
    plt.subplot(2, 3, 4)
    plt.imshow(F_edge, cmap='gray')
    plt.title('F_edge')
    plt.subplot(2, 3, 5)
    plt.imshow(M_edge, cmap='gray')
    plt.title('M_edge')
    plt.show()

if __name__ == '__main__':
    test()









