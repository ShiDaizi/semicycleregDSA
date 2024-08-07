from PIL import Image
import os
import cv2
import torch
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
import utils
import random


class DSADataset(Dataset):
    def __init__(self, path, transform=None):
        super(DSADataset, self).__init__()
        self.path = path
        self.transform = transform
        h5f = h5py.File(self.path, 'r')
        self.keys =list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.path, 'r')
        key = self.keys[index]
        Img_F, Img_M, Edge_F, Edge_M = h5f[key]
        Img_F = np.array(Img_F, dtype=np.float32)
        Img_M = np.array(Img_M, dtype=np.float32)
        Edge_F = np.array(Edge_F, dtype=np.float32)
        Edge_M = np.array(Edge_M, dtype=np.float32)
        if self.transform is not None:
            augmentations = self.transform(image=Img_F, image0=Img_M, image1=Edge_F, image2=Edge_M)
            Img_F = augmentations['image']
            Img_M = augmentations['image0']
            Edge_F = augmentations['image1']
            Edge_M = augmentations['image2']
            Img_F = transforms.RandomRotation(0.5)(Img_F)
        return Img_F, Img_M, Edge_F, Edge_M



def prepare_data(root_F, root_M, file_name='data'):
    print('Preparing data...')

    h5f = h5py.File(f'../data/{file_name}.h5', 'w')
    supported = ['.dcm', '.png', '.PNG', '.jpg', '.JPG']
    files_F = os.listdir(root_F)
    files_M = os.listdir(root_M)
    files = list(set(files_F) & set(files_M))
    files.sort()
    files = [i for i in files if os.path.splitext(i)[-1] in supported]
    num = 0
    for i in range(len(files)):
        Img_F = cv2.imread(os.path.join(root_F, files[i]))
        Img_M = cv2.imread(os.path.join(root_M, files[i]))
        Img_F = torch.tensor(Img_F[:, :, 0].copy(), dtype=torch.float32)
        Img_M = torch.tensor(Img_M[:, :, 0].copy(), dtype=torch.float32)
        Img_F = utils.data_normalize(Img_F)
        Img_M = utils.data_normalize(Img_M)
        Edge_F = utils.calc_edge2d(Img_F, grad=False).squeeze(0).squeeze(0)
        Edge_M = utils.calc_edge2d(Img_M, grad=False).squeeze(0).squeeze(0)
        Edge_F = utils.data_normalize(Edge_F)
        Edge_M = utils.data_normalize(Edge_M)
        h5f.create_dataset(name=str(num), data=list([Img_F, Img_M, Edge_F, Edge_M]))
        num += 1
    h5f.close()

    print(f'Data, # samples {num}')


def test():
    root_F = '../data/F_skull'
    root_M = '../data/M_skull'
    # prepare_data(root_F, root_M, file_name='data')
    dataset = DSADataset(path=config.PATH, transform=config.transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.NUM_WORKS, pin_memory=True)
    F, M, F_edge, M_edge = next(iter(loader))

    F = F.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    M = M.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    F_edge = F_edge.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    M_edge = M_edge.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    print(F.shape, F.max(), F.min())
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(F, cmap='gray', vmax=1)
    plt.title('Fixed')
    plt.subplot(2, 3, 2)
    plt.imshow(M, cmap='gray', vmin=0, vmax=1)
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
    root_F = '../data/F_test'
    root_M = '../data/M_test'
    prepare_data(root_F, root_M, file_name='test')
    # test()
