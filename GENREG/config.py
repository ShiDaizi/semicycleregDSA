import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import random
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 1e-2
LAMBDA_DISC = 1
LAMBDA_GRAD = 10
LAMBDA_ID = 100
LAMBDA_EDGE = 1000
LAMBDA_FLOW = 10
NUM_WORKS = 0
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_R = './checkpoints/GEN_R.pth.tar'
CHECKPOINT_GEN_B = './checkpoints/GEN_B.pth.tar'
CHECKPOINT_DISC_R = './checkpoints/DISC_R.pth.tar'
CHECKPOINT_DISC_B = './checkpoints/DISC_B.pth.tar'
inshape = (512, 512)
#inshape = (704, 704)
nb_unet_features = [[16, 32, 32, 32], [32, 32, 32, 16, 16]]
nb_gen_features = [[32, 64, 128, 128], [128, 128, 64, 32, 1]]
nb_disc_features = [32, 64, 128]
ROOT_F = '../data/F_1'
ROOT_M = '../data/M_1'


transform = A.Compose(
    [
        A.Resize(height=528, width=528),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=0.5, interpolation=cv2.INTER_LINEAR, p=0.5),
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
        A.CenterCrop(height=inshape[0], width=inshape[1]),
        ToTensorV2(), #ToTensorV2() not convert to [0, 1]
    ],
    additional_targets={'image0': 'image'},
)

def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


