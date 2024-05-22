import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_DIR = './data/train'
TEST_DIR = './data/test'
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = './checkpoints'
CHECKPOINT_DISC = './checkpoints'

transform = A.Compose(
    [
        A.Resize(height=750, width=750),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=0.5, interpolation=cv2.INTER_LINEAR, p=0.5),
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
        A.CenterCrop(height=712, width=712),
        ToTensorV2(), #ToTensorV2() not convert to [0, 1]
    ],
    additional_targets={'image0': 'image'},
)

