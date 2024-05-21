import torch
from torchvision import transforms

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

transform = transforms.Compose([
    transforms.Resize(width=512, height=512),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5)),
])

