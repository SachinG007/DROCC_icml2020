from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from datasets.main import load_dataset
from utils import *
from dataset import *
from models import CIFAR10_LeNet
from train_cifar_pgd import *


#PARSER ARGUMENTS
torch.set_printoptions(precision=5)
# torch.manual_seed(2)
# np.random.seed(2)
parser = argparse.ArgumentParser(description='PyTorch Simple Training')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--mom', type=float, default=0.0, metavar='M',
                    help='momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model_dir', default='./log/model',
                    help='directory of model for saving checkpoint')
parser.add_argument('--one_class_adv', type=int, default=1, metavar='N',
                    help='adv loss to be used or not')
parser.add_argument('--inp_radius', type=float, default=0.2, metavar='N',
                    help='radius of the ball in input space')
parser.add_argument('--inp_lamda', type=float, default=1, metavar='N',
                    help='Weight to the adversarial loss for input layer')
parser.add_argument('--reg', type=float, default=0, metavar='N',
                    help='weight reg')
parser.add_argument('--restore', type=int, default=1, metavar='N',
                    help='load model ')
parser.add_argument('--normal_class', type=int, default=0, metavar='N',
                    help='normal class nmber')
parser.add_argument('--optim', type=int, default=0, metavar='N',
                    help='0 : Adam 1: SGD')
#PARSER ARGUMENTS OVER
args = parser. parse_args()

# settings
#Checkpoint store path
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

def main():
    dataset = load_dataset("cifar10", "data", args.normal_class)
    train_loader, test_loader = dataset.loaders(batch_size=args.batch_size)
    model = CIFAR10_LeNet().to(device)
    model = nn.DataParallel(model)

    #Restore from checkpoint 
    if args.restore == 1:
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model_cifar.pt')))
        print("Saved Model Loaded")

    # Training the model
    train(args, model, device, train_loader, test_loader)

if __name__ == '__main__':
    main()