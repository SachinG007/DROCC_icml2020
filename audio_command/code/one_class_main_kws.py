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
from utils import *
from dataset import *
from models import LSTM_FC
from train_kws import *


#PARSER ARGUMENTS
torch.set_printoptions(precision=5)
parser = argparse.ArgumentParser(description='PyTorch Simple Training')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--hd', type=int, default=64, metavar='N',
                    help='Num hidden nodes')
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
parser.add_argument('--seed', type=int, default='0')
parser.add_argument('--restore', type=int, default=1, metavar='N',
                    help='load model ')
parser.add_argument('--optim', type=int, default=0, metavar='N',
                    help='0 : Adam 1: SGD')
parser.add_argument('-d', '--data_path', type=str, default='.')
#PARSER ARGUMENTS OVER
args = parser. parse_args()

# settings
#Checkpoint store path
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

def load_data(path):
    train_data = np.load(os.path.join(path, 'train.npy'), allow_pickle = True)
    train_lab = np.ones((train_data.shape[0]))
    # pdb.set_trace()
    test_data = np.load(os.path.join(path, 'test_data.npy'), allow_pickle = True)
    test_lab = np.load(os.path.join(path, 'test_labels.npy'), allow_pickle = True)
    test_data = np.asarray(test_data)
    test_lab = np.asarray(test_lab)
    ##preprocessing 
    mean=np.mean(train_data,0)
    std=np.std(train_data,0)
    train_data=(train_data-mean)/std

    test_data = (test_data - mean)/std
    train_samples = train_data.shape[0]
    test_samples = test_data.shape[0]
    timesteps = 98
    num_input = 32
    train_data = np.reshape(train_data, (train_samples, timesteps, num_input))
    test_data = np.reshape(test_data, (test_samples, timesteps, num_input))
    return train_data, train_lab, test_data, test_lab

def main():
    train_data, train_lab, test_data, test_lab = load_data(args.data_path)
    model = LSTM_FC(input_dim=32, num_classes=1, num_hidden_nodes=args.hd).to(device)
    #Restore from checkpoint 
    if args.restore == 1:
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model_marvin-nn-optim.pt')))
        print("Saved Model Loaded")
    # Training the model
    train(args, model, device,train_data, train_lab, test_data, test_lab)

if __name__ == '__main__':
    main()
