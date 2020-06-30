from __future__ import print_function
import os
from random import sample
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pdb
from utils import *
from dataset import *
from models import *
from train_lfoc import *

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
parser.add_argument('--mom', type=float, default=0.9, metavar='M',
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
parser.add_argument('--gamma', type=float, default=1.0, metavar='N',
                    help='r to gamma * r projection')
parser.add_argument('--step_size', type=float, default=1.0, metavar='N',
                    help='step size')
parser.add_argument('--seed', type=int, default='0')
parser.add_argument('--optim', type=int, default=0, metavar='N',
                    help='0 : Adam 1: SGD')
parser.add_argument('-d', '--data_path', type=str, default='.')
#PARSER ARGUMENTS OVER
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# settings
#Checkpoint store path
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

def load_data(path, args):
    train_data = np.load(os.path.join(path, 'train_data.npy'))
    train_lab = np.load(os.path.join(path, 'train_labels.npy'))
    print(train_data.shape)
    train_data = train_data[:4500]    
    train_lab = train_lab[:4500]
    # import pdb;pdb.set_trace()
    # train_data = train_data[1680-args.batch_size:1680 + 10 * args.batch_size]
    # train_lab = train_lab[1680-args.batch_size:1680 + 10 * args.batch_size]
    test_data_norm = np.load(os.path.join(path, 'test_others_data.npy'))
    test_lab_norm = np.load(os.path.join(path, 'test_others_labels.npy'))
    test_data_close =  np.load(os.path.join(path, 'test_cn_data.npy'))
    test_lab_close =  np.load(os.path.join(path, 'test_cn_labels.npy'))
    print("Train Data: ", train_data.shape)
    print("Normal test Data: ", test_data_norm.shape)
    print("Close test Data: ", test_data_close.shape)
    ##preprocessing 
    mean=np.mean(train_data,0)
    #pdb.set_trace()
    std=np.std(train_data,0)
    std[std[:]<0.00001]=1
    train_data=(train_data-mean)/std
    test_data_norm=(test_data_norm-mean)/std
    test_data_close=(test_data_close-mean)/std
    train_samples = train_data.shape[0]
    test_samples_norm = test_data_norm.shape[0]
    test_samples_close = test_data_close.shape[0]
    timesteps = 98
    num_input = 32
    train_data = np.reshape(train_data, (train_samples, num_input, timesteps))
    test_data_norm = np.reshape(test_data_norm, (test_samples_norm, num_input, timesteps))
    test_data_close = np.reshape(test_data_close, (test_samples_close, num_input, timesteps))
    train_data = np.transpose(train_data, (0,2,1))
    test_data_norm = np.transpose(test_data_norm, (0,2,1))
    test_data_close = np.transpose(test_data_close, (0,2,1))
    return train_data, train_lab, test_data_norm, test_lab_norm, test_data_close, test_lab_close

def main():

    train_data, train_lab, test_data_norm, test_lab_norm, test_data_close, test_lab_close = load_data(args.data_path, args)
    model = LSTM_FC(input_dim=32, num_classes=1, num_hidden_nodes=args.hd).to(device)
    #Restore from checkpoint 
    if args.restore == 1:
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model_lfoc.pt')))
        print("Saved Model Loaded")
    # Training the model
    train(args, model, device, train_data, train_lab, test_data_norm, test_lab_norm, test_data_close, test_lab_close)

if __name__ == '__main__':
    main()
