from __future__ import print_function
import os
from random import sample
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pdb
from utils import *
from dataset import *
from models import *
from train_lfoc import *

class Args:
    def __init__(self):
        self.batch_size = 128
        self.epochs = 150
        self.hd = 64
        self.lr = 0.005
        self.mom = 0.0
        self.model_dir = './log/model'
        self.one_class_adv = 1
        self.inp_radius = 1.5
        self.inp_lamda = 0.05 #mu in paper
        self.reg = 0
        self.restore = 0
        self.seed = 0
        self.optim = 0
        self.data_path = '.'

args=Args()

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
    path="/mnt/lfoc/follow-fpr/"
    train_data = np.load(os.path.join(path, 'train_data.npy'))
    train_lab = np.load(os.path.join(path, 'train_labels.npy'))
    test_data_norm = np.load(os.path.join(path, 'test_others_data.npy'))
    test_lab_norm = np.load(os.path.join(path, 'test_others_labels.npy'))
    test_data_close =  np.load(os.path.join(path, 'test_cn_data.npy'))
    test_lab_close =  np.load(os.path.join(path, 'test_cn_labels.npy'))
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
    train_data = np.reshape(train_data, (train_samples, timesteps, num_input))
    test_data_norm = np.reshape(test_data_norm, (test_samples_norm, timesteps, num_input))
    test_data_close = np.reshape(test_data_close, (test_samples_close, timesteps, num_input))

    return train_data, train_lab, test_data_norm, test_lab_norm, test_data_close, test_lab_close

def main():

    train_data, train_lab, test_data_norm, test_lab_norm, test_data_close, test_lab_close = load_data(args.data_path)
    model = LSTM_FC(input_dim=32, num_classes=1, num_hidden_nodes=args.hd).to(device)
    model = nn.DataParallel(model)
    #Restore from checkpoint 
    if args.restore == 1:
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model_lfoc.pt')))
        print("Saved Model Loaded")
    # Training the model
    train(args, model, device, train_data, train_lab, test_data_norm, test_lab_norm, test_data_close, test_lab_close)

if __name__ == '__main__':
    main()  
