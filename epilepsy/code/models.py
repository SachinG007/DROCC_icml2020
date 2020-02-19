import torch
import torch.nn.functional as F
import torch.nn as nn
from base.base_net import BaseNet
import numpy as np 
from collections import OrderedDict
import pdb;


class TwoLayer(nn.Module):
    def __init__(self,
                 input_dim=2,
                 num_classes=1, 
                 num_hidden_nodes=20
    ):

        super(TwoLayer, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_nodes = num_hidden_nodes
        
        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(self.input_dim, self.num_hidden_nodes)),
            ('relu1', activ)]))
        self.size_final = self.num_hidden_nodes

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.size_final, self.num_classes))]))
        # self.lamda = nn.Parameter(0 * torch.ones([1, 1]))
        # self.inp_lamda = nn.Parameter(0 * torch.ones([1, 1]))

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, self.size_final))
        return logits
        
    def half_forward_start(self, input):
        return self.feature_extractor(input)

    def half_forward_end(self, input):
        return self.classifier(input.view(-1, self.size_final))


class FourLayer(nn.Module):
    def __init__(self,
                 input_dim=2,
                 num_classes=1, 
                 num_hidden_nodes=2
    ):

        super(FourLayer, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_nodes = num_hidden_nodes
        
        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.input_dim, self.num_hidden_nodes)),
            ('relu1', activ),
            ('fc2', nn.Linear(self.num_hidden_nodes, int(self.num_hidden_nodes/2))),
            ('relu2', activ),
            ]))
        self.size_final = int(self.num_hidden_nodes/2)

        self.classifier = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(int(self.num_hidden_nodes/2), int(self.num_hidden_nodes/4))),
            ('relu3', activ),
            ('fc4', nn.Linear(int(self.num_hidden_nodes/4), self.num_classes))]))

        # self.lamda = nn.Parameter(0 * torch.ones([1, 1]))
        # self.inp_lamda = nn.Parameter(0 * torch.ones([1, 1]))


    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, self.size_final))
        return logits
        
    def half_forward_start(self, input):
        return self.feature_extractor(input)

    def half_forward_end(self, input):
        return self.classifier(input.view(-1, self.size_final))

class SixLayer(nn.Module):
    def __init__(self,
                 input_dim=2,
                 num_classes=1, 
                 num_hidden_nodes=512
    ):

        super(SixLayer, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_nodes = num_hidden_nodes
        
        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.input_dim, self.num_hidden_nodes)),
            ('relu1', activ),
            ('fc2', nn.Linear(self.num_hidden_nodes, int(self.num_hidden_nodes/2))),
            ('relu2', activ),
            ('fc3', nn.Linear(int(self.num_hidden_nodes/2), int(self.num_hidden_nodes/4))),
            ('relu3', activ),
            ]))
        self.size_final = int(self.num_hidden_nodes/4)

        self.classifier = nn.Sequential(OrderedDict([
            ('fc4', nn.Linear(int(self.num_hidden_nodes/4), int(self.num_hidden_nodes/8))),
            ('relu4', activ),
            ('fc5', nn.Linear(int(self.num_hidden_nodes/8), int(self.num_hidden_nodes/8))),
            ('relu5', activ),
            ('fc6', nn.Linear(int(self.num_hidden_nodes/8), self.num_classes))]))

        # self.lamda = nn.Parameter(0 * torch.ones([1, 1]))
        # self.inp_lamda = nn.Parameter(0 * torch.ones([1, 1]))


    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, self.size_final))
        return logits
        
    def half_forward_start(self, input):
        return self.feature_extractor(input)

    def half_forward_end(self, input):
        return self.classifier(input.view(-1, self.size_final))


class MNIST_LeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.rep_dim = 64
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, 1, bias=False)

    def forward(self, x):
        x = x.view(x.shape[0],1,28,28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    # def half_forward_start(self, input):
    #     x = x.view(x.shape[0],1,28,28)
    #     x = self.conv1(input)
    #     x = self.pool(F.leaky_relu(self.bn1(x)))
    #     x = self.conv2(x)
    #     x = self.pool(F.leaky_relu(self.bn2(x)))
    #     x = x.view(x.size(0), -1)
    #     return x

    # def half_forward_end(self, input):
    #     x = self.fc1(input)
    #     x = self.fc2(x)
    #     return x

class CIFAR10_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, int(self.rep_dim/2), bias=False)
        self.fc3 = nn.Linear(int(self.rep_dim/2), 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def half_forward_start(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        return x

    def half_forward_end(self, x):
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ThreeLayer_hitachi(nn.Module):
    def __init__(self,
                 input_dim=320,
                 num_classes=1, 
                 num_hidden_nodes=64
    ):

        super(ThreeLayer_hitachi, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_nodes = num_hidden_nodes
        
        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.input_dim, self.num_hidden_nodes)),
            ('relu1', activ),
            ('fc2', nn.Linear(self.num_hidden_nodes, int(self.num_hidden_nodes/2))),
            ('relu2', activ),
            ]))
        self.size_final = int(self.num_hidden_nodes/2)

        self.classifier = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(self.size_final, self.num_classes))]))

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, self.size_final))
        return logits

class LSTM_FC(nn.Module):
    def __init__(self,
                 input_dim=32,
                 num_classes=1, 
                 num_hidden_nodes=8
    ):

        super(LSTM_FC, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_nodes = num_hidden_nodes
        self.encoder = nn.LSTM(input_size=self.input_dim, hidden_size=self.num_hidden_nodes,
                                num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.num_hidden_nodes, self.num_classes)
        activ = nn.ReLU(True)

    def forward(self, input):
        features = self.encoder(input)[0][:,-1,:]
        # pdb.set_trace()
        logits = self.fc(features)
        return logits

    def half_forward_start(self, x):
        features = self.encoder(x)[0][:,-1,:]
        return features

    def half_forward_end(self, x):
        logits = self.fc(x)
        return logits