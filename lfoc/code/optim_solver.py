import numpy as np 
import torch
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

def check_a(grad, diff, radius, device):
    sum_left = torch.sum(grad*diff**2, dim=1)
    sum_right = (torch.ones((grad.shape[0],1)) * radius**2).to(device)
    cond_a = sum_left > sum_right.squeeze(1)
    lamda = torch.ones((grad.shape[0],1))
    lamda[cond_a == True] = 0
    return lamda

def cal_left(lam, grad, diff, radius, device):
    n1 = lam**2 * grad**2
    d1 = (1 + lam * grad)**2 + 1e-10
    term = diff**2 * n1/d1
    term_sum = torch.sum(term)
    return term_sum

def check_right(lam, grad, diff, radius, device):
    n1 = lam**2 * grad**3
    d1 = (1 + lam * grad)**2 + 1e-10
    term = diff**2 * n1/d1
    term_sum = torch.sum(term)
    if term_sum > radius**2:
        return cal_left(lam, grad, diff, radius, device)
    else:
        return torch.tensor(float('inf'))

def range_lamda(grad):
    lam, _ = torch.max(grad, dim=1)
    eps, _ = torch.min(grad, dim=1)
    lam = -1 / lam + eps*0.0001
    return lam

def normalizer(grad):
    grad_norm = torch.sum(torch.abs(grad), dim=1)
    grad_norm = torch.unsqueeze(grad_norm, axis = 1)
    # grad_norm = np.repeat(grad_norm, grad.shape[1], axis = 1)
    grad_norm = grad_norm.repeat(1, grad.shape[1])
    grad = grad/grad_norm * grad.shape[1]
    return grad

def optim_solver( grad, diff, radius, device):
    # grad = grad.detach().cpu().numpy()
    # diff = diff.detach().cpu().numpy()
    grad = normalizer(grad)
    
    lamda = check_a(grad, diff, radius, device)

    temp_lamda = range_lamda(grad).detach().cpu().numpy()
    final_lamda =  torch.zeros((grad.shape[0],1))
    
    for idx in range(temp_lamda.shape[0]):
        if lamda[idx] != 0:
            min_left = np.inf
            for k in range(10):
                val = np.random.uniform(low = temp_lamda[idx], high = 0)
                left_val = check_right(val, grad, diff, radius, device)
                if left_val < min_left:
                    min_left = left_val
                    best_lam = val
            
            final_lamda[idx] = best_lam
        else:
            final_lamda[idx] = 0
    # print(np.count_nonzero(final_lamda))
    final_lamda = final_lamda.to(device)
    # pdb.set_trace()
    for j in range(diff.shape[0]):
        diff[j,:] = diff[j,:]/(1+final_lamda[j]*grad[j,:])

    return diff
