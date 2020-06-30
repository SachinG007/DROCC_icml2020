import numpy as np 
import torch
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

def check_mhlnbs(grad, diff, radius, device, gamma):
    mhlnbs_dis = torch.sqrt(torch.sum(grad*diff**2, dim=1))
    # print(mhlnbs_dis)
    lamda = torch.zeros((grad.shape[0],1))
    lamda[mhlnbs_dis < radius] = 1
    lamda[mhlnbs_dis > (gamma * radius)] = 2
    # sum_right = (torch.ones((grad.shape[0],1)) * radius**2).to(device)
    # cond_a = sum_left > sum_right.squeeze(1)
    # lamda = torch.ones((grad.shape[0],1))
    # lamda[cond_a == True] = 0
    return lamda, mhlnbs_dis

def cal_left1(lam, grad, diff, radius, device):
    n1 = lam**2 * grad**2
    d1 = (1 + lam * grad)**2 + 1e-10
    term = diff**2 * n1/d1
    term_sum = torch.sum(term)
    return term_sum

def cal_left2(nu, grad, diff, radius, device, gamma):
    n1 = grad**2
    d1 = (nu + grad)**2 + 1e-10
    term = diff**2 * n1/d1
    term_sum = torch.sum(term)
    return term_sum

def check_right1(lam, grad, diff, radius, device):
    n1 = grad
    d1 = (1 + lam * grad)**2 + 1e-10
    term = diff**2 * n1/d1
    term_sum = torch.sum(term)
    if term_sum > radius**2:
        return cal_left1(lam, grad, diff, radius, device)
    else:
        return np.inf

def check_right2(nu, grad, diff, radius, device, gamma):
    n1 = grad*nu**2
    d1 = (nu + grad)**2 + 1e-10
    term = diff**2 * n1/d1
    term_sum = torch.sum(term)
    if term_sum < (gamma*radius)**2:
        return cal_left2(nu, grad, diff, radius, device, gamma)
    else:
        # return torch.tensor(float('inf'))
        return np.inf

def range_lamda(grad):
    lam, _ = torch.max(grad, dim=1)
    eps, _ = torch.min(grad, dim=1)
    lam = -1 / lam + eps*0.0001
    return lam

def range_nu(grad, mhlnbs_dis, radius, gamma):
    alpha = (gamma*radius)/mhlnbs_dis
    max_sigma, _ = torch.max(grad, dim=1)
    nu = (alpha/(1-alpha))*max_sigma
    return nu

def optim_solver( grad, diff, radius, device, gamma=2):
    # print('optim solver')
    # grad = grad.detach().cpu().numpy()
    # diff = diff.detach().cpu().numpy()
    # grad = normalizer(grad)
    
    lamda, mhlnbs_dis = check_mhlnbs(grad, diff, radius, device, gamma)

    temp_lamda = range_lamda(grad).detach().cpu().numpy()
    temp_nu = range_nu(grad, mhlnbs_dis, radius, gamma).detach().cpu().numpy()
    final_lamda =  torch.zeros((grad.shape[0],1))
    
    for idx in range(lamda.shape[0]):
        if lamda[idx] == 1:
            min_left = np.inf
            best_lam = 0
            for k in range(40):
                val = np.random.uniform(low = temp_lamda[idx], high = 0)
                left_val = check_right1(val, grad[idx], diff[idx], radius, device)
                if left_val < min_left:
                    min_left = left_val
                    best_lam = val
            
            final_lamda[idx] = best_lam
        
        if lamda[idx] == 2:
            min_left = np.inf
            best_lam = np.inf
            for k in range(40):
                val = np.random.uniform(low = 0, high = temp_nu[idx])
                left_val = check_right2(val, grad[idx], diff[idx], radius, device, gamma)
                if left_val < min_left:
                    min_left = left_val
                    best_lam = val
            
            final_lamda[idx] = 1.0/best_lam       

        else:
            final_lamda[idx] = 0
    # print(np.count_nonzero(final_lamda))
    final_lamda = final_lamda.to(device)
    # pdb.set_trace()
    for j in range(diff.shape[0]):
        diff[j,:] = diff[j,:]/(1+final_lamda[j]*grad[j,:])

    return diff, final_lamda
