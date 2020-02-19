import torch 
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 
from utils import *
# import pdb
import advertorch
from optim_solver import optim_solver
'''
#Calculate the adversarial loss
1) Sample points randomly around the training data points
2) Apply gradient descent with loss as CE with positive class 
   and a distance loss term ( so that the points are near the trainin data)
   Repeat this step for N times
3) Project the points on the sphere of radius R
4) Pass the calculated adversarial points through the model, and calculate the CE loss wrt target class 0
'''
def one_class_adv_loss(model, 
                       x_natural, 
                       target, 
                       radius,
                       device,
                       epoch,
                       grad_list,
                       hidden_adv = 0,
                       step_size=.01, 
                       num_gradient_steps=100): 
    #Model has to be only used for evaluation here, no weight updates
    # model.eval()
    batch_size = len(x_natural)
    
    #Randomly sample points around the traininf data -> We will do SGD on these to find the adversarial points
    x_adv = torch.randn(x_natural.shape).to(device).detach().requires_grad_()
    x_adv_sampled = x_adv + x_natural

    for step in range(num_gradient_steps):
        with torch.enable_grad():
            #Targets for Adversarial points - Positive in our case since we want to sample points 
            #near manifold classified positive
            new_targets = torch.zeros(batch_size, 1).to(device)
            new_targets = torch.squeeze(new_targets)
            new_targets = new_targets.to(torch.float)
            
            #If the points correspond to the hidden layer, pass them through the later half of the network only
            logits = model.module.half_forward_end(x_adv_sampled)         
            logits = torch.squeeze(logits, dim = 1)
            new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)  

            grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
            # model.eval()
            grad_flattened = grad.view(grad.shape[0],-1)
            grad_norm = torch.norm(grad_flattened, p=2, dim = 1)
            for u in range(grad.ndim - 1):
                grad_norm = torch.unsqueeze(grad_norm, dim = u+1)
            if grad.ndim==2:
                grad_norm = grad_norm.repeat(1, grad.shape[1])
            if grad.ndim==3:
                grad_norm = grad_norm.repeat(1, grad.shape[1], grad.shape[2])
            grad_normalized = grad / grad_norm 
        with torch.no_grad():
            # x_adv = x_adv.detach() + step_size*grad_normalized
            x_adv_sampled.add_(step_size*grad_normalized)

        if  (step+1) % 2 == 0 :
            eta_x_adv = x_adv_sampled - x_natural
            #THE PROJECTION OPERATOR CODE
            eta_x_adv = optim_solver(grad_list, eta_x_adv, radius, device)

            eta_x_adv_flattened = eta_x_adv.view(eta_x_adv.shape[0], -1)
            norm = torch.norm(eta_x_adv_flattened, p = 2, dim = 1)
            for u in range(eta_x_adv.ndim - 1):
                norm = torch.unsqueeze(norm, dim = u+1)
            if eta_x_adv.ndim == 2:
                norm_eta = norm.repeat(1,eta_x_adv.shape[1])
            if eta_x_adv.ndim == 3:
                norm_eta = norm.repeat(1,eta_x_adv.shape[1],eta_x_adv.shape[2])

            eta_x_adv = eta_x_adv * 1 *radius / norm_eta
            # eta_x_adv = eta_x_adv*grad_list
            # pdb.set_trace()
            x_adv_sampled = x_natural + eta_x_adv  #These adv_points are now on the surface of hyper-sphere

    #Pass the adv_points throught the network and calculate the loss
    adv_pred = model.module.half_forward_end(x_adv_sampled)

    adv_pred = torch.squeeze(adv_pred, dim = 1)
    new_targets = torch.squeeze(new_targets)
    adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets * 0))

    adv_pos = torch.sigmoid(adv_pred) > 0.5
    num_pos = torch.sum(adv_pos)
    num_adv_pts = num_pos

    return adv_loss, num_adv_pts
