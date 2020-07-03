import torch 
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 

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
                       step_size=.001, 
                       num_gradient_steps=50): 
    #Model has to be only used for evaluation here, no weight updates
    model.eval()
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
            
            logits = model(x_adv_sampled)         
            logits = torch.squeeze(logits, dim = 1)
            new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)

            # model.train()
            grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
            # model.eval()
            grad_flattened = torch.reshape(grad, (grad.shape[0],-1))
            grad_norm = torch.norm(grad_flattened, p=2, dim = 1)
            for u in range(grad.ndim - 1):
                grad_norm = torch.unsqueeze(grad_norm, dim = u+1)
            if grad.ndim==2:
                grad_norm = grad_norm.repeat(1, grad.shape[1])
            if grad.ndim==3:
                grad_norm = grad_norm.repeat(1, grad.shape[1], grad.shape[2])
            grad_normalized = grad / grad_norm 
        with torch.no_grad():
            x_adv_sampled.add_(step_size*grad_normalized)

        if (step+1)%10==0:
            #Take the sampled adversarial points to the surface of hyper-sphere
            h = x_adv_sampled - x_natural
            norm_h = torch.sqrt(torch.sum(h**2, dim=tuple(range(1, h.dim()))))
            # compute alpha in function of the value of the norm of h (by batch)
            alpha = torch.clamp(norm_h, radius, gamma * radius).to(device)
            # make use of broadcast to project h
            proj = (alpha / norm_h).view(-1, *[1]*(h.dim()-1))
            h = proj * h

            x_adv_sampled = x_natural + h  #These adv_points are now on the surface of hyper-sphere


    adv_pred = model(x_adv_sampled)
    adv_pred = torch.squeeze(adv_pred, dim = 1)
    new_targets = torch.squeeze(new_targets)
    adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets * 0))
    adv_pos = torch.sigmoid(adv_pred) > 0.5
    num_pos = torch.sum(adv_pos)
    num_adv_pts = num_pos

    return adv_loss, num_adv_pts