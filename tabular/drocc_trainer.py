import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

#trainer class for DROCC
class drocc_trainer():
    def __init__(self, model, optimizer, input_lamda, train_loader, val_loader, device):
        
        self.model = model
        self.optimizer = optimizer
        self.input_lamda = input_lamda
        self.device = device

    #LR SCHEDULER
    def adjust_learning_rate(self, epoch, total_epochs, only_ce_epochs, learning_rate):
        """decrease the learning rate"""
        #We dont want to consider the only ce 
        #based epochs for the lr scheduler
        epoch = epoch - only_ce_epochs
        drocc_epochs = total_epochs - only_ce_epochs
        lr = learning_rate
        if epoch <= drocc_epochs:
            lr = learning_rate * 0.001
        if epoch <= 0.90*drocc_epochs:
            lr = learning_rate * 0.01  
        if epoch <= 0.60*drocc_epochs:
            lr = learning_rate * 0.1  
        if epoch <= 0.30*drocc_epochs:
            lr = learning_rate    
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, lr, total_epochs, only_ce_epochs=50, train_loader, val_loader):
        for epoch in range(total_epochs): 
            #Make the weights trainable
            self.model.train()
            self.adjust_learning_rate(epoch, total_epochs, only_ce_epochs, learning_rate)
            
            #Placeholder for the respective 2 loss values
            epoch_adv_loss = torch.tensor([0]).type(torch.float32).detach()  #AdvLoss @ Input Layer
            epoch_ce_loss = 0  #Cross entropy Loss
            
            batch_idx = -1
            for data, target, _ in train_loader:
                batch_idx += 1
                data, target = data.to(self.device), target.to(self.device)
                # Data Processing
                data = data.to(torch.float)
                target = target.to(torch.float)
                target = torch.squeeze(target)

                self.optimizer.zero_grad()
                
                #Extract the logits for cross entropy loss
                logits = self.model(data)
                logits = torch.squeeze(logits, dim = 1)
                ce_loss = F.binary_cross_entropy_with_logits(logits, target)
                #Add to the epoch variable for printing average CE Loss
                epoch_ce_loss += ce_loss

                #Some Placeholders
                adv_loss_inp = 0  #AdvLoss @ Input Layer
                '''
                Adversarial Loss is calculated only for the positive data (label==1) points 
                We donot calculate adversarial loss for the negative points(label==0) if in case they are avaibale
                Hence the condition => torch.sum(target) == args.batch_size in the next line (Positive points have label = 1)
                '''
                if  epoch >= only_ce_epochs and torch.sum(target) == args.batch_size:
            
                    #AdvLoss in the input layer
                    adv_loss_inp, _ = self.one_class_adv_loss(model, data, target,
                                                        args.inp_radius, self.device, epoch, args.gamma)
                    epoch_adv_loss += adv_loss_inp

                    loss = ce_loss + adv_loss_inp*self.input_lamda
                else: 
                    #If only CE based training has to be done
                    loss = ce_loss
                
                #Backprop
                loss.backward()
                self.optimizer.step()
                    
            epoch_ce_loss = epoch_ce_loss/(batch_idx + 1)  #Average CE Loss
            epoch_adv_loss = epoch_adv_loss/(batch_idx + 1) #Average AdvLoss @Input Layer

            test_score = self.test(val_loader)
            
            print('Epoch: {}, CE Loss: {}, AdvLoss: {}, AUC/F1: {}'.format(epoch, epoch_ce_loss.item(), epoch_adv_loss.item(), test_score))

            if (epoch+1) % 10 == 0 and epoch > only_ce_epochs:
                print("TEST F-Score", max_auc)        

    def test(self, test_loader):
            
        label_score = []
        batch_idx = -1
        for data, target, _ in test_loader:
            batch_idx += 1
            data, target = data.to(self.device), target.to(self.device)
            data = data.to(torch.float)
            target = target.to(torch.float)
            target = torch.squeeze(target)

            logits = self.model(data)
            logits = torch.squeeze(logits, dim = 1)
            sigmoid_logits = torch.sigmoid(logits)
            scores = sigmoid_logits#.cpu().detach().numpy()
            label_score += list(zip(target.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
        # Compute AUC
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        thresh = np.percentile(scores, 20)
        y_pred = np.where(scores >= thresh, 1, 0)   
        #test_auc is F1 score here
        prec, recall, test_metric, _ = precision_recall_fscore_support(labels, y_pred, average="binary")
        # auc = roc_auc_score(labels, scores)
        return test_metric
        
    
    def one_class_adv_loss():
        '''
        #Calculate the adversarial loss
        1) Sample points randomly around the training data points
        2) Apply gradient descent with loss as CE with positive class 
        Repeat this step for N times
        3) Project the points between spheres of radius R and gamma*R
        4) Pass the calculated adversarial points through the model, and calculate the CE loss wrt target class 0
        '''
        def one_class_adv_loss(x_natural, target, radius, device,
                            epoch, gamma, step_size=.001, num_gradient_steps=50): 

            batch_size = len(x_natural)
            
            #Randomly sample points around the traininf data -> We will do SGD on these to find the adversarial points
            x_adv = torch.randn(x_natural.shape).to(device).detach().requires_grad_()
            x_adv_sampled = x_adv + x_natural

            for step in range(num_gradient_steps):
                with torch.enable_grad():

                    new_targets = torch.zeros(batch_size, 1).to(self.device)
                    new_targets = torch.squeeze(new_targets)
                    new_targets = new_targets.to(torch.float)
                    
                    logits = self.model(x_adv_sampled)         
                    logits = torch.squeeze(logits, dim = 1)
                    new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)

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

