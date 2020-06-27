import torch
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from utils import *
from sklearn.metrics import roc_auc_score
#The loss file for adversarial loss
from loss_cifar_pgd import one_class_adv_loss

max_auc = 0
max_auc_epoch = 0

#LR SCHEDULER
def adjust_learning_rate(optimizer, epoch, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch <= args.epochs:
        lr = args.lr * 0.01
    if epoch <= 0.80*args.epochs:
        lr = args.lr * 0.1 
    if epoch <= 0.40*args.epochs:
        lr = args.lr    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, 
          model, 
          device, 
          train_loader,
          test_loader):
    
    #Which Optimizer To use
    if args.optim == 1:
        optimizer = optim.SGD(model.parameters(),
                                  lr=args.lr,
                                  momentum=args.mom)
        print("using SGD")
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr)
        print("using Adam")

    #Some Placeholders for model analysis
    for epoch in range(args.epochs): 
        #Make the weights trainable
        model.train()
        adjust_learning_rate(optimizer,epoch, args)
        #lamda_1 is the weightage for adversarial loss at the input layer
        lamda_1 = args.inp_lamda

        #placeholder for total number of adversrial points detected on the surface of hyper-sphere
        total_adv_pts_inp = torch.tensor([0]).type(torch.float32).detach() #Adversarial Points detected in the input layer

        #Placeholder for the respective 2 loss values
        epoch_adv_loss_inp = torch.tensor([0]).type(torch.float32).detach()  #AdvLoss @ Input Layer
        epoch_ce_loss = 0  #Cross entropy Loss

        #Iterating over the batches
        batch_idx = -1
        for data, target, _ in train_loader:
            batch_idx += 1
            data, target = data.to(device), target.to(device)
            target = 1 - target #DataLoader labels normal class as 0 , our code considers it as 1

            optimizer.zero_grad()
            # Data Processing
            data = data.to(torch.float)
            target = target.to(torch.float)
            target = torch.squeeze(target)

            #Extract the logits for cross entropy loss
            logits = model(data)
            #CE Loss
            logits = torch.squeeze(logits, dim = 1)
            ce_loss = F.binary_cross_entropy_with_logits(logits, target)
            #Add to the epoch variable for printing average CE Loss
            epoch_ce_loss += ce_loss

            #Some Placeholders
            adv_loss_inp = 0  #AdvLoss @ Input Layer
            num_adv_pts_inp = 0  #Number of Adversarial Points actually detected on the surface of hyper-sphere for Input Layer
            '''
            Adversarial Loss is calculated only for the positive data points 
            We donot calculate adversarial loss for the negative points
            Hence the condition => torch.sum(target) == args.batch_size in the next line (Positive points have label = 1)
            '''
            # args.one_class_adv => If Adversarial training has to be performed
            # pdb.set_trace()
            if  args.one_class_adv and torch.sum(target) == args.batch_size:
        
                #AdvLoss in the input layer
                adv_loss_inp, num_adv_pts_inp = one_class_adv_loss(model, data, target,
                                                      args.inp_radius, device, epoch)
                epoch_adv_loss_inp += adv_loss_inp
                total_adv_pts_inp += num_adv_pts_inp

                #Total Loss
                loss = ce_loss + adv_loss_inp*lamda_1
            else: 
                #If only CE based training has to be done
                loss = ce_loss
            
            #Backprop
            loss.backward()
            optimizer.step()

        epoch_ce_loss = epoch_ce_loss/(batch_idx + 1)  #Average CE Loss
        epoch_adv_loss_inp = epoch_adv_loss_inp/(batch_idx + 1) #Average AdvLoss @Input Layer
        # pdb.set_trace()
        print("Epoch : ", str(epoch), " CE Loss : ", epoch_ce_loss.item(),
             " AdvLoss@Input : ", epoch_adv_loss_inp.item())

        test(args,model,device, test_loader, epoch)

        # if (epoch+1) % 10 == 0:
        #     torch.save(model.state_dict(),os.path.join(args.model_dir, 'model_cifar.pt'))
        #     print("TEST AUC", max_auc)

    f = open('cifar_' + str(args.normal_class) + '.txt','a+')
    f.write(" inp_radius " +str(args.inp_radius) + " inp_lamda " + str(args.inp_lamda) + " gamma " + str(args.gamma) +  " Optim " + str(args.optim) + " LR " + str(args.lr) +
    " MAX AUC " + str(max_auc) + " @epoch " + str(max_auc_epoch) + "\n") 
    f.close() 

    

def test(args, 
        model, 
        device, 
        test_loader,
        epoch):
        
    label_score = []
    test_epoch_ce_loss = 0
    # tot_correct = 0
    batch_idx = -1
    for data, target, _ in test_loader:
        batch_idx += 1
        target = 1 - target
        data, target = data.to(device), target.to(device)
        data = data.to(torch.float)
        target = target.to(torch.float)
        target = torch.squeeze(target)

        #Extract the logits for cross entropy loss
        logits = model(data)
        logits = torch.squeeze(logits, dim = 1)
        ce_loss = F.binary_cross_entropy_with_logits(logits, target)
        test_epoch_ce_loss += ce_loss
        sigmoid_logits = torch.sigmoid(logits)
        scores = logits[:]
        # Save triples of (label, score) in a list
        label_score += list(zip(target.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

    # Compute AUC
    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)
    print(test_auc)
    global max_auc
    global max_auc_epoch
    if test_auc > max_auc and epoch > args.epochs - 10:
        max_auc = test_auc
        max_auc_epoch = epoch