import torch
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from utils import *
import pdb
from sklearn.metrics import roc_auc_score
#The loss file for adversarial loss
from loss_lfoc import one_class_adv_loss
import numpy as np

only_ce_epochs = 100
max_close_test_auc_epoch = 0
check = 0
max_recall = 0
max_precision = 0
max_recall5 = 0
max_precision5 = 0

#Precision Recall Calculation
def pr(pos1, pos2, far, close, fpr):
    all_pos = pos1 #Since Positive in both far negatives and close negatives are same (pos1=pos2)
    all_neg = np.concatenate((far, close), axis = 0)
    num_neg = all_neg.shape[0]
    idx = int((1-fpr) * num_neg)
    all_neg.sort()
    thresh = all_neg[idx]
    tp = np.sum(all_pos > thresh)
    recall = tp/all_pos.shape[0]
    fp = int(fpr * num_neg)
    precision = tp/(tp+fp)
    return precision, recall

#LR SCHEDULER
def adjust_learning_rate(optimizer, epoch, args):
    """decrease the learning rate"""
    #We dont want to consider the only ce 
    #based epochs for the lr scheduler
    epoch = epoch - only_ce_epochs
    te = args.epochs - only_ce_epochs
    lr = args.lr
    if epoch <= te:
        lr = args.lr * 0.01
    if epoch <= 0.80*te:
        lr = args.lr * 0.1 
    if epoch <= 0.40*te:
        lr = args.lr   
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, 
          model, 
          device, 
          train_data, train_lab,
          test_data_norm, test_lab_norm,
          test_data_close, test_lab_close):
    
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
    train_data = torch.from_numpy(train_data)
    train_lab = torch.from_numpy(train_lab)
    for epoch in range(args.epochs): 
        #Make the weights trainable
        model.train()
        adjust_learning_rate(optimizer,epoch, args)
        #lamda_2 is the weightage for adversarial loss at the hidden layer
        lamda_2 = args.inp_lamda
        #placeholder for total number of adversrial points detected on the surface of hyper-sphere
        total_adv_pts_mid = torch.tensor([0]).type(torch.float32).detach() #Adv_points detected in the hidden layer
        #Placeholder for the respective 2 loss values
        epoch_adv_loss_mid = torch.tensor([0]).type(torch.float32).detach()  #AdvLoss @ Hidden Layer
        epoch_ce_loss = 0  #Cross entropy Loss
        #Iterating over the batches
        num_iters = int(train_data.shape[0]/args.batch_size)
        batch_idx = -1
        tot_batches = 6
        for j in range(num_iters):
            batch_idx += 1
            data=train_data[j*args.batch_size:(j+1)*args.batch_size]
            target=train_lab[j*args.batch_size:(j+1)*args.batch_size]            
            data, target = data.to(device), target.to(device)

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

            #Extract the hidden dimension features (for hidden layer adversrial loss)
            mid_feats = model.half_forward_start(data)  #The data has been passed only through the first layer of the Network
            #Some Placeholders
            adv_loss_mid = 0  #AdvLoss @ Hidden Layer
            num_adv_pts_mid = 0  #Number of Adversarial Points actually detected on the surface of hyper-sphere for Hidden Layer
            '''
            Adversarial Loss is calculated only for the positive data points 
            We donot calculate adversarial loss for the negative points
            Hence the condition => torch.sum(target) == args.batch_size in the next line (Positive points have label = 1)
            '''
            # args.one_class_adv => If Adversarial training has to be performed
            if  epoch > only_ce_epochs and args.one_class_adv and torch.sum(target) == args.batch_size:
                # print("Batch idx ", batch_idx, " adv ")
                #AdvLoss in the hidden layer
                grad_list = grad_analyzer(args, model, device, data, target)
        
                adv_loss_mid, num_adv_pts_mid = one_class_adv_loss(model, mid_feats, target,
                                                    args.inp_radius ,device, epoch, grad_list, hidden_adv = 1)
                epoch_adv_loss_mid += adv_loss_mid
                total_adv_pts_mid += num_adv_pts_mid

                loss = ce_loss + adv_loss_mid*lamda_2

            else: 
                #If only CE based training has to be done
                loss = ce_loss
            
            #Backprop
            loss.backward(retain_graph = True)
            optimizer.step()
                
        epoch_ce_loss = epoch_ce_loss/(batch_idx + 1)  #Average CE Loss
        epoch_adv_loss_mid = epoch_adv_loss_mid/(batch_idx + 1) #Average AdvLoss @Hidden Layer
        print("Epoch : ", str(epoch), " CE Loss : ", epoch_ce_loss.item(),
             " AdvLoss@Hidden : ", epoch_adv_loss_mid.item())

        if args.one_class_adv:
            print(" NumDetectedAdvPoints@Hidden ", total_adv_pts_mid.item())

        norm_test_auc, pos1, far = test(args,model,device, test_data_norm, test_lab_norm, epoch)
        close_test_auc, pos2, close = test(args,model,device, test_data_close, test_lab_close, epoch)
        global max_close_test_auc_epoch, max_recall, max_precision, max_recall5, max_precision5
        log_precision, log_recall = pr(pos1, pos2, far, close, 0.03)
        log_precision5, log_recall5 = pr(pos1, pos2, far, close, 0.05)
        if epoch > only_ce_epochs and log_recall > max_recall:
            max_recall = log_recall
            max_recall5 = log_recall5
            max_precision = log_precision
            max_precision5 = log_precision5
            max_close_test_auc_epoch = epoch

        if (epoch+1) % 5 == 0 and epoch > only_ce_epochs:
            torch.save(model.state_dict(),os.path.join(args.model_dir, 'model_lfoc.pt'))
            print("Precision FPR0.03", max_precision, "Recall FPR0.03", max_recall)
            print("Precision FPR0.05", max_precision5, "Recall FPR0.05", max_recall5)
    #All epochs done

def test(args, 
        model, 
        device, 
        test_data, test_lab,
        epoch):
        
    label_score = []
    outputs = []
    test_data = torch.from_numpy(test_data)
    test_lab = torch.from_numpy(test_lab)

    data=test_data
    target=test_lab           
    data, target = data.cuda(), target.cuda()
    data = data.to(torch.float)
    target = target.to(torch.float)
    target = torch.squeeze(target)
    logits = model(data)
    logits = torch.squeeze(logits, dim = 1)
    sigmoid_logits = torch.sigmoid(logits)
    scores = sigmoid_logits#.cpu().detach().numpy()
    label_score += list(zip(target.cpu().data.numpy().tolist(),
                                    scores.cpu().data.numpy().tolist()))

    # Compute AUC
    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    pos_scores = scores[labels==1]
    neg_scores = scores[labels==0]
    test_auc = roc_auc_score(labels, scores)
    # print("TEST AUC", test_auc)
    
    return test_auc, pos_scores, neg_scores

def grad_analyzer(args, 
        model, 
        device, 
        data, target):

    total_train_pts = len(data)
    data = data.to(torch.float)
    target = target.to(torch.float)
    target = torch.squeeze(target)

    #Extract the logits for cross entropy loss
    g_data = data
    g_data = g_data.detach().requires_grad_()
    # logits = model(g_data)
    hid_state = model.half_forward_start(g_data)
    logits = model.half_forward_end(hid_state)
    logits = torch.squeeze(logits, dim = 1)
    ce_loss = F.binary_cross_entropy_with_logits(logits, target)
    
    grad = torch.autograd.grad(ce_loss, hid_state)[0]

    return grad
