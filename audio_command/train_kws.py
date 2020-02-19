import torch
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from utils import *
from sklearn.metrics import roc_auc_score
from loss_kws import one_class_adv_loss
import numpy as np

only_ce_epochs = 100
max_auc = 0
max_auc_epoch = 0

#LR SCHEDULER
def adjust_learning_rate(optimizer, epoch, args):
    """decrease the learning rate"""
    #We dont want to consider the only ce 
    #based epochs for the lr scheduler
    epoch = epoch - only_ce_epochs
    te = args.epochs - only_ce_epochs
    lr = args.lr
    if epoch <= te:
        lr = args.lr * 0.001
    if epoch <= 0.90*te:
        lr = args.lr * 0.01  
    if epoch <= 0.60*te:
        lr = args.lr * 0.1  
    if epoch <= 0.30*te:
        lr = args.lr    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, 
          model, 
          device, 
          train_data, train_lab,
          test_data, test_lab):
    
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
        #lamda_1 is the weightage for adversarial loss at the input layer
        lamda_1 = args.inp_lamda
        total_adv_pts_inp = torch.tensor([0]).type(torch.float32).detach() #Adversarial Points detected in the input layer
        #Placeholder for the respective 2 loss values
        epoch_adv_loss_inp = torch.tensor([0]).type(torch.float32).detach()  #AdvLoss @ Input Layer
        epoch_ce_loss = 0  #Cross entropy Loss
        #Iterating over the batches
        num_iters = int(train_data.shape[0]/args.batch_size)
        batch_idx = -1
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
            if  epoch > only_ce_epochs and args.one_class_adv and torch.sum(target) == args.batch_size:
        
                #AdvLoss in the input layer
                adv_loss_inp, num_adv_pts_inp = one_class_adv_loss(model, data, target,
                                                    args.inp_radius,device, epoch)
                epoch_adv_loss_inp += adv_loss_inp
                total_adv_pts_inp += num_adv_pts_inp

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

        if args.one_class_adv:
            print(" NumDetectedAdvPoints@Input ", total_adv_pts_inp.item())
        #Visualize the classifier performance every 10th epoch

        test(args,model,device, test_data, test_lab, epoch)
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(),os.path.join(args.model_dir, 'model_epilepsy.pt'))
            print("TEST AUC", max_auc)

    #All epochs done
    

def test(args, 
        model, 
        device, 
        test_data, test_lab,
        epoch):
        
    label_score = []
    outputs = []
    # tot_correct = 0
    num_iters = int(test_data.shape[0]/args.batch_size)
    batch_idx = -1
    test_data = torch.from_numpy(test_data)
    test_lab = torch.from_numpy(test_lab)
    for j in range(num_iters):
        batch_idx += 1
        data=test_data[j*args.batch_size:(j+1)*args.batch_size]
        target=test_lab[j*args.batch_size:(j+1)*args.batch_size]            
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
    # scores[np.abs(scores) > 10000] = 10000
    test_auc = roc_auc_score(labels, scores)
    global max_auc
    global max_auc_epoch
    if epoch > only_ce_epochs and test_auc > max_auc:
        max_auc = test_auc
        max_auc_epoch = epoch
    # print("MAX TEST AUC", max_auc, " @epoch ", max_auc_epoch)
    # if epoch % 10 == 0:
    #     print(str(np.mean(scores[labels==1])) + " +/- " + str(np.std(scores[labels==1])))
    #     print(str(np.mean(scores[labels==0])) + " +/- " + str(np.std(scores[labels==0])))

