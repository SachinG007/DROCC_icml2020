import numpy as np
import matplotlib.pyplot as plt
from dataset import CustomDataset
import torch
import pdb
import plotly
import plotly.graph_objs as go
LARGE = 10000
"""
Visualizing a two dimensional dataset
"""
def visualize_dataset(X, y):
    colormap = np.array(['r', 'b', 'g'])
    y = y.astype(int)
    plt.scatter(X[:, 0], X[:, 1], c=colormap[np.ravel(y)])
    return plt

def visualize_dataset_3D(data_matrix, y):
    ax = plt.axes(projection='3d')
    trace = go.Scatter3d(
        x=data_matrix[:, 0],#[1, 2, 3],  # <-- Put your data instead
        y=data_matrix[:, 1],#[4, 5, 6],  # <-- Put your data instead
        z=data_matrix[:, 2],#[7, 8, 9],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 5,
            'opacity': 0.8,
            'color':y.flatten(),
        }
    )

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    data = [trace]
    plot_figure = go.Figure(data=data, layout=layout)
    # Render the plot.
    plotly.offline.iplot(plot_figure)
    plotly.offline.iplot(plot_figure, filename='preds.html')
    print("saved plt")


def visualize_classifier(model, device, test_loader, **kwargs):

    #plot the old data as green
    f = 1
    Fs = 1020#samples will  remain consta nt
    sample = 1020
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs)
    wave = wave / 2
    x = (x - sample/2)/sample
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    data_matrix = np.concatenate((x, wave), axis = 1)
    neg_points = np.array([[-0.25, 0], [0.25, 0],[0, 0.25], [0, -0.25]])
    data_matrix = np.concatenate((data_matrix, neg_points), axis = 0)
    #mark the old data green
    y = np.ones((sample+4,1)) * 2
    y[sample:sample + 4] = np.ones((4,1)) * 2

    plt = visualize_dataset(data_matrix, y)


    data_matrix = np.random.rand(1024*3,2) * 1 - 0.5 
    # data_matrix = np.random.rand(1024,2) * 10 - 5 
    y = np.ones((1024*3,1))
    #vis dataset
    # colormap = np.array(['r', 'b'])
    # y = y.astype(int)
    # plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=colormap[np.ravel(y)])
    # plt.show()
    # plt.savefig("data_vis.png")
    # plt.close()

    # import pdb;pdb.set_trace()
    train_dataset = CustomDataset(data_matrix, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False)

    preds = []
    #data_matrix = []
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            data = data.to(torch.float)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            pred = pred.cpu().numpy()
            preds.append(pred)
            #data_matrix.append(data.cpu().numpy())

    preds = np.asarray(preds)
    preds = np.ravel(preds)
    #data_matrix = np.asarray(data_matrix)
    #data_matrix = np.reshape(data_matrix, [-1, 2])
    plt = visualize_dataset(data_matrix, preds)

    return plt, data_matrix, preds

def visualize_classifier_2D(args, model, device, test_loader, mean, std, **kwargs):


    sample = 1024*20
    data_matrix = np.random.rand(sample,2) * 4 - 2
    # data_matrix[:,2] = data_matrix[:,2] + 3
    y = np.ones((sample* 2,1))

    # import pdb;pdb.set_trace()
    data_matrix_n = (data_matrix - mean)/std
    train_dataset = CustomDataset(data_matrix_n, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False)

    preds = []
    #data_matrix = []
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            data = data.to(torch.float)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            pred = pred.cpu().numpy()
            preds.append(pred)
            #data_matrix.append(data.cpu().numpy())

    preds = np.asarray(preds)

    f = 1
    Fs = 1024#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs)
    wave = wave
    x = (x - sample/2)/(sample/2)
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    data_matrix_orig = np.concatenate((x, wave), axis = 1)
    data_matrix = np.concatenate((data_matrix, data_matrix_orig), axis = 0)

    orig_labs = np.ones((1024,)) * 2
    preds = np.ravel(preds)
    preds = np.concatenate((preds, orig_labs), axis = 0)

    y = preds.astype(int)
    # print(np.shape(data_matrix))
    colormap = np.array(['r', 'b', 'y'])
    plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=colormap[np.ravel(y)])
    plt.show()
    # plt.savefig('/mnt/one_class_results/results_6D/data_matrix_dist' + str(args.dist_lambda)+ " inp_radius " +str(args.inp_radius) +  " hid_radius " +str(args.hidden_radius) +
    #         " inp_lamda " +str(args.inp_lamda) + " hidden_lamda " +str(args.hidden_lamda) + " Optim " + str(args.optim) + " hd " + str(args.hd)+ '.png')
    plt.savefig('vis.png')
    plt.close()
    #data_matrix = np.asarray(data_matrix) 
    #data_matrix = np.reshape(data_matrix, [-1, 2])
    # np.save('/mnt/one_class_results/results_2D/data_matrix_dist' + str(args.dist_lambda)+ '_opt_' + str(args.optim) + '_hd_rad_' + str(args.hidden_radius) + '_hd_' + str(args.hd) + '.npy', data_matrix)
    # np.save('/mnt/one_class_results/results_2D/preds_dist' + str(args.dist_lambda)+ '_opt_' + str(args.optim) + '_hd_rad_' + str(args.hidden_radius) + '_hd_' + str(args.hd) + '.npy', preds)
    # visualize_dataset_3D(data_matrix, preds)

    # return plt, data_matrix, preds    


def visualize_classifier_3D(args, model, device, test_loader, mean, std, **kwargs):


    sample = 1024*2*5
    data_matrix = np.random.rand(sample,3) * 4 - 2
    # data_matrix[:,2] = data_matrix[:,2] + 3
    y = np.ones((sample* 2,1))

    # import pdb;pdb.set_trace()
    data_matrix_n = (data_matrix - mean)/std
    train_dataset = CustomDataset(data_matrix_n, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False)

    preds = []
    #data_matrix = []
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            data = data.to(torch.float)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            pred = pred.cpu().numpy()
            preds.append(pred)
            #data_matrix.append(data.cpu().numpy())

    preds = np.asarray(preds)

    f = 1
    Fs = 1024#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs)
    wave = wave
    x = (x - sample/2)/(sample/2)
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    z = np.random.rand(sample,1) * 0
    data_matrix_orig = np.concatenate((x, wave, z), axis = 1)
    data_matrix = np.concatenate((data_matrix, data_matrix_orig), axis = 0)

    orig_labs = np.ones((1024,)) * 2
    preds = np.ravel(preds)
    preds = np.concatenate((preds, orig_labs), axis = 0)
    #data_matrix = np.asarray(data_matrix) 
    #data_matrix = np.reshape(data_matrix, [-1, 2])
    np.save('results/data_matrix_dist' + str(args.dist_lambda)+ '_opt_' + str(args.optim) + '_radius_' + str(args.hidden_radius) + '_hd_' + str(args.hd) + '.npy', data_matrix)
    np.save('results/preds_dist' + str(args.dist_lambda)+ '_opt_' + str(args.optim) + '_radius_' + str(args.hidden_radius) + '_hd_' + str(args.hd) + '.npy', preds)
    # visualize_dataset_3D(data_matrix, preds)

    # return plt, data_matrix, preds    

def visualize_classifier_nD(args, model, device, test_loader, mean, std, **kwargs):


    sample = 1024*20
    data_matrix1 = np.random.rand(sample,2) * 4 - 2
    data_matrix2 = np.random.rand(sample,args.noise_dim) * 2 - 1
    data_matrix = np.concatenate((data_matrix1, data_matrix2), axis = 1)
    # data_matrix[:,2] = data_matrix[:,2] + 3
    y = np.ones((sample,1))

    # import pdb;pdb.set_trace()
    data_matrix_n = (data_matrix - mean)/std
    train_dataset = CustomDataset(data_matrix_n, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False)

    preds = []
    #data_matrix = []
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            data = data.to(torch.float)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            pred = pred.cpu().numpy()
            preds.append(pred)
            #data_matrix.append(data.cpu().numpy())

    preds = np.asarray(preds)

    preds = np.ravel(preds)
    #data_matrix = np.asarray(data_matrix) 
    data_matrix = data_matrix[:,0:2]
    # import pdb;pdb.set_trace()

    f = 1
    Fs = 1024#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs)
    wave = wave
    x = (x - sample/2)/(sample/2)
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    data_matrix_orig = np.concatenate((x, wave), axis = 1)
    data_matrix = np.concatenate((data_matrix, data_matrix_orig), axis = 0)
    orig_labs = np.ones((1024,)) * 2
    preds = np.concatenate((preds, orig_labs), axis = 0)
    # print(num)
    #data_matrix = np.reshape(data_matrix, [-1, 2])
    y = preds.astype(int)
    # print(np.shape(data_matrix))
    colormap = np.array(['r', 'b', 'y'])
    plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=colormap[np.ravel(y)])
    plt.show()
    plt.savefig('/mnt/one_class_results/results_6D/data_matrix_dist' + str(args.dist_lambda)+ " inp_radius " +str(args.inp_radius) +  " hid_radius " +str(args.hidden_radius) +
            " inp_lamda " +str(args.inp_lamda) + " hidden_lamda " +str(args.hidden_lamda) + " Optim " + str(args.optim) + " hd " + str(args.hd)+ '.png')
    # plt.savefig('vis_nD.png')
    plt.close()
    # np.save('/mnt/one_class_results/results_4D/data_matrix_dist' + str(args.dist_lambda)+ " inp_radius " +str(args.inp_radius) +  " hid_radius " +str(args.hidden_radius) +
    #         " inp_lamda " +str(args.inp_lamda) + " hidden_lamda " +str(args.hidden_lamda) + " Optim " + str(args.optim) + " hd " + str(args.hd)+ '.npy', data_matrix)
    # np.save('/mnt/one_class_results/results_4D/preds_dist' + str(args.dist_lambda)+ " inp_radius " +str(args.inp_radius) +  " hid_radius " +str(args.hidden_radius) +
    #         " inp_lamda " +str(args.inp_lamda) + " hidden_lamda " +str(args.hidden_lamda) + " Optim " + str(args.optim) + " hd " + str(args.hd)+ '.npy', preds)

def visualize_classifier_nD_neg(args, model, device, test_loader, mean, std, **kwargs):


    sample = 1024*5
    data_matrix1 = np.random.rand(sample,2) * 4 - 2
    data_matrix2 = np.random.rand(sample,args.noise_dim) * 2 - 1
    data_matrix = np.concatenate((data_matrix1, data_matrix2), axis = 1)
    # data_matrix[:,2] = data_matrix[:,2] + 3
    y = np.ones((sample,1))

    # import pdb;pdb.set_trace()
    data_matrix_n = (data_matrix - mean)/std
    train_dataset = CustomDataset(data_matrix_n, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False)

    preds = []
    #data_matrix = []
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            data = data.to(torch.float)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            pred = pred.cpu().numpy()
            preds.append(pred)
            #data_matrix.append(data.cpu().numpy())

    preds = np.asarray(preds)

    preds = np.ravel(preds)
    #data_matrix = np.asarray(data_matrix) 
    data_matrix = data_matrix[:,0:2]
    # import pdb;pdb.set_trace()

    f = 1
    Fs = 1024#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs)
    wave = wave
    x = (x - sample/2)/(sample/2)
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    data_matrix_orig = np.concatenate((x, wave), axis = 1)
    data_matrix = np.concatenate((data_matrix, data_matrix_orig), axis = 0)
    orig_labs = np.ones((1024,)) * 2
    preds = np.concatenate((preds, orig_labs), axis = 0)
    # print(num)
    #data_matrix = np.reshape(data_matrix, [-1, 2])
    y = preds.astype(int)
    # print(np.shape(data_matrix))
    colormap = np.array(['r', 'b', 'y'])
    plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=colormap[np.ravel(y)])
    plt.show()
    plt.savefig('/mnt/one_class_results/results_6D_neg/data_matrix_dist' + str(args.dist_lambda)+ " inp_radius " +str(args.inp_radius) +  " hid_radius " +str(args.hidden_radius) +
            " inp_lamda " +str(args.inp_lamda) + " hidden_lamda " +str(args.hidden_lamda) + " Optim " + str(args.optim) + " hd " + str(args.hd)+ '.png')
    plt.close()
    # np.save('/mnt/one_class_results/results_4D_neg/data_matrix_dist' + str(args.dist_lambda)+ " inp_radius " +str(args.inp_radius) +  " hid_radius " +str(args.hidden_radius) +
    #         " inp_lamda " +str(args.inp_lamda) + " hidden_lamda " +str(args.hidden_lamda) + " Optim " + str(args.optim) + " hd " + str(args.hd)+ '.npy', data_matrix)
    # np.save('/mnt/one_class_results/results_4D_neg/preds_dist' + str(args.dist_lambda)+ " inp_radius " +str(args.inp_radius) +  " hid_radius " +str(args.hidden_radius) +
    #         " inp_lamda " +str(args.inp_lamda) + " hidden_lamda " +str(args.hidden_lamda) + " Optim " + str(args.optim) + " hd " + str(args.hd)+ '.npy', preds)

def test_acc_calc(args, model, device, test_loader, mean, std, **kwargs):

    f = 1
    Fs = 1024#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs) * 0.9
    wave = wave
    x = (x - sample/2)/(sample/2)
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    z = np.random.rand(sample,args.noise_dim) * 2 - 1
    data_matrix = np.concatenate((x, wave, z), axis = 1)
    y = np.ones((1024,1))
    # import pdb;pdb.set_trace()
    data_matrix_n = (data_matrix - mean)/std
    train_dataset = CustomDataset(data_matrix_n, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False)

    preds = []
    #data_matrix = []
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            data = data.to(torch.float)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            pred = pred.cpu().numpy()
            preds.append(pred)
            #data_matrix.append(data.cpu().numpy())

    preds = np.asarray(preds)
    preds = np.reshape(preds,(1024,1))
    target = np.ones((1024,1))
    # import pdb;pdb.set_trace()
    correct = np.sum(preds==target)
    print("test acc : " , correct)
    return correct

def test_acc_calc_outz(args, model, device, test_loader, mean, std, **kwargs):


    f = 1
    Fs = 1024#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs) * 0.9
    wave = wave
    x = (x - sample/2)/(sample/2)
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    z = np.random.rand(sample,args.noise_dim) + 1
    data_matrix = np.concatenate((x, wave, z), axis = 1)
    y = np.ones((1024,1))
    # import pdb;pdb.set_trace()
    data_matrix_n = (data_matrix - mean)/std
    train_dataset = CustomDataset(data_matrix_n, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False)

    preds = []
    #data_matrix = []
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            data = data.to(torch.float)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            pred = pred.cpu().numpy()
            preds.append(pred)
            #data_matrix.append(data.cpu().numpy())

    preds = np.asarray(preds)
    preds = np.reshape(preds,(1024,1))
    target = np.ones((1024,1))
    # import pdb;pdb.set_trace()
    correct = np.sum(preds==target)
    print("out z test acc : " , correct)

    return correct

def test_acc_calc_very_outz(args, model, device, test_loader, mean, std, **kwargs):


    f = 1
    Fs = 1024#samples will  remain consta nt
    sample = Fs
    x = np.arange(sample)
    wave = np.sin(2 * np.pi * f * x / Fs) * 0.9
    wave = wave
    x = (x - sample/2)/(sample/2)
    x = np.reshape(x,(sample,1))
    wave = np.reshape(wave,(sample,1))
    z = np.random.rand(sample,args.noise_dim) + 2
    data_matrix = np.concatenate((x, wave, z), axis = 1)
    y = np.ones((1024,1))
    # import pdb;pdb.set_trace()
    data_matrix_n = (data_matrix - mean)/std
    train_dataset = CustomDataset(data_matrix_n, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False)

    preds = []
    #data_matrix = []
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            data = data.to(torch.float)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            pred = pred.cpu().numpy()
            preds.append(pred)
            #data_matrix.append(data.cpu().numpy())

    preds = np.asarray(preds)
    preds = np.reshape(preds,(1024,1))
    target = np.ones((1024,1))
    # import pdb;pdb.set_trace()
    correct = np.sum(preds==target)
    print("very out z test acc : " , correct)

    return correct


#Visulaize sphere classifier
def visualize_classifier_sphere(args, model, device, train_loader, mean, std):
    sample_p = 128 * 16 * 10
    data_matrix = np.random.rand(sample_p,3) * 6 - 3
    y = np.ones((sample_p,1))
    data_matrix = (data_matrix - mean)/std
    train_dataset = CustomDataset(data_matrix, y)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False)

    preds = []
    #data_matrix = []
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            data = data.to(torch.float)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            pred = pred.cpu().numpy()
            preds.append(pred)
            #data_matrix.append(data.cpu().numpy())
    preds = np.asarray(preds)
    preds = np.ravel(preds)
    np.save('/mnt/one_class_results/sphere/data_matrix_dist' + " inp_radius " +str(args.inp_radius) +  " hid_radius " +str(args.hidden_radius) +
            " inp_lamda " +str(args.inp_lamda) + " hidden_lamda " +str(args.hidden_lamda) + " Optim " + str(args.optim) + " hd " + str(args.hd)+ '.npy', data_matrix)
    np.save('/mnt/one_class_results/sphere/preds_dist' + " inp_radius " +str(args.inp_radius) +  " hid_radius " +str(args.hidden_radius) +
            " inp_lamda " +str(args.inp_lamda) + " hidden_lamda " +str(args.hidden_lamda) + " Optim " + str(args.optim) + " hd " + str(args.hd)+ '.npy', preds)