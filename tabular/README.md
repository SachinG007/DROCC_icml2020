# Deep Robust One-Class Classification 
In this directory we present examples of how to use the `DROCCTrainer` to replicate results in our [paper](https://arxiv.org/abs/2002.12718).


## Tabular Experiments
Data is expected in the following format:
```
train_data.npy: features of train data
test_data.npy: features of test data
train_labels.npy: labels for train data
test_labels.npy: labels for test data
```

### Arrhythmia and Thyroid
* Download the datasets from the ODDS Repository, [Arrhythmia](http://odds.cs.stonybrook.edu/arrhythmia-dataset/) and [Thyroid](http://odds.cs.stonybrook.edu/annthyroid-dataset/). 
* The data is divided for training as presented in previous works: [DAGMM](https://openreview.net/forum?id=BJJLHbb0-) and [GOAD](https://openreview.net/forum?id=H1lK_lBtvS).
* To generate the training and test data, use the `process_odds_dataset.py` script as follows and the output is generated in the current directory.
```
python process_odds_dataset.py -d <path/to/data/arrythmia.mat>
```
The directory with this data is referred to as "root_data" in the following sections.

### Abalone
* Download the dataset from the UCI Repository [here](http://archive.ics.uci.edu/ml/datasets/Abalone). This will consists of `abalone.data`. 
* To generate the training and test data, use the `process_abalone.py` script and the output is generated in the current directory.
```
python process_abalone.py -d <path/to/data/abalone.data>
```
The output path is referred to as "root_data" in the following section.

### Command to run experiments to reproduce results
#### Abalone 
```
python3 main.py --hd 128 --lr 0.001 --inp_lamda 1 --inp_radius 3 --batch_size 256 --epochs 200 --one_class_adv 1 --optim 0 --restore 0 -d "root_data"
```

#### Arrhythmia
```
python3 main.py --hd 128 --lr 0.0001 --inp_lamda 1 --inp_radius 16 --batch_size 256 --epochs 200 --one_class_adv 1 --optim 0 --restore 0 -d "root_data"
```

#### Thyroid
```
python3 main.py --hd 128 --lr 0.001 --inp_lamda 1 --inp_radius 2.5 --batch_size 256 --epochs 200 --one_class_adv 1 --optim 0 --restore 0 -d "root_data"
```
