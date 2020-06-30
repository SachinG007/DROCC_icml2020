# one-class for EPILEPSY 

## Installation
We use Python 3.6 to code. It needs packages mentioned in `requirements.txt` to run. Please ensure to use torchvision `0.2.1`
```
pip3 install virtualenv
virtualenv myenv
source myenv/bin/activate
pip3 install -r requirements.txt
```

## Data Processing
### Arrhythmia and Thyroid
* Download the datasets from the ODDS Repository [here](http://odds.cs.stonybrook.edu/arrhythmia-dataset/) and [here](http://odds.cs.stonybrook.edu/annthyroid-dataset/). 
* The data is divided for training as in previous works: [DAGMM](https://openreview.net/forum?id=BJJLHbb0-) and [GOAD](https://openreview.net/forum?id=H1lK_lBtvS).
* Generate the train and test files in the following format:
```
train_data.npy: features of train data
test_data.npy: features of test data
test_labels.npy: labels for test data
```
The directory with this data is referred to as "root_data" in the following sections.

### Abalone
* Download the dataset from the UCI Repository [here](http://archive.ics.uci.edu/ml/datasets/Abalone). This will consists of `abalone.data`. 
* To generate the training and test data, use the `process_dataset.py` script in the directory with the data. and the output is generated in the same directory.
```
python code/process_dataset.py
```
The output path is referred to as "root_data" in the following section.

## Command to run experiments to reproduce results
### Abalone 
```
python3 one_class_main_tabular.py --hd 128 --lr 0.001 --inp_lamda 1 --inp_radius 3 --batch_size 256 --epochs 200 --one_class_adv 1 --optim 0 --restore 0 -d "root_data"
```

### Arrhythmia
```
python3 one_class_main_tabular.py --hd 128 --lr 0.0001 --inp_lamda 1 --inp_radius 16 --batch_size 256 --epochs 200 --one_class_adv 1 --optim 0 --restore 0 -d "root_data"
```

### Thyroid
```
python3 one_class_main_tabular.py --hd 128 --lr 0.001 --inp_lamda 1 --inp_radius 2.5 --batch_size 256 --epochs 200 --one_class_adv 1 --optim 0 --restore 0 -d "root_data"
```

## Arguments Detail
inp_lamda => Weightage to the loss from adversarially sampled negative points (\mu in the paper)  
inp_radius => radius corresponding to the definition of Ni(r)   
hd => FC Hidden Dimension  
one_class_adv => Use the DROCC loss formulation or not (1:Use  0: Dont Use)  
optim => 0: Adam   1: SGD(M)  

