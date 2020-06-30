# one-class for TimeSeries

## Installation
We use Python 3.6 to code. It needs packages mentioned in `requirements.txt` to run. Please ensure to use torchvision `0.2.1`
```
pip3 install virtualenv
virtualenv myenv
source myenv/bin/activate
pip3 install -r requirements.txt
```

## Data Processing
### Audio-Keywords
* Download the Audio Commands dataset and generate MFCC features following [this](https://github.com/microsoft/EdgeML/tree/master/examples/pytorch/FastCells/KWS-training). Generate the features for a. the keyword and b. all classes except the keyword.
* Use the `code/process_dataset.py` script to generate the training and testing data. The directory containing the generated files is `root_data` in the following section.

### Epilepsy
* Download the dataset from the UCI Repository [here](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition). This will consists of a `data.csv` file. 
* To generate the training and test data, use the `code/process_dataset_epilepsy.py` script

```
python code/process_dataset_epilepsy.py -d <path to folder with data.csv> -o <output path>
```
The output path is referred to as "root_data" in the following section.

## Example Usage for Audio Commands Dataset
```
cd code/   
python3  one_class_main_kws.py --hd 512 --lr 0.001 --inp_lamda 1 --gamma 2 --step_size 0.1 --inp_radius 16 --batch_size 256 --epochs 30 --one_class_adv 1 --optim 0 --restore 0 -d "root_data"
```

## Example Usage for Epilepsy Dataset
```
cd code/   
python3  one_class_main_epilepsy.py --hd 128 --lr 0.00001 --inp_lamda 0.5 --gamma 2 --step_size 0.1 --inp_radius 10 --batch_size 256 --epochs 200 --one_class_adv 1 --optim 0 --restore 0 -d "root_data"
```

## Arguments Detail
inp_lamda => Weightage to the loss from adversarially sampled negative points (\mu in the paper)  
inp_radius => radius corresponding to the definition of Ni(r)  
hd => LSTM Hidden Dimension  
one_class_adv => Use the DROCC loss formulation or not (1:Use  0: Dont Use)  
optim => 0: Adam   1: SGD(M)  
step_size => step size for gradient ascent

