# one-class for LFOC 

## Installation
We use Python 3.6 to code. It needs packages mentioned in `requirements.txt` to run. Please ensure to use torchvision `0.2.1`
```
pip3 install virtualenv
virtualenv myenv
source myenv/bin/activate
pip3 install -r requirements.txt
```

## DataProcessing

## Example Usage for LFOC Dataset
```
cd code/   
python3  one_class_main_HAR.py --hd 80 --lr 0.01 --inp_lamda 1 --inp_radius 25 --batch_size 256 --epochs 100 --one_class_adv 1 --optim 0 --restore 0 -d "root_data"
```

## Arguments Detail
inp_lamda => Weightage to the loss from adversarially sampled negative points (\mu in the paper)
inp_radius => radius corresponding to the definition of Ni(r)
hd => LSTM Hidden Dimension
one_class_adv => Use the DROCC loss formulation or not (1:Use  0: Dont Use)
optim => 0: Adam   1: SGD(M)

