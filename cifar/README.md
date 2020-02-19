# one-class for CIFAR 

## Installation
We use Python 3.6 to code. It needs packages mentioned in `requirements.txt` to run. Please ensure to use torchvision `0.2.1`
```
pip3 install virtualenv
virtualenv myenv
source myenv/bin/activate
pip3 install -r requirements.txt
```

## Example Usage for CIFAR-10
```
cd code/ 
python3  one_class_main_cifar.py  --inp_lamda 1  --inp_radius 8 --lr 0.001 --batch_size 256 --epochs 40  --one_class_adv 1 --optim 0 --restore 0 --normal_class 0    
```

## Arguments Detail
inp_lamda => Weightage to the loss from adversarially sampled negative points (\mu in the paper)
inp_radius => radius corresponding to the definition of Ni(r)
normal_class => Selects the normal class, rest all 9 classes are considered anomalous
one_class_adv => Use the DROCC loss formulation or not (1:Use  0: Dont Use)
optim => 0: Adam   1: SGD(M)

## Jupyter Notebooks
We have also added sample jupyter notebooks

## Citations
We have used the CIFAR dataloaders for One Class Classification Task from https://github.com/lukasruff/Deep-SVDD-PyTorch