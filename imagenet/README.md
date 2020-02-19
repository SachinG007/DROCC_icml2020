# one-class for ImageNet

## Installation
We use Python 3.6 to code. It needs packages mentioned in `requirements.txt` to run. Please ensure to use torchvision `0.5.0`
```
pip3 install virtualenv
virtualenv myenv
source myenv/bin/activate
pip3 install -r requirements.txt
```

## Example Usage for CIFAR-10
```
cd code/ 
CUDA_VISIBLE_DEVICES="0,1,2,3"  python3 one_class_main_imagenet.py  --lr 0.001 --inp_lamda 1 --inp_radius 16 --batch_size 256 --ep 100 --one_class_adv 1 --optim 1  --restore 0 --normal_class 1 --data_path_train "root_imagenet_train_data" --data_path_test "root_imagenet_test_data"
```

## Arguments Detail
inp_lamda => Weightage to the loss from adversarially sampled negative points (\mu in the paper)
inp_radius => radius corresponding to the definition of Ni(r)
normal_class => Selects the normal class, rest all 9 classes are considered anomalous
one_class_adv => Use the DROCC loss formulation or not (1:Use  0: Dont Use)
optim => 0: Adam   1: SGD(M)
