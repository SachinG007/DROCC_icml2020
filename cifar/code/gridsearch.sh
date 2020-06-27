#!/bin/bash
for norm_class in 4 6 7 8
do
for inp_rad in 8 12 16 20 24 28 32 36 40 44
do
for lr in 0.01 0.001
do
    CUDA_VISIBLE_DEVICES="0"  python3  one_class_main_cifar.py  --inp_lamda 0.5  --inp_radius $inp_rad --optim 1 --lr $lr --batch_size 256 --epochs 50 --seed 10  --one_class_adv 1 --restore 0 --normal_class $norm_class &
    CUDA_VISIBLE_DEVICES="1"  python3  one_class_main_cifar.py  --inp_lamda 0.5  --inp_radius $inp_rad --optim 0 --lr $lr --batch_size 256 --epochs 50 --seed 10  --one_class_adv 1 --restore 0 --normal_class $norm_class &
    CUDA_VISIBLE_DEVICES="2"  python3  one_class_main_cifar.py  --inp_lamda 1.0  --inp_radius $inp_rad --optim 1 --lr $lr --batch_size 256 --epochs 50 --seed 10  --one_class_adv 1 --restore 0 --normal_class $norm_class &
    CUDA_VISIBLE_DEVICES="3"  python3  one_class_main_cifar.py  --inp_lamda 1.0  --inp_radius $inp_rad --optim 0 --lr $lr --batch_size 256 --epochs 50 --seed 10  --one_class_adv 1 --restore 0 --normal_class $norm_class
   # echo $(bc <<< "$adv_lam_sum + $r1")
done
done
done