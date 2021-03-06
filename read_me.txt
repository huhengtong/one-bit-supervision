##############################################################
One-bit Supervision for Image Classification
##############################################################
This code is the official implementation of "One-bit Supervision for Image Classification"

######################################
Requirements:
  Python 3; Pytorch 1.0.0

######################################
Training:
  To train the models on CIFAR100, run these commands:
    For stage 0: python main_stage0.py --train-subdir trainset_path  --test-subdir testset_path  --arch 'cifar_shakeshake26' --labeled-batch-size 50  -b 512 --epochs 180  --lr 0.2  --lr-rampdown-epochs 210 --nesterov 'true'  --ema-decay 0.97  --dataset cifar100  --consistency 1000  --consistency-rampup 5  --logit-distance-cost 0.01
    For stage 1: python main_stage1.py --train-subdir trainset_path  --test-subdir testset_path  --arch 'cifar_shakeshake26' --labeled-batch-size 200  -b 512  --epochs 180  --lr 0.2  --lr-rampdown-epochs 210 --nesterov 'true'  --ema-decay 0.97  --dataset cifar100  --consistency 1000  --consistency-rampup 5  --logit-distance-cost 0.01
    For stage 2: python main_stage2.py --train-subdir trainset_path  --test-subdir testset_path  --arch 'cifar_shakeshake26' --labeled-batch-size 320  -b 512  --epochs 180  --lr 0.2  --lr-rampdown-epochs 210 --nesterov 'true'  --ema-decay 0.97  --dataset cifar100  --consistency 1000  --consistency-rampup 5  --logit-distance-cost 0.01

######################################
Evaluation:
  To evaluate my model on CIFAR100, run:
    python main_stage2.py --arch 'cifar_shakeshake26'  --evaluate pretrained_model_path

######################################
Pre-trained Models:
  Be limited to the size of the supplementary material, the pre-trained models are not included.

######################################
Datasets:
  The three datasets we used, namely, CIFAR100, Mini-Imagenet, and Imagenet, are all publicly available.
  
######################################
Results:

#############  Top 1 Accuracy
CIFAR100       0.7376
Mini-Imagenet  0.4554
Imagenet       0.6040

##############################################################