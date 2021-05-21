import numpy as np
import torch
import torchvision.datasets
import torch.nn as nn
import pdb
import torchvision.transforms as transforms
import os

from mean_teacher import data, datasets
from mean_teacher.resnet import resnet50
from mean_teacher import architectures
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def read_dataset_initial(dataset, num):
    data_dict = {}
    #dataset = torchvision.datasets.cifar100
    for idx in range(len(dataset.imgs)):
        path, label = dataset.imgs[idx]
        if label in data_dict.keys():
            data_dict[label].append(idx)
        else:
            data_dict[label] = [idx]

    data_selected = np.array(list(data_dict.values()))
    #print(data_selected.shape)
    labeled_data = []
    for elem in data_selected:
        #ind = np.random.permutation(500)[0:num]
        labeled_data.append(np.array(elem[:num]))
    return np.concatenate(labeled_data)


def modify_dataset(dataset):
    for idx in range(len(dataset.imgs)):
        path, label_idx = dataset.imgs[idx]
        dataset.imgs[idx] = path, (label_idx, idx)


def pred_dataset(unlabeled_idxs, dataset, model):
    model.eval()
    sampler = SubsetRandomSampler(unlabeled_idxs)
    batch_sampler = BatchSampler(sampler, 256, drop_last=False)
    pred_loader = torch.utils.data.DataLoader(dataset,
                                                batch_sampler=batch_sampler,
                                                num_workers=8,
                                                pin_memory=True)

    top1_pred_list, traget_list, idx_list = [], [], []

    for i, (input, (target, idx)) in enumerate(pred_loader):
        input = input.cuda()
        #print(input.size())
        output, _, feats = model(input)
        #print(output, output.size())
        _, pred = torch.max(output.data.cpu(), dim=1)

        top1_pred_list.append(pred)
        traget_list.append(target)
        idx_list.append(idx)
    top1_pred_list = torch.cat(top1_pred_list, dim=0)
    traget_list = torch.cat(traget_list, dim=0)
    idx_list = torch.cat(idx_list, dim=0)

    return top1_pred_list, traget_list, idx_list


def one_bit_super_stage1(pred_list, target_list, idx_list):
    ind_sort_chosed = range(27000)

    ind_predTrue = np.where(pred_list[ind_sort_chosed] == target_list[ind_sort_chosed])[0]
    ind_predFalse = np.where(pred_list[ind_sort_chosed] != target_list[ind_sort_chosed])[0]

    pred_true = idx_list[ind_sort_chosed][ind_predTrue]
    pred_false = idx_list[ind_sort_chosed][ind_predFalse]
    false_pred = pred_list[ind_sort_chosed][ind_predFalse]

    return pred_true, pred_false, false_pred


def one_bit_super_stage2(pred_list, target_list, idx_list):
    ind_sort_chosed = np.random.permutation(len(pred_list))[0:20000]

    ind_predTrue = np.where(pred_list[ind_sort_chosed] == target_list[ind_sort_chosed])[0]
    ind_predFalse = np.where(pred_list[ind_sort_chosed] != target_list[ind_sort_chosed])[0]

    pred_true = idx_list[ind_sort_chosed][ind_predTrue]
    pred_false = idx_list[ind_sort_chosed][ind_predFalse]
    false_pred = pred_list[ind_sort_chosed][ind_predFalse]

    return pred_true, pred_false, false_pred


def read_model(checkpoint_path):
    #model = resnet50(num_classes=100)
    model_factory = architectures.__dict__['cifar_shakeshake26']
    model_params = dict(num_classes=100)
    model = model_factory(**model_params)
    model = nn.DataParallel(model).cuda()

    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    best_prec1 = checkpoint['best_prec1']
    print('best_acc', best_prec1)
    print('Epoch', checkpoint['epoch'])
    #pdb.set_trace()
    model.load_state_dict(checkpoint['ema_state_dict'])
    return model


def read_dataset():
    mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

    channel_stats = dict(mean=mean_pix,
                         std=std_pix)

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    traindir = 'mini-imagenet/train'
    #traindir = args.train_subdir
    dataset = torchvision.datasets.ImageFolder(traindir, eval_transformation)
    return dataset


def read_cifar100():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    traindir = 'data-local/bin/cifar100_train'
    dataset = torchvision.datasets.ImageFolder(traindir, eval_transformation)
    return dataset


def main():
    ### read the data
    dataset = read_cifar100()
    #dataset = read_dataset()

    labeled_idxs_initial = read_dataset_initial(dataset, 30)
    #labeled_idxs_initial = np.load('index_cifar100/3000initial/labeled_idxs_initial.npy')
    labeled_idxs_stage1 = np.load('/cache/index_cifar100/rand_select/pred_true_stage1.npy')
    labeled_idxs = np.concatenate((labeled_idxs_initial, labeled_idxs_stage1))
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))

    modify_dataset(dataset)
    ### read model
    checkpoint_path = '/cache/cifar100/checkpoints/best_stage1.ckpt'
    model = read_model(checkpoint_path)
    #pdb.set_trace()

    ### predict the dataset
    top1_pred_list, traget_list, idx_list = pred_dataset(unlabeled_idxs, dataset, model)

    pred_true, pred_false, false_pred = one_bit_super_stage2(top1_pred_list, traget_list, idx_list)
    #print(len(pred_true), len(pred_false), len(false_pred))

    #pdb.set_trace()
    os.makedirs('/cache/index_cifar100/rand_select', exist_ok=True)

    np.save('/cache/index_cifar100/rand_select/pred_true_stage2.npy', pred_true)
    np.save('/cache/index_cifar100/rand_select/pred_false_stage2.npy', pred_false)
    np.save('/cache/index_cifar100/rand_select/false_pred_stage2.npy', false_pred)
    print('Anotation done!')


if __name__ == '__main__':
    main()
