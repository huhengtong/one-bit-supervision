import numpy as np
import torchvision
import torch
import pdb

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from mean_teacher import data
from mean_teacher.utils import *


NO_LABEL = -1


def create_data_loaders_onestage(train_transformation,
                        eval_transformation,
                        args):
    traindir = 'mini-imagenet/train'
    testdir = 'mini-imagenet/test'
    # traindir = 'data-local/bin/cifar100_train'
    # testdir = 'data-local/bin/cifar100_test'

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    labeled_idxs_initial = np.load('index_miniimagenet/labeled_idxs_256.npy')
    labeled_idxs_stage1 = np.load('index_miniimagenet/one_stage/pred_true.npy')

    pred_false_stage1 = np.load('index_miniimagenet/one_stage/pred_false.npy')
    false_pred_stage1 = np.load('index_miniimagenet/one_stage/false_pred.npy')

    # labeled_idxs_initial = np.load('index_cifar100/3000initial/labeled_idxs_initial.npy')
    # labeled_idxs_stage1 = np.load('index_cifar100/3000initial/pred_true_onestage.npy')
    #
    # pred_false_stage1 = np.load('index_cifar100/3000initial/pred_false_onestage.npy')
    # false_pred_stage1 = np.load('index_cifar100/3000initial/false_pred_onestage.npy')

    labeled_idxs = np.concatenate((labeled_idxs_initial, labeled_idxs_stage1))

    false_pred_dict = dict(zip(pred_false_stage1, false_pred_stage1))

    labeled_idxs, unlabeled_idxs = relabel_dataset_onestage(dataset, labeled_idxs, false_pred_dict)
    print(len(labeled_idxs), len(unlabeled_idxs))
    #pdb.set_trace()

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(testdir, eval_transformation),
        batch_size=100,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, test_loader


def relabel_dataset_onestage(dataset, labeled_idxs, false_pred_dict):
    for idx in range(len(dataset.imgs)):
        false_target = np.ones(100)
        path, label_idx = dataset.imgs[idx]
        if idx in labeled_idxs:
            dataset.imgs[idx] = path, (label_idx, false_target)
        elif idx in false_pred_dict.keys():
            false_target[false_pred_dict[idx]] = 0
            dataset.imgs[idx] = path, (NO_LABEL, false_target)
        else:
            dataset.imgs[idx] = path, (NO_LABEL, false_target)
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))

    return labeled_idxs, unlabeled_idxs
