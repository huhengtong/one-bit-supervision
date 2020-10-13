import re
import argparse
import os
import shutil
import time
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import pdb

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from mean_teacher.resnet import resnet50
from mean_teacher.functions_initial import *

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0


def main(context):
    global global_step
    global best_prec1

    dirpath = args.checkpoint_path
    os.makedirs(dirpath, exist_ok=True)

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)

    def create_model(ema=False):
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.arch))
        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
        model = model_factory(**model_params)

        model = nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        validate(eval_loader, model)
        LOG.info("Evaluating the EMA model:")
        validate(eval_loader, ema_model)
        return

    for epoch in range(args.start_epoch, args.epochs):
        print('EPOCH:', epoch)
        start_time = time.time()
        # train for one epoch
        train(train_loader, model, ema_model, optimizer, epoch)

        LOG.info("Evaluating the primary model:")
        prec1 = validate(eval_loader, model)
        print('accuracy', prec1)
        LOG.info("Evaluating the EMA model:")
        ema_prec1 = validate(eval_loader, ema_model)
        print('ema_accuracy', ema_prec1)
        # LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
        is_best = ema_prec1 > best_prec1
        best_prec1 = max(ema_prec1, best_prec1)

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, dirpath, epoch + 1)


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
        labeled_data.append(np.array(elem[:num]))
    return np.concatenate(labeled_data)


def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)


def create_data_loaders(train_transformation,
                        eval_transformation,
                        args):
    traindir = args.train_subdir
    testdir = args.test_subdir

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    #dir_path = '/cache/index_cifar100/rand_select/'
    labeled_idxs = read_dataset_initial(dataset, 30)

    labeled_idxs, unlabeled_idxs = data.relabel_dataset_initial(dataset, labeled_idxs)

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
        batch_size=200,
        shuffle=False,
        num_workers=2 * args.workers,
        pin_memory=True,
        drop_last=False)

    return train_loader, test_loader


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, ema_model, optimizer, epoch):
    global global_step

    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    # switch to train mode
    model.train()
    ema_model.train()

    for i, ((input, ema_input), target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        input_var = input.cuda()
        with torch.no_grad():
            ema_input_var = ema_input.cuda()
        target_var = target.cuda()

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        ema_logit, _, _ = ema_model(ema_input_var)
        class_logit, cons_logit, _ = model(input_var)

        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)

        if args.logit_distance_cost >= 0:
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
        else:
            res_loss = 0

        class_loss = class_criterion(class_logit, target_var) / minibatch_size

        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
        else:
            consistency_loss = 0

        loss = class_loss + consistency_loss + res_loss
        if loss.item() > 1e5:
            print(class_loss.item(), consistency_loss.item(), res_loss.item())
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)


def validate(eval_loader, model):
    meters = AverageMeterSet()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    iter_num, prec = 0, 0
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)
        with torch.no_grad():
            input_var = input.cuda()
            target_var = target

        # minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        # compute output
        output1, output2, _ = model(input_var)
        prec += accuracy(output1.data.cpu(), target_var.data)
        iter_num += 1

    return prec / iter_num


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.ckpt'
    # print(dirpath, filename)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best_initial.ckpt')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target):
    _, prediction = torch.max(output, dim=1)
    acc = np.sum(prediction.numpy() == target.numpy()) / len(target)
    return acc


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))

