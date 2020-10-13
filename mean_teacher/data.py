"""Functions to load data from folders and augment it"""

import itertools
import logging
import os.path
import pdb
import torch

from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler


LOG = logging.getLogger('main')
NO_LABEL = -1




class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def relabel_dataset_stage1(dataset, labeled_idxs, false_pred_dict):
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


def relabel_dataset_stage2(dataset, labeled_idxs, false_pred_dict1, false_pred_dict2):
    for idx in range(len(dataset.imgs)):
        false_target = np.ones(100)
        path, label_idx = dataset.imgs[idx]
        if idx in labeled_idxs:
            if idx in false_pred_dict1.keys():
                false_target[false_pred_dict1[idx]] = 0
            dataset.imgs[idx] = path, (label_idx, false_target)
        elif idx in false_pred_dict1.keys():
            #print(idx)
            false_target[false_pred_dict1[idx]] = 0
            if idx in false_pred_dict2.keys():
                false_target[false_pred_dict2[idx]] = 0
                # print(false_target)
                # pdb.set_trace()
            dataset.imgs[idx] = path, (NO_LABEL, false_target)
        elif idx in false_pred_dict2.keys():
            false_target[false_pred_dict2[idx]] = 0
            dataset.imgs[idx] = path, (NO_LABEL, false_target)
        else:
            dataset.imgs[idx] = path, (NO_LABEL, false_target)
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))

    return labeled_idxs, unlabeled_idxs


def relabel_dataset_initial(dataset, labeled_idxs):
    for idx in range(len(dataset.imgs)):
        path, label_idx = dataset.imgs[idx]

        if idx in labeled_idxs:
            dataset.imgs[idx] = path, label_idx
        else:
            dataset.imgs[idx] = path, NO_LABEL
    unlabeled_idxs = sorted(set(range(len(dataset.imgs))) - set(labeled_idxs))

    return labeled_idxs, unlabeled_idxs



class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
