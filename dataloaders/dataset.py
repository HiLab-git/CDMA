import argparse
import itertools
import os
from random import seed, shuffle
import random
import time
import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
join = os.path.join
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import monai
import torch.optim as optim
from monai.data import decollate_batch, PILReader
import logging
import csv
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    SpatialPadd,
    RandSpatialCropd,
    RandRotate90d,
    Resized,
    ScaleIntensityd,
    RandAxisFlipd,  
    EnsureTyped,
)
from monai.transforms.intensity.dictionary import BrightnessContrastd


def get_train_loader(args, train_files, labeled_idxs, unlabeled_idxs):
    train_transforms = Compose(
        [
            # image three channels (H, W, 3); label: (H, W)
            LoadImaged(keys=["img", "label"],
                       reader=PILReader, dtype=np.uint8),
            # BrightnessContrastd(keys=["img"], allow_missing_keys=True),
            # label: (1, H, W)
            AddChanneld(keys=["label"], allow_missing_keys=True),
            AsChannelFirstd(keys=['img'], channel_dim=-1,
                            allow_missing_keys=True),  # image: (3, H, W)
            # Do not scale label
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            Resized(keys=["img", "label"], spatial_size=(
                args.input_size, args.input_size)),
            RandAxisFlipd(keys=["img", "label"], prob=0.5),
            RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
            EnsureTyped(keys=["img", "label"]),
        ]
    )
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # sampler
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=False
    )
    return train_loader

def get_val_loader(args, val_files, batch_size=None):
    if batch_size:
        bs = batch_size
    else:
        bs = args.batch_size
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
            AddChanneld(keys=["label"], allow_missing_keys=True),
            AsChannelFirstd(keys=['img'], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            Resized(keys=["img","label"], spatial_size=(args.input_size,args.input_size)),
            EnsureTyped(keys=["img", "label"]),
        ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=args.num_workers, drop_last=True)
    return val_loader

def get_val_WSI_loader(val_files, args):
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"],
                       reader=PILReader, dtype=np.uint8),
            AddChanneld(keys=["label"], allow_missing_keys=True),
            AsChannelFirstd(keys=['img'], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            EnsureTyped(keys=["img", "label"]),
        ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1,
                            shuffle=False, num_workers=args.num_workers)
    return val_loader

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
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
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