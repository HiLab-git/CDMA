#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 下午4:41
# @Author  : chuyu zhang
# @File    : metrics.py
# @Software: PyCharm

import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from medpy import metric


def get_soft_label(input_tensor, num_class, data_type='float'):
    """
        convert a label tensor to one-hot label 
        input_tensor: tensor with shae [B, 1, D, H, W] or [B, 1, H, W]
        output_tensor: shape [B, num_class, D, H, W] or [B, num_class, H, W]
    """

    shape = input_tensor.shape
    if len(shape) == 5:
        output_tensor = torch.nn.functional.one_hot(
            input_tensor[:, 0], num_classes=num_class).permute(0, 4, 1, 2, 3)
    elif len(shape) == 4:
        output_tensor = torch.nn.functional.one_hot(
            input_tensor[:, 0], num_classes=num_class).permute(0, 3, 1, 2)
    else:
        raise ValueError(
            "dimention of data can only be 4 or 5: {0:}".format(len(shape)))

    if(data_type == 'float'):
        output_tensor = output_tensor.float()
    elif(data_type == 'double'):
        output_tensor = output_tensor.double()
    else:
        raise ValueError(
            "data type can only be float and double: {0:}".format(data_type))

    return output_tensor


def reshape_prediction_and_ground_truth(predict, soft_y):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(predict.size())
    num_class = list(predict.size())[1]
    if(tensor_dim == 5):
        soft_y = soft_y.permute(0, 2, 3, 4, 1)
        predict = predict.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        soft_y = soft_y.permute(0, 2, 3, 1)
        predict = predict.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))

    predict = torch.reshape(predict, (-1, num_class))
    soft_y = torch.reshape(soft_y,  (-1, num_class))

    return predict, soft_y


def get_classwise_dice(predict, soft_y, pix_w=None):
    """
    get dice scores for each class in predict (after softmax) and soft_y
    """

    if(pix_w is None):
        y_vol = torch.sum(soft_y,  dim=0)
        p_vol = torch.sum(predict, dim=0)
        intersect = torch.sum(soft_y * predict, dim=0)
    else:
        y_vol = torch.sum(soft_y * pix_w,  dim=0)
        p_vol = torch.sum(predict * pix_w, dim=0)
        intersect = torch.sum(soft_y * predict * pix_w, dim=0)
    dice_score = (2.0 * intersect + 1e-5) / (y_vol + p_vol + 1e-5)
    return dice_score


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
            (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dc, jc, hd, asd


def dice(input, target, ignore_index=None):
    smooth = 1.
    # using clone, so that it can do change to original target.
    iflat = input.clone().view(-1)
    tflat = target.clone().view(-1)
    if ignore_index is not None:
        mask = tflat == ignore_index
        tflat[mask] = 0
        iflat[mask] = 0
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class DiceMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.train_dice_list = []

    def add_batch(self, pred, gt):
        n = pred.shape[0]
        for i in range(n):
            pred_seg = torch.argmax(pred[i], dim=0)
            pred_seg = pred_seg.cpu().numpy()
            outputs_argmax = np.expand_dims((np.expand_dims(pred_seg, 0)), 0)
            outputs_argmax = torch.tensor(outputs_argmax).long()

            soft_out = get_soft_label(outputs_argmax, self.num_classes)
            labels_prob = gt[i].unsqueeze(0).unsqueeze(0).long()
            labels_prob = get_soft_label(labels_prob, self.num_classes)
            soft_out, labels_prob = reshape_prediction_and_ground_truth(
                soft_out, labels_prob)
            dice_list = get_classwise_dice(soft_out, labels_prob)
            self.train_dice_list.append(dice_list.cpu().numpy())

    def compute_dice(self, verbose=False):
        train_dice_list = np.asarray(self.train_dice_list)*100
        train_dice_list = train_dice_list[1:]
        train_cls_dice = train_dice_list.mean(axis=0)
        train_avg_dice = train_dice_list.mean(axis=1)
        train_std_dice = train_avg_dice.std()
        train_scalers = {'avg_dice': train_avg_dice.mean(
        ), 'class_dice': train_cls_dice, 'std_dice': train_std_dice}

        if verbose:
            print("%.2f" % train_cls_dice[0], "%.2f" % train_cls_dice[1],
                  "%.2f" % train_cls_dice[2], "%.2f" % train_cls_dice.mean())
        else:
            print("%.2f" % train_cls_dice.mean())
        return train_cls_dice.mean()
