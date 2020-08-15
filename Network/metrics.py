# -*- coding: utf-8 -*-
import torch
import math
import numpy as np

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

class Result(object):
    def __init__(self):
        self.mae = 0
        self.rmse = 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.mae = np.inf
        self.rmse = np.inf
        self.data_time, self.gpu_time = 0, 0

    def update(self, mae, gpu_time, data_time):
        self.mae = mae
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target, label):
        abs_diff = []
        count = 0
        for i in range(len(label)):
            if label[i]!=-1:
                abs_diff1 = (output[count,label[i],:,:] - target[count,:,:]).abs()
                abs_diff.append(float(abs_diff1.mean()))
            else:
                abs_diff2 = (output[count,0,:,:] - target[count,:,:]).abs() + (output[count,1,:,:] - target[count,:,:]).abs() + (output[count,2,:,:] - target[count,:,:]).abs()
                abs_diff.append(float(abs_diff2.mean()))
            count += 1
        self.mae = np.mean(abs_diff)
        self.data_time = 0
        self.gpu_time = 0


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_mae = 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n
        self.sum_mae += n*result.mae
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time

    def average(self):
        avg = Result()
        avg.update(self.sum_mae / self.count, self.sum_gpu_time / self.count, self.sum_data_time / self.count)
        return avg
