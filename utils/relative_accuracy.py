# !/usr/bin/python
# -*- coding:utf-8 -*-  
# @author: Shengjia Yan
# @date: 2018-06-20 Wednesday
# @email: i@yanshengjia.com
# Copyright @ Shengjia Yan. All Rights Reserved.

import numpy as np

def relative_accuracy(y_true, y_pred, gap = 2.0):
    sum = y_true.size
    correct = 0.0
    for i in range(sum):
        if abs(y_true[i] - y_pred[i]) <= gap:
            correct += 1
    racc = float(correct / sum)
    return racc
