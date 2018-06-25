# !/usr/bin/python
# -*- coding:utf-8 -*-  
# @author: Shengjia Yan
# @date: 2018-06-08 Friday
# @email: i@yanshengjia.com
# Copyright @ Shengjia Yan. All Rights Reserved.
'''
This module splits the raw dataset into trainset and testset.
By default, 80% train, 10% dev and 10% test.
'''

import csv
import codecs
import random

raw_dataset_path = '../data/essay_features.csv'
trainset_path    = '../data/trainset/essay.train.csv'
devset_path      = '../data/devset/essay.dev.csv'
testset_path     = '../data/testset/essay.test.csv'
testset_ratio    = 0.1

def split_dataset(dataset_path):
    dataset = []
    with codecs.open(dataset_path, 'r') as file:
        reader = csv.DictReader(file)
        for line in reader:
            dataset.append(line)
    
    # make sure to have the same split each time this code is run
    # dataset.sort()
    random.seed(666)
    random.shuffle(dataset)

    dataset_size  = len(dataset)
    split_1  = int((1 - 2 * testset_ratio) * dataset_size)
    split_2  = int((1 - testset_ratio) * dataset_size)
    trainset = dataset[:split_1]
    devset   = dataset[split_1:split_2]
    testset  = dataset[split_2:]
    return trainset, devset, testset

def save_dataset(dataset_name, dataset):
    if dataset_name == 'trainset':
        dataset_path = trainset_path
    elif dataset_name == 'devset':
        dataset_path = devset_path
    elif dataset_name == 'testset':
        dataset_path = testset_path
    else:
        raise ValueError('dataset_name should be \'trainset\' or \'devset\' or \'testset\'!')
    
    with codecs.open(dataset_path, 'w') as o_file:
        writer = csv.DictWriter(o_file, dataset[0].keys())
        writer.writeheader()
        for row in dataset:
            writer.writerow(row)
    print('{} size: {}'.format(dataset_name, len(dataset)))

def main():
    trainset, devset, testset = split_dataset(raw_dataset_path)
    save_dataset('trainset', trainset)
    save_dataset('devset', devset)
    save_dataset('testset', testset)

if __name__ == '__main__':
    main()