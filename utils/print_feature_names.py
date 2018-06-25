# !/usr/bin/python
# -*- coding:utf-8 -*-  
# @author: Shengjia Yan
# @date: 2018-06-07 Thursday
# @email: i@yanshengjia.com
# Copyright @ Shengjia Yan. All Rights Reserved.
'''
This module prints all features' names.
'''

import csv
import codecs

feature_filepath = '../data/essay_features.csv'

def print_feature_names():
    with codecs.open(feature_filepath, 'r') as file:
        reader = csv.DictReader(file)
        for line in reader:
            print("\n".join(list(line.keys())))
            print(len(list(line.keys())))
            break

def main():
    print_feature_names()

if __name__ == '__main__':
    main()