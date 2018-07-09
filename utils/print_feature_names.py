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
import yaml
import codecs

feature_filepath = '../data/essay_features.csv'
config_path = '../config/feature.config'

def print_feature_names():
    with codecs.open(feature_filepath, 'r') as file:
        reader = csv.DictReader(file)
        for line in reader:
            feature_list = list(line.keys())
            # print("\n".join(feature_list))
            # print(len(feature_list))
            return feature_list

def compare_with_config(feature_list):
    with open(config_path, 'r') as stream:
        feature_config = yaml.load(stream)
    
    print('Deprecated features:')
    for feature_type in list(feature_config.keys()):
        print('\n[' + feature_type + ']')
        config_feature_list = feature_config[feature_type]
        deprecated_features = []
        for feature in config_feature_list:
            if feature not in feature_list:
                deprecated_features.append(feature)
        print("\n".join(deprecated_features))

def main():
    feature_list = print_feature_names()
    compare_with_config(feature_list)

if __name__ == '__main__':
    main()