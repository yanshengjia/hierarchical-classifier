# !/usr/bin/python
# -*- coding:utf-8 -*-  
# @author: Shengjia Yan
# @date: 2018-06-12 Tuesday
# @email: i@yanshengjia.com
# Copyright @ Shengjia Yan. All Rights Reserved.
"""
This module implements data preprocessing.
"""

import csv
import json
import codecs


raw_essay_features_path = '../data/essay_features_72.csv'
scaled_essay_features_path = '../data/essay_features_72_scaled.csv'


def scale_scores(self, raw_path, upper_limit = 15.0):
    '''
    scale all essays' total_score to a preset value
    '''
    features = []
    with codecs.open(raw_path, 'r') as raw_file:
        reader = csv.DictReader(raw_file)
        for line in reader:
            total_score = float(line['total_score'])
            score = float(line['score'])
            
            if total_score != upper_limit:
                scale_ratio = upper_limit / total_score
                new_score = round(score * scale_ratio)
                line['total_score'] = upper_limit
                line['score'] = new_score
            
            features.append(line)
    return features
    
def save_scaled_scores(self, save_path, features):
    with codecs.open(save_path, 'w') as o_file:
        writer = csv.DictWriter(o_file, features[0].keys())
        writer.writeheader()
        for row in features:
            writer.writerow(row)

def main():
    pass

if __name__ == '__main__':
    main()


