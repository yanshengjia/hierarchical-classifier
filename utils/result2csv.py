# !/usr/bin/python
# -*- coding:utf-8 -*-  
# @author: Shengjia Yan
# @date: 2018-06-12 Tuesday
# @email: i@yanshengjia.com
# Copyright @ Shengjia Yan. All Rights Reserved.

import csv
import json
import codecs

result_json_path = '../data/results/result.json'
result_csv_path = '../data/results/result.csv'

def result2csv(json_path, csv_path):
    res = []
    with codecs.open(json_path, mode='r', encoding="utf8", errors='ignore') as in_file:
        for line in in_file:
            line = line.strip()
            one = json.loads(line)
            res.append(one)
    
    with open(csv_path, mode='w', encoding="utf8", errors='ignore') as out_file:
        writer = csv.DictWriter(out_file, res[0].keys())
        writer.writeheader()
        for row in res:
            writer.writerow(row)


def main():
    result2csv(result_json_path, result_csv_path)

if __name__ == '__main__':
    main()
