# !/usr/bin/python
# -*- coding:utf-8 -*-  
# @author: Shengjia Yan
# @date: 2018-06-05 Tuesday
# @email: i@yanshengjia.com
# Copyright @ Shengjia Yan. All Rights Reserved.
"""
This module prepares and runs the whole hierarchical classification model.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import sys
sys.path.append('../utils')
import time
import pickle
import argparse
import logging
import random
import numpy as np

from dataset import HCDataset
from hc_model import HCModel
from plot_confusion_matrix import *

logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] (%(asctime)s) (%(name)s) %(message)s',
        handlers=[
            logging.FileHandler('../data/log/hc.log', encoding='utf8'),
            logging.StreamHandler()
        ])
logger = logging.getLogger('main')


def parse_args():
	"""
	Parses command line arguments.
	"""
	parser = argparse.ArgumentParser('Hierarchical Classification on essay feature dataset')
	parser.add_argument('--prepare', action='store_true', help='create the directories, check data')
	parser.add_argument('--train', action='store_true', help='train the model')
	parser.add_argument('--evaluate', action='store_true', help='evaluate the model on dev set')
	parser.add_argument('--predict', action='store_true', help='predict the answers for test set with trained model')

	train_settings = parser.add_argument_group('train settings')
	train_settings.add_argument('--cv', action='store_true', help='use cross validation')
	train_settings.add_argument('--epochs', type=int, default=10, help='train epochs')
	train_settings.add_argument('--optim', default='adam', help='optimizer type')
	train_settings.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
	train_settings.add_argument('--weight_decay', type=float, default=0, help='weight decay')
	train_settings.add_argument('--dropout_keep_prob', type=float, default=1, help='dropout keep rate')
	train_settings.add_argument('--batch_size', type=int, default=32, help='train batch size')

	model_settings = parser.add_argument_group('model settings')
	model_settings.add_argument('--model_type', choices=['single', 'multi'], help='single channel model: base + fuse; multichannel model: combo + fuse')
	model_settings.add_argument('--base', choices=['gbdt', 'rf', 'svc', 'mnb', 'lrcv', 'lr'], help='choose the algorithm for all classifiers in base layer')
	model_settings.add_argument('--c1', choices=['gbdt', 'rf', 'svc', 'mnb', 'lrcv', 'lr'], default='lr', help='choose the algorithm for classifier 1 (lexical) in base layer')
	model_settings.add_argument('--c2', choices=['gbdt', 'rf', 'svc', 'mnb', 'lrcv', 'lr'], default='lr', help='choose the algorithm for classifier 2 (grammar) in base layer')
	model_settings.add_argument('--c3', choices=['gbdt', 'rf', 'svc', 'mnb', 'lrcv', 'lr'], default='lr', help='choose the algorithm for classifier 3 (sentence) in base layer')
	model_settings.add_argument('--c4', choices=['gbdt', 'rf', 'svc', 'mnb', 'lrcv', 'lr'], default='lr', help='choose the algorithm for classifier 4 (structure) in base layer')
	model_settings.add_argument('--c5', choices=['gbdt', 'rf', 'svc', 'mnb', 'lrcv', 'lr'], default='lr', help='choose the algorithm for classifier 5 (content) in base layer')
	model_settings.add_argument('--combo', choices=['gbdt', 'rf', 'svc', 'mnb', 'lrcv', 'lr'], default='lr', help='choose the algorithm for combo layer')
	model_settings.add_argument('--fuse', choices=['gbdt', 'rf', 'svc', 'mnb', 'lrcv', 'lr'], default='lr', help='choose the algorithm for fuse layer')

	path_settings = parser.add_argument_group('path settings')
	path_settings.add_argument('--data_files', nargs='+',
                               default=['../data/essay_features.csv'],
                               help='list of files that contain the preprocessed data for cross validation')
	path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/trainset/essay.train.csv'],
                               help='list of files that contain the preprocessed train data')
	path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/devset/essay.dev.csv'],
                               help='list of files that contain the preprocessed dev data')
	path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/testset/essay.test.csv'],
                               help='list of files that contain the preprocessed test data')
	path_settings.add_argument('--model_dir',
                               default='../data/models/',
                               help='the dir to store models')					   
	path_settings.add_argument('--result_dir',
                               default='../data/results/',
                               help='the dir to output the results')
	path_settings.add_argument('--summary_dir',
                               default='../data/summary/',
                               help='the dir to write tensorboard summary')
	path_settings.add_argument('--config_path',
                               default='../config/feature.config',
                               help='path of the config file.')
	path_settings.add_argument('--log_path',
                               default='../data/log/hc.log',
                               help='path of the log file.')
	path_settings.add_argument('--essay_path',
                               default='../data/essays.txt',
                               help='path of the essay file.')
	return parser.parse_args()

def prepare(args):
	"""
	checks data, creates the directories
	"""
	pass

def train(args):
	"""
	trains the hierarchical classification model
	"""
	logger.info('Loading dataset and configuration...')
	hc_data = HCDataset(args.config_path, args.data_files, args.train_files, args.dev_files, args.test_files)
	logger.info('Initialize the model...')
	hc_model = HCModel(args)
	logger.info('Training the model...')
	hc_model.train(hc_data)
	logger.info('Done with model training!')


def evaluate(args):
	"""
	evaluate the trained model on dev files
	"""
	logger.info('Done with model evaluation!')


def predict(args):
	"""
	predicts scores for test files
	"""
	assert len(args.test_files) > 0, 'No test files are provided.'
	logger.info('Loading dataset and configuration...')
	hc_data = HCDataset(args.config_path, args.data_files, args.train_files, args.dev_files, args.test_files)
	logger.info('Restoring the model...')
	hc_model = HCModel(args)
	hc_model.restore(hc_data)
	logger.info('Predicting scores for test set...')
	hc_model.predict(hc_data)

def main():
	"""
	Prepares and runs the whole system.
	"""
	args = parse_args()
	logger.info('Running with args : {}'.format(args))

	if args.prepare:
		prepare(args)
	if args.train:
		train(args)
	if args.evaluate:
		evaluate(args)
	if args.predict:
		predict(args)


if __name__ == '__main__':
	main()