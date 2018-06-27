# !/usr/bin/python
# -*- coding:utf-8 -*-  
# @author: Shengjia Yan
# @date: 2018-06-05 Tuesday
# @email: i@yanshengjia.com
# Copyright @ Shengjia Yan. All Rights Reserved.
"""
This module implements the hierarchical classification models.
"""

import os
import sys
sys.path.append('../utils')
import time
import json
import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from scipy.stats.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib

from dataset import HCDataset
from plot_confusion_matrix import *
from relative_accuracy import *

import logging
logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] (%(asctime)s) (%(name)s) %(message)s',
        handlers=[
            logging.FileHandler('../data/log/hc.log', encoding='utf8'),
            logging.StreamHandler()
        ])
logger = logging.getLogger('hc_model')


class HCModel(object):
    """
	Implements the main hierarchical classification Model
	"""
    def __init__(self, args):
        # basic config
        self.model_type = args.model_type
        self.base       = args.base
        self.combo      = args.combo
        self.fuse       = args.fuse
        
        if self.base:
            self.c1 = self.base
            self.c2 = self.base
            self.c3 = self.base
            self.c4 = self.base
            self.c5 = self.base
        else:
            self.c1 = args.c1
            self.c2 = args.c2
            self.c3 = args.c3
            self.c4 = args.c4
            self.c5 = args.c5
        self.base_models = {
                            'lexical': self.c1, 
                            'grammar': self.c2, 
                            'sentence': self.c3, 
                            'structure': self.c4, 
                            'content': self.c5
                        }
        
        # save info
        self.result_dir  = args.result_dir
        self.result_path = args.result_dir + 'result.json'
        self.cm_path     = args.result_dir + 'cm.png'
        self.model_dir   = args.model_dir

    def _crated_model(self, model_type=''):
        '''
        Selects the classification model
        '''
        if model_type == 'gbdt':
            model = GradientBoostingClassifier(n_estimators=100)
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100)
        elif model_type == 'svc':
            model = svm.SVC(kernel='linear', probability=True)    # kernel type: ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        elif model_type == 'mnb':
            model = MultinomialNB()
        elif model_type == 'lrcv':
            model = LogisticRegressionCV()
        elif model_type == 'lr':
            model = LogisticRegression()
        else:
            raise NotImplementedError('The model {} is not implemented.'.format(model_type))
        return model

    def _base(self, data):
        '''
        The base layer, including 5 classifiers
        Args:
            data: the HCDataset class implemented in dataset.py
        '''
        self.base_layer = {}    # the dict of base classifiers
        self.base_feature = {}  # the dict of input features for base classifiers
        self.n_base_features = data.n_base_features
        for feature_type in data.feature_types:
            logger.info('  Generate {} features and labels from train set.'.format(feature_type))
            features, self.labels = data._gen_input(data.train_set, feature_type=feature_type)
            self.base_feature[feature_type] = features
            self.base_labels = data.bucketize(self.labels)
            model_type = self.base_models[feature_type]
            assert model_type != '', 'The model type for {} classifier is not specified.'.format(feature_type)
            logger.info('  Train {} model for {} classifier.'.format(model_type, feature_type))
            base_model = self._crated_model(model_type)
            base_model.fit(features, self.base_labels)
            self.base_layer[feature_type] = base_model

    def _combo(self, data):
        '''
        The combo layer, including all feature combinations (31 types) of classifiers
        Args:
            data: the HCDataset class implemented in dataset.py
        '''
        self.combo_layer = {}    # the dict of combo classifiers
        self.combo_feature = {}  # the dict of input features for combo classifiers
        self.n_combo_features = data.n_combo_features
        for feature_type in data.combo_feature_types:
            logger.info('  Generate {} features and labels from train set.'.format(feature_type))
            f = feature_type.split('_')
            feature_list = []
            for f_type in f:
                features, self.labels = data._gen_input(data.train_set, feature_type=f_type)
                feature_list.append(features)
            features = self._concat_features(feature_list)
            self.combo_feature[feature_type] = features
            self.combo_labels = data.bucketize(self.labels)
            model_type = self.combo
            assert model_type != '', 'The model type for {} classifier is not specified.'.format(feature_type)
            logger.info('  Train {} model for {} classifier.'.format(model_type, feature_type))
            combo_model = self._crated_model(model_type)
            combo_model.fit(features, self.combo_labels)
            self.combo_layer[feature_type] = combo_model

    def _fuse(self, features):
        '''
        The fuse layer, including 1 classifier
        '''
        self.fuse_layer = self._crated_model(self.fuse)
        self.fuse_layer.fit(features, self.labels)
        self.fuse_output_dim = 16

    def _concat_features(self, feature_list):
        '''
        Concat feature matrixs horizontally
        Args:
            feature_list: the list of feature matrix
        Rtype:
            features: (n_samples, ?)
        '''
        first_flag = True
        for feature in feature_list:
            if first_flag:
                features = feature
                first_flag = False
            else:
                features = np.concatenate((features, feature), axis=1)
        return features

    def _build_model(self, data):
        if self.model_type == 'single':
            logger.info('Building base layer...')
            self._base(data)
            logger.info('Generating base layer output...')
            base_output_list = []
            for feature_type in data.feature_types:
                base_output = self.base_layer[feature_type].predict_proba(self.base_feature[feature_type])
                base_output_list.append(base_output)
            base_output_features = self._concat_features(base_output_list)
            self.base_output_dim = base_output_features[0].size
            logger.info('Base output dim: {}'.format(self.base_output_dim))
            logger.info('Building fuse layer...')
            self._fuse(base_output_features)
            logger.info('Fuse output dim: {}'.format(self.fuse_output_dim))

            logger.info('------------------------------')
            logger.info('Model Architecture:')
            logger.info('* Base Layer:')
            for feature_type in data.feature_types:
                logger.info('  * {} classifier: {}'.format(feature_type, self.base_models[feature_type]))
            logger.info('* Fuse Layer: {}'.format(self.fuse))
            logger.info('------------------------------')
        else:
            logger.info('Building combo layer...')
            self._combo(data)
            logger.info('Generating combo layer output...')
            combo_output_list = []
            for feature_type in data.combo_feature_types:
                combo_output = self.combo_layer[feature_type].predict_proba(self.combo_feature[feature_type])
                combo_output_list.append(combo_output)
            combo_output_features = self._concat_features(combo_output_list)
            self.combo_output_dim = combo_output_features[0].size
            logger.info('Combo output dim: {}'.format(self.combo_output_dim))
            logger.info('Building fuse layer...')
            self._fuse(combo_output_features)
            logger.info('Fuse output dim: {}'.format(self.fuse_output_dim))

            logger.info('------------------------------')
            logger.info('Model Architecture:')
            logger.info('* Combo Layer:')
            for feature_type in data.combo_feature_types:
                logger.info('  * {} classifier: {}'.format(feature_type, self.combo))
            logger.info('* Fuse Layer: {}'.format(self.fuse))
            logger.info('------------------------------')

    def cross_validation(self):
        '''
        Use cross validation
        '''
        pass

    def train(self, data, evaluate=True, save=True):
        """
        Train the model with data
        Args:
            data: the HCDataset class implemented in dataset.py
            evaluate: whether to evaluate the model on test set after training
            save: whether to save the models trained on train set
        """
        self._build_model(data)

        if evaluate:
            self.evaluate(data, dataset='train')
            self.evaluate(data, dataset='test')
        
        if save:
            self.save_models(data)


    def evaluate(self, data, dataset='test'):
        """
        Evaluate the model
        Args:
            data: the HCDataset class implemented in dataset.py
            dataset: which dataset to evaluate model, choices = ['train', 'dev', 'test']
        """
        logger.info('Evaluating the model on {} set:'.format(dataset))

        if dataset == 'train':
            data_set = data.train_set
        elif dataset == 'dev':
            data_set = data.dev_set
        else:
            data_set = data.test_set

        if self.model_type == 'single':
            base_output_list = []
            for feature_type in data.feature_types:
                features, y_true = data._gen_input(data_set, feature_type=feature_type)
                base_output = self.base_layer[feature_type].predict_proba(features)
                base_output_list.append(base_output)
            base_output_features = self._concat_features(base_output_list)
            y_pred = self.fuse_layer.predict(base_output_features)
        else:
            combo_output_list = []
            for feature_type in data.combo_feature_types:
                f = feature_type.split('_')
                feature_list = []
                for f_type in f:
                    features, y_true = data._gen_input(data_set, feature_type=f_type)
                    feature_list.append(features)
                features = self._concat_features(feature_list)
                combo_output = self.combo_layer[feature_type].predict_proba(features)
                combo_output_list.append(combo_output)
            combo_output_features = self._concat_features(combo_output_list)
            y_pred = self.fuse_layer.predict(combo_output_features)

        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        lwk = cohen_kappa_score(y_true, y_pred, weights='linear')
        prs, p_value = pearsonr(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        racc = relative_accuracy(y_true, y_pred)
        logger.info('  [{:5}]  QWK: {:.3f}, LWK: {:.3f}, PRS: {:.3f}, ACC: {:.3f}, RACC: {:.3f}'.format(dataset, qwk, lwk, prs, acc, racc)) 
        return qwk, lwk, prs, acc, racc
    
    def predict(self, data, dataset='test'):
        """
        Predict with the trained model
        Args:
            data: the HCDataset class implemented in dataset.py
            dataset: which dataset to evaluate model, choices = ['train', 'dev', 'test']
        """
        if dataset == 'train':
            data_set = data.train_set
        elif dataset == 'dev':
            data_set = data.dev_set
        else:
            data_set = data.test_set
        
        if self.model_type == 'single':
            base_output_list = []
            base_pred_label_list = []
            for feature_type in data.feature_types:
                features, y_true = data._gen_input(data_set, feature_type=feature_type)
                base_output = self.base_layer[feature_type].predict_proba(features)
                base_output_list.append(base_output)
                base_pred_label = self.base_layer[feature_type].predict(features)
                base_pred_label_list.append(base_pred_label)
            base_output_features = self._concat_features(base_output_list)
            y_pred = self.fuse_layer.predict(base_output_features)
            print(base_pred_label_list)
            print(y_pred)
        else:
            combo_output_list = []
            for feature_type in data.combo_feature_types:
                f = feature_type.split('_')
                feature_list = []
                for f_type in f:
                    features, y_true = data._gen_input(data_set, feature_type=f_type)
                    feature_list.append(features)
                features = self._concat_features(feature_list)
                combo_output = self.combo_layer[feature_type].predict_proba(features)
                combo_output_list.append(combo_output)
            combo_output_features = self._concat_features(combo_output_list)
            y_pred = self.fuse_layer.predict(combo_output_features)


    def draw_confusion_matrix(self, y_true, y_pred):
        cm = build_confusion_matrix(y_true, y_pred, self.fuse_output_dim, self.cm_path)
        logger.info('Confusion matrix fig saved in: {}'.format(self.cm_path))
        return cm

    def _build_result(self):
        result = {}
        result.update(self.base_models)
        result['fuse']     = self.algo2
        result['dev_qwk']  = self.dev_qwk
        result['dev_lwk']  = self.dev_lwk
        result['dev_prs']  = self.dev_prs
        result['dev_acc']  = self.dev_acc
        result['test_qwk'] = self.test_qwk
        result['test_lwk'] = self.test_lwk
        result['test_prs'] = self.test_prs
        result['test_acc'] = self.test_acc
        return result

    def save_results(self):
        with open(self.result_path, 'a') as o_file:
            res = self._build_result()
            res_json = json.dumps(res)
            o_file.write(res_json + '\n')
        logger.info('Evaluation results saved in: {}'.format(self.result_path))

    def save_models(self, data):
        """
        Saves the scalers and models into model_dir with feature_type as the model indicator
        """
        for feature_type in data.feature_types:
            scaler_prefix = feature_type
            scaler_path   = self.model_dir + '/scaler/' + scaler_prefix + '.scaler'
            joblib.dump(data.scalers[feature_type], scaler_path)
        logger.info('Scalers saved in {}'.format(self.model_dir + '/scaler'))

        if self.model_type == 'single':
            for feature_type in data.feature_types:
                model_prefix = feature_type
                model_path   = self.model_dir + self.model_type + '/' + model_prefix + '.pkl'
                joblib.dump(self.base_layer[feature_type], model_path)
        else:
            for feature_type in data.combo_feature_types:
                model_prefix = feature_type
                model_path   = self.model_dir + self.model_type + '/' + model_prefix + '.pkl'
                joblib.dump(self.combo_layer[feature_type], model_path)
        joblib.dump(self.fuse_layer, self.model_dir + self.model_type + '/fuse.pkl')
        logger.info('Models saved in {}'.format(self.model_dir + self.model_type))
    
    def restore(self, data):
        """
        Restores the model from model_dir with feature_type as the model indicator
        """
        if self.model_type == 'single':
            self.base_layer = {}
            for feature_type in data.feature_types:
                model_prefix = feature_type
                model_path   = self.model_dir + self.model_type + '/' + model_prefix + '.pkl'
                self.base_layer[feature_type] = joblib.load(model_path)
        else:
            self.combo_layer = {}
            for feature_type in data.combo_feature_types:
                model_prefix = feature_type
                model_path   = self.model_dir + self.model_type + '/' + model_prefix + '.pkl'
                self.combo_layer[feature_type] = joblib.load(model_path)
        self.fuse_layer = joblib.load(self.model_dir + self.model_type + '/fuse.pkl')
        logger.info('Models restored from {}'.format(self.model_dir))


