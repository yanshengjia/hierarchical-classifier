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
        self.algo1   = args.algo1
        self.algo2   = args.algo2
        
        if self.algo1:
            self.algo1_1 = self.algo1
            self.algo1_2 = self.algo1
            self.algo1_3 = self.algo1
            self.algo1_4 = self.algo1
            self.algo1_5 = self.algo1
        else:
            self.algo1_1 = args.algo1_1
            self.algo1_2 = args.algo1_2
            self.algo1_3 = args.algo1_3
            self.algo1_4 = args.algo1_4
            self.algo1_5 = args.algo1_5
        self.base_models = {
                            'lexical': self.algo1_1, 
                            'grammar': self.algo1_2, 
                            'sentence': self.algo1_3, 
                            'structure': self.algo1_4, 
                            'content': self.algo1_5
                        }
        
        # save info
        self.result_path = args.result_dir + 'result.json'

    def _crated_model(self, model_type=''):
        '''
        Selects the classification model
        '''
        if model_type == 'gbdt':
            model = GradientBoostingClassifier()
        elif model_type == 'rf':
            model = RandomForestClassifier()
        elif model_type == 'svc':
            model = svm.SVC(kernel='linear', probability=True)
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
        self.n_classifiers = len(list(data.feature_types))
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

    def _fuse(self, features):
        '''
        The fuse layer, including 1 classifier
        '''
        self.fuse_layer = self._crated_model(self.algo2)
        self.fuse_layer.fit(features, self.labels)

    def _concat_features(self, feature_list):
        '''
        Concat feature matrixs horizontally
        Args:
            feature_list: the list of feature matrix, each is (n_samples, 3)
        Rtype:
            features: (n_samples, 15)
        '''
        first_flag = True
        for feature in feature_list:
            if first_flag:
                features = feature
                first_flag = False
            else:
                features = np.concatenate((features, feature), axis=1)
        assert features[0].size == 3 * self.n_classifiers, 'The dim of base features is incorrect!'
        return features

    def _build_model(self, data):
        logger.info('Building base layer...')
        self._base(data)
        logger.info('Generating base layer output...')
        base_output_list = []
        for feature_type in data.feature_types: 
            base_output = self.base_layer[feature_type].predict_proba(self.base_feature[feature_type])
            base_output_list.append(base_output)
        base_output_features = self._concat_features(base_output_list)
        logger.info('Building fuse layer...')
        self._fuse(base_output_features)

        logger.info('------------------------------')
        logger.info('Model Architecture:')
        logger.info('* Base Layer:')
        for feature_type in data.feature_types:
            logger.info('  * {} classifier: {}'.format(feature_type, self.base_models[feature_type]))
        logger.info('* Fuse Layer: {}'.format(self.algo2))
        logger.info('------------------------------')

    def cross_validation(self):
        '''
        Use cross validation
        '''
        pass

    def train(self, data, evaluate=True):
        """
        Train the model with data
        Args:
            data: the HCDataset class implemented in dataset.py
            evaluate: whether to evaluate the model on test set after training
        """
        self._build_model(data)

        if evaluate:
            self.evaluate(data)

    def evaluate(self, data, save=True):
        """
        Evaluate the model
        Args:
            data: the HCDataset class implemented in dataset.py
            save: whether to save the evaluation results
        """
        logger.info('Evaluating the model on dev set:')
        base_output_list = []
        for feature_type in data.feature_types:
            features, y_true = data._gen_input(data.dev_set, feature_type=feature_type)
            base_output = self.base_layer[feature_type].predict_proba(features)
            base_output_list.append(base_output)
        base_output_features = self._concat_features(base_output_list)
        y_pred = self.fuse_layer.predict(base_output_features)

        self.dev_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        self.dev_lwk = cohen_kappa_score(y_true, y_pred, weights='linear')
        self.dev_prs, p_value = pearsonr(y_true, y_pred)
        self.dev_acc = accuracy_score(y_true, y_pred)
        logger.info('  [DEV]  QWK: %.3f, LWK: %.3f, PRS: %.3f, ACC: %.3f' % (self.dev_qwk, self.dev_lwk, self.dev_prs, self.dev_acc))

        logger.info('Evaluating the model on test set:')
        base_output_list = []
        for feature_type in data.feature_types:
            features, y_true = data._gen_input(data.test_set, feature_type=feature_type)
            base_output = self.base_layer[feature_type].predict_proba(features)
            base_output_list.append(base_output)
        base_output_features = self._concat_features(base_output_list)
        y_pred = self.fuse_layer.predict(base_output_features)

        self.test_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        self.test_lwk = cohen_kappa_score(y_true, y_pred, weights='linear')
        self.test_prs, p_value = pearsonr(y_true, y_pred)
        self.test_acc = accuracy_score(y_true, y_pred)
        logger.info('  [TEST] QWK: %.3f, LWK: %.3f, PRS: %.3f, ACC: %.3f' % (self.test_qwk, self.test_lwk, self.test_prs, self.test_acc))
        logger.info('Done with model evaluation!')

        if save:
            self.save_results()
    
    def predict(self):
        pass
    
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
        logger.info('The evaluation results saved.')
    
    def save_models(self, model_dir, model_predix):
        pass
    
    def restore(self, model_dir, model_predix):
        pass






