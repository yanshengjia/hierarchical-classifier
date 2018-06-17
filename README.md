# Hierarchical Classifier

A hierarchical classification system based on traditional machine learning models (LR, SVC, GBDT, RF) and deep learning models (LSTM + Attention).

The idea of hierarchical classification is similar with Blending / Stacking in Ensemble Learning.

## Introduction

Divide all the features extracted from essays into 5 categories:

- Lexical Features
- Grammar Error Features
- Sentence Features
- Structure Features
- Content Features

将这5大类共96维特征，将其输入层次分类器，得到最终的作文得分预测值。

The hierarchical classifier has 2 layers:

- Base layer: 5个分类器，分别对应5类作文特征：词法特征，句法特征，结构特征，内容特征，语法错误特征
  - input：从作文文本中抽取出的特征向量，按类别切成5个特征向量，词法特征向量输入词法特征分类器
  - output：3类，代表作文在某一维度表现的好中坏，e.g. (0, 0, 1)
  - label：按作文得分分桶 (bucketize)，根据数值范围将其值分为不同的类别，这里是好中坏三类
- Fuse layer: 第一层5个分类器输出的5个向量拼接而成，共15维
  - output：16类，对应于初中作文分数范围 [0, 15]
  - label：作文得分真值

## Environment

* python
* sklearn
* numpy
* scipy
* tensorflow

## Data

我们使用作文特征数据集来训练和评估层次分类器。

这个数据集可以在这找到：`hierarchical-classifier/data/essay_features.csv`

使用下面的命令可重现地切分数据集，默认是80%训练集，10%验证集和10%测试集。切分的比例可以通过修改 `testset_ratio` 自行设定。

```shell
cd /path/to/hierarchical-classifier/utils
python3 split_dataset.py
```


## Feature Configuration

所有特征与其从属的类别的关系写在 `conf/feature.config` 中，YAML 格式。可以自行添加新的类别和特征。

类别名后带有冒号，每个特征采用区块格式 (block format)，也就是短杠+空格作为起始。

例子：

```yaml
category:
- feature0
- feature1
- feature2
```

## Usage

下面的命令可以在作文特征数据集上训练和评估层次分类器，每个分类器都使用 LR。

```shell
cd /path/to/hierarchical-classifier/src
python3 run.py
	--train
	--train_files ../data/trainset/essay.train.csv # allow multiple files, separated by space
	--dev_files ../data/devset/essay.dev.csv
	--test_files ../data/testset/essay.test.csv
	--result_dir ../data/results/
	--c1 lr
	--c2 lr
	--c3 lr
	--c4 lr
	--c5 lr
	--fuse lr
```

You can see the list of available options by running:

```shell
python3 run.py -h
```

```shell
usage: Hierarchical Classification on essay feature dataset
       [-h] [--prepare] [--train] [--evaluate] [--predict] [--cv]
       [--epochs EPOCHS] [--optim OPTIM] [--learning_rate LEARNING_RATE]
       [--weight_decay WEIGHT_DECAY] [--dropout_keep_prob DROPOUT_KEEP_PROB]
       [--batch_size BATCH_SIZE] [--base {gbdt,rf,svc,mnb,lrcv,lr}]
       [--c1 {gbdt,rf,svc,mnb,lrcv,lr}] [--c2 {gbdt,rf,svc,mnb,lrcv,lr}]
       [--c3 {gbdt,rf,svc,mnb,lrcv,lr}] [--c4 {gbdt,rf,svc,mnb,lrcv,lr}]
       [--c5 {gbdt,rf,svc,mnb,lrcv,lr}] [--fuse {gbdt,rf,svc,mnb,lrcv,lr}]
       [--data_files DATA_FILES [DATA_FILES ...]]
       [--train_files TRAIN_FILES [TRAIN_FILES ...]]
       [--dev_files DEV_FILES [DEV_FILES ...]]
       [--test_files TEST_FILES [TEST_FILES ...]] [--model_dir MODEL_DIR]
       [--result_dir RESULT_DIR] [--summary_dir SUMMARY_DIR]
       [--config_path CONFIG_PATH] [--log_path LOG_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --prepare             create the directories, check data
  --train               train the model
  --evaluate            evaluate the model on dev set
  --predict             predict the answers for test set with trained model

train settings:
  --cv                  use cross validation
  --epochs EPOCHS       train epochs
  --optim OPTIM         optimizer type
  --learning_rate LEARNING_RATE
                        learning rate
  --weight_decay WEIGHT_DECAY
                        weight decay
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        dropout keep rate
  --batch_size BATCH_SIZE
                        train batch size

model settings:
  --base {gbdt,rf,svc,mnb,lrcv,lr}
                        choose the algorithm for all classifiers in base layer
  --c1 {gbdt,rf,svc,mnb,lrcv,lr}
                        choose the algorithm for classifier 1 (lexical) in
                        base layer
  --c2 {gbdt,rf,svc,mnb,lrcv,lr}
                        choose the algorithm for classifier 2 (grammar) in
                        base layer
  --c3 {gbdt,rf,svc,mnb,lrcv,lr}
                        choose the algorithm for classifier 3 (sentence) in
                        base layer
  --c4 {gbdt,rf,svc,mnb,lrcv,lr}
                        choose the algorithm for classifier 4 (structure) in
                        base layer
  --c5 {gbdt,rf,svc,mnb,lrcv,lr}
                        choose the algorithm for classifier 5 (content) in
                        base layer
  --fuse {gbdt,rf,svc,mnb,lrcv,lr}
                        choose the algorithm for fuse layer

path settings:
  --data_files DATA_FILES [DATA_FILES ...]
                        list of files that contain the preprocessed data for
                        cross validation
  --train_files TRAIN_FILES [TRAIN_FILES ...]
                        list of files that contain the preprocessed train data
  --dev_files DEV_FILES [DEV_FILES ...]
                        list of files that contain the preprocessed dev data
  --test_files TEST_FILES [TEST_FILES ...]
                        list of files that contain the preprocessed test data
  --model_dir MODEL_DIR
                        the dir to store models
  --result_dir RESULT_DIR
                        the dir to output the results
  --summary_dir SUMMARY_DIR
                        the dir to write tensorboard summary
  --config_path CONFIG_PATH
                        path of the config file.
  --log_path LOG_PATH   path of the log file.
```

