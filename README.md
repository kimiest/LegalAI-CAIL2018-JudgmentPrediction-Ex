# CAIL2018-罪名预测任务-包含三个子任务（1.罪名预测 2.法条推荐 3.刑期预测）

<br>
<br>
<br>
CAIL2018罪名预测任务官方数据集链接：(https://github.com/thunlp/CAIL)
<br>
<br>
<br>

## 1. 说明

（1）主要基于Pytorch和HuggingFace Transformers框架<br>
（2）模型网络结构为：基于BERT的多任务学习<br>
（3）使用了动态学习率和权重衰减<br>
（4）使用了pin_memory提升数据加载速度<br>
（5）全代码包含详尽的中文注释<br>

## 2. 文件介绍

（1）models文件夹--->包含模型网络结构BERT.py  <br>
（2）data_cail2018文件夹--->包含官方发布的训练数据train.json和测试数据test.json，以及全部的罪名标签unique_accusation_labels.json和全部的法条标签unique_article_labels.json  <br>
（3）preprocessing文件夹--->包含预处理和EDA代码  <br>

## 3. 模型训练
直接运行run.py可以执行模型训练

## 4. 模型测试
直接运行test.py可以执行模型测试
