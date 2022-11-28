from utils import read_json
from tkinter import _flatten
import json

# ********************************************************************
# 读取全部数据中的罪名标签，得到unique的罪名label集合，并以list形式返回
# ********************************************************************
def save_unique_accusation_labels(train_path, test_path):
    train_data = read_json(train_path)
    test_data = read_json(test_path)

    labels = [x['meta']['accusation'][0] for x in (train_data + test_data)]
    unique_labels = list(set(labels))

    with open('../data_cail2018/unique_accusation_labels.json', 'w', encoding='utf-8') as f:
        json.dump(unique_labels, f)


# ********************************************************************
# 读取全部数据中的法条标签，得到unique的法条label集合，并以list形式返回
# ********************************************************************
def save_unique_article_labels(train_path, test_path):
    train_data = read_json(train_path)
    test_data = read_json(test_path)

    articles = [x['meta']['relevant_articles'] for x in (train_data + test_data)]
    unique_labels = list(set(_flatten(articles)))

    with open('../data_cail2018/unique_article_labels.json', 'w', encoding='utf-8') as f:
        json.dump(unique_labels, f)

