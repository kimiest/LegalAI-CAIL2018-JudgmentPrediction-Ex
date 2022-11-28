from utils import read_json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json


def statistics():
    all_data = read_json(r'./data_cail2018/train.json')
    facts = []
    labels = []
    for item in all_data:
        facts.append(item['fact'].strip())
        labels.append(item['meta']['accusation'][0])

    '''统计文本的最大长度、平均长度、最小长度，以字符为单位'''
    fact_length = list(map(len, facts))
    max_length = np.max(fact_length)
    mean_length = np.mean(fact_length)
    mix_length = np.min(fact_length)
    num_long_sent = np.sum(x > 512 for x in fact_length)

    return max_length, mean_length, mix_length, num_long_sent

from collections import defaultdict
import matplotlib.pyplot as plt

to_labels = [1, 2, 3, 6, 8, 11, 18, 31, 52, 87, 146, 243]
count = defaultdict(int)
for item in read_json(r'..\data_cail2018\train.json'):
    term = item['meta']['term_of_imprisonment']['imprisonment']
    is_death = item['meta']['term_of_imprisonment']['death_penalty']
    is_life = item['meta']['term_of_imprisonment']['life_imprisonment']
    for i, label in enumerate(to_labels):
        if term < label:
            count[str(i)] += 1
            break
index = count.keys()
values = count.values()
plt.bar(index, values)
plt.title("刑期分布", fontsize=20, fontname="Times New Roman")
plt.show()






