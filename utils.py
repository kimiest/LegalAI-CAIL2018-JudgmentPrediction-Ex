import json
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer


# ***************************************************
# 读取大尺寸.json格式的数据，返回格式大致为：[{}, {}, ...]
# ***************************************************
def read_json(path):
    tempt = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            tempt.append(dic)
    return tempt



# ***************************************************
# 从训练集中划分出：1.训练集（4/5） 2.开发验证集（1/5）
# 得到训练集和开发验证集的id列表：train_data_ids = valid_data_ids = []
# ***************************************************
def divide_data(data, split_k):
    fact_list = [x['fact'].strip for x in data]
    label_list = [x['meta']['accusation'][0] for x in data]
    skf = StratifiedKFold(n_splits=split_k)
    for train_data_ids, valid_data_ids in skf.split(fact_list, label_list):
        return train_data_ids, valid_data_ids



# ***************************************************
# 覆写Dataset类，用于得到模型输入
# ***************************************************
class MyDataset(Dataset):
    def __init__(self, data):  # data=[{},{},{}...]  unique_labels=[]
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.data = data

        # 获取罪名标签编码器（列表），用于将str类型的标签映射为int类型，共196个不同的罪名
        with open(r'./data_cail2018/unique_accusation_labels.json') as f:
            self.label_encoder_1 = json.load(f)

        # 获取法条标签编码器并拟合，用于将str类型的标签映射为one-hot类型，共183个不同的法条
        with open(r'./data_cail2018/unique_article_labels.json') as f:
            unique_article_labels = json.load(f)
        self.label_encoder_2 = MultiLabelBinarizer()
        self.label_encoder_2.fit([unique_article_labels])  # 一定记得再套一层[]

        # 获取刑期标签编码器（列表），将连续的刑期映射为离散的类别标签
        self.label_encoder_3 = [-1, -2, 1, 2, 3, 6, 8, 11, 18, 31, 52, 87, 146, 243]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fact = self.data[idx]['fact'].strip()

        # 获取罪名标签，并将str格式映射为int格式
        label_1 = self.label_encoder_1.index(self.data[idx]['meta']['accusation'][0])

        # 获取法条标签，将str格式映射为one-hot，因为是多标签（一定要注意再套一层[]，卡了很久）
        label_2 = self.label_encoder_2.transform([self.data[idx]['meta']['relevant_articles']])

        # 获取刑期标签，将连续值映射为离散值
        label_3 = None
        if self.data[idx]['meta']['term_of_imprisonment']['death_penalty']:
            label_3 = self.label_encoder_3.index(-1)
        elif self.data[idx]['meta']['term_of_imprisonment']['life_imprisonment']:
            label_3 = self.label_encoder_3.index(-2)
        else:
            for i, border in enumerate(self.label_encoder_3):
                if self.data[idx]['meta']['term_of_imprisonment']['imprisonment'] < border:
                    label_3 = i
                    break
            if label_3 == None:
                label_3 = len(self.label_encoder_3)

        # 对str类型的事实描述进行tokenize
        inputa = self.tokenizer(fact,
                                truncation=True,
                                add_special_tokens=True,
                                max_length=200,
                                padding='max_length',
                                return_tensors='pt')

        # inputa.keys() = input_ids, attention_mask, token_type_ids
        # inputa['input_ids'].shape = [batch_size, 1, max_length]
        # label_1.shape = [batch_size]
        # label_2.shape = [batch_size, 1, 183]
        return inputa, label_1, label_2.squeeze(), label_3



# *********************************************************************
# 利用 1.read_json函数 2.MyDataset类 返回DataLoader类型的训练/开发验证数据
# *********************************************************************
def get_train_valid_dataloader(path, train_batch_size, valid_batch_size, split_k):
    train_data = read_json(path)
    train_data_ids, valid_data_ids = divide_data(train_data, split_k)

    train_ds = MyDataset([train_data[i] for i in train_data_ids])
    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, pin_memory=True)

    valid_ds = MyDataset([train_data[i] for i in valid_data_ids])
    valid_dl = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle=True, pin_memory=True)
    return train_dl, valid_dl



# **************************************************************
# 利用 1.read_json函数 2.MyDataset类 返回DataLoader类型的测试数据
# **************************************************************
def get_test_dataloader(path, test_batch_size):
    test_data = read_json(path)

    test_ds = MyDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=True, pin_memory=True)
    return test_dl



# ********************************************************
# 选择学习率调整策略 1.（默认）余弦模拟退火 2.余弦模拟退火热重启
# ********************************************************
def fetch_scheduler(optimizer, schedule='CosineAnnealingLR'):
    if schedule == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500,
                                                   eta_min=1e-6)
    elif schedule == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100,
                                                             eta_min=1e-6)
    elif schedule == None:
        return None

    return scheduler



# *********************************************
# 设置可人工赋值的随机种子，以保证结果可复现
# *********************************************
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


