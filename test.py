import torch
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import torch.nn as nn
import torch

from models.BERT import MyModel
from utils import get_test_dataloader

# *******************************************************************************
# 也可以选择直接调用valid_one_epoch在测试集上跑一个epoch，
# 但这样会导致valid_one_epoch代码太多（开发验证阶段/最后测试阶返回不同的评价指标）
# 为了代码清晰可读以及整洁，单独重写了一个test函数，尽管许多内容与valid_one_epoch重复
# 另一个代价是许多类需要重新实例化
# *******************************************************************************
@torch.no_grad()
def test(model, criterion1, criterion2, test_dl, device, checkpoint_path):
    model.eval()
    num_examples = 0
    y_preds_1 = []
    y_preds_2 = []
    y_preds_3 = []
    y_truths_1 = []
    y_truths_2 = []
    y_truths_3 = []

    '''加载模型权重，开始测试'''
    model.load_state(torch.load(checkpoint_path))
    bar = tqdm(enumerate(test_dl), total=len(test_dl))
    for i, batch in bar:
        batch_inputs = batch[0].to(device)
        batch_labels_1 = batch[1].to(device)
        batch_labels_2 = batch[2].to(device)
        batch_labels_3 = batch[3].to(device)

        '''获取模型输出，并计算损失'''
        out1, out2, out3 = model(batch_inputs)

        '''将真实标签存到列表中'''
        y_truths_1.append(batch_labels_1.cpu().detach().numpy())
        y_truths_2.append(batch_labels_2.cpu().detach().numpy())
        y_truths_3.append(batch_labels_3.cpu().detach().numpy())

        '''将预测标签存到列表中'''
        batch_preds_1 = out1.argmax(dim=-1)
        y_preds_1.append(batch_preds_1.cpu().detach().numpy())
        y_preds_2.append(batch_preds_2.cpu().detach().numpy())
        batch_preds_3 = out3.argmax(dim=-1)
        y_preds_3.append(batch_preds_3.cpu().detach().numpy())

    '''计算整个测试集上的：Accuracy, MicF1, MacF1'''
    y_preds_1 = np.concatenate(y_preds_1)
    y_truths_1 = np.concatenate(y_truths_1)
    Accuracy_1 = metrics.accuracy_score(y_truths_1, y_preds_1)
    MicF1_1 = metrics.f1_score(y_truths_1, y_preds_1, average='micro')
    MacF1_1 = metrics.f1_score(y_truths_1, y_preds_1, average='macro')

    y_preds_2 = np.concatenate(y_preds_2)
    y_truths_2 = np.concatenate(y_truths_2)
    Accuracy_2 = metrics.accuracy_score(y_truths_2, y_preds_2)
    MicF1_2 = metrics.f1_score(y_truths_2, y_preds_2, average='micro')
    MacF1_2 = metrics.f1_score(y_truths_2, y_preds_2, average='macro')

    y_preds_3 = np.concatenate(y_preds_3)
    y_truths_3 = np.concatenate(y_truths_3)
    Accuracy_3 = metrics.accuracy_score(y_truths_3, y_preds_3)
    MicF1_3 = metrics.f1_score(y_truths_3, y_preds_3, average='micro')
    MacF1_3 = metrics.f1_score(y_truths_3, y_preds_3, average='macro')
    return Accuracy_1, MicF1_1, MacF1_1, Accuracy_2, MicF1_2, MacF1_2, Accuracy_3, MacF1_3, MicF1_3

if __name__ == '__main__':
    '''实例化： 1.模型 2.设备 3.损失函数'''
    model = MyModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''来测试一下吧'''
    test_dl = get_test_dataloader(r'./data_cail2018/test.json', test_batch_size=8)
    checkpoint_path = r'./results/saved_checkpoint.pth'
    AAccuracy_1, MicF1_1, MacF1_1, Accuracy_2, MicF1_2, MacF1_2, Accuracy_3, MacF1_3, MicF1_3 = \
        test(model, criterion, test_dl, device, checkpoint_path)
    print(f'Accuracy_1={Accuracy_1}, MicF1_1={MicF1_1}, MacF1_1={MacF1_1}'
          f'Accuracy_2={Accuracy_2}, MicF1_2={MicF1_2}, MacF1_2={MacF1_2}'
          f'Accuracy_3={Accuracy_3}, MicF1_2={MicF1_2}, MacF1_3={MacF1_3}')

