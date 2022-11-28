import time
import copy
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW
import gc
import warnings

from models.BERT import MyModel
from utils import get_train_valid_dataloader, fetch_scheduler, set_seed
from train import train_one_epoch
from valid import valid_one_epoch


set_seed(2000)
gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()
torch.cuda.empty_cache()
warnings.filterwarnings('ignore')

'''实例化：1.设备 2.模型 3.优化器 4.损失函数 5.学习率调整策略'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel()
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.BCEWithLogitsLoss()
scheduler = fetch_scheduler(optimizer=optimizer, schedule='CosineAnnealingLR')

'''获取DataLoader类型的训练数据'''
train_dl, valid_dl = get_train_valid_dataloader(r'./data_cail2018/train.json',
                                                train_batch_size=8,
                                                valid_batch_size=8,
                                                split_k=5)

'''训练它3个epoch，保存最佳模型权重，查看最佳：1.验证损失 2.验证准确率'''
model.to(device)
best_model_state = copy.deepcopy(model.state_dict())
best_valid_loss = np.inf
best_valid_accuracy_1 = 0.0
start_time = time.time()
for epoch in range(1, 4):
    '''进行一轮训练和验证'''
    train_loss, train_accuracy_1 = train_one_epoch(model, optimizer, scheduler,
                                                   criterion1, criterion2,
                                                   train_dl, device, epoch)
    valid_loss, valid_accuracy_1 = valid_one_epoch(model,
                                                   criterion1, criterion2,
                                                   valid_dl, device, epoch)

    '''如果验证损失降低了，则：1.保存模型状态 2.更新最佳验证损失 3.更新最佳验证准确率'''
    if valid_loss <= best_valid_loss:
        print(f'best valid loss has improved ({best_valid_loss}---->{valid_loss})')
        best_valid_loss = valid_loss
        best_valid_accuracy_1 = valid_accuracy_1
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, './results/saved_checkpoint.pth')
        print('A new best model state  has saved')
end_time = time.time()
print('Training Finish !!!!!!!!')
print(f'best valid loss == {best_valid_loss}, best ACC_task1 == {best_valid_accuracy_1}')
time_cost = end_time - start_time
print(f'training cost time == {time_cost}s')