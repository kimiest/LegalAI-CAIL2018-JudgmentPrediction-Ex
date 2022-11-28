from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score

# **********************
# 执行一个epoch的训练
# **********************
def train_one_epoch(model, optimizer, scheduler, criterion1, criterion2, train_dl, device, epoch):
    model.train()
    num_examples = 0
    total_loss_1 = 0
    total_loss_2 = 0
    total_loss_3 = 0
    total_correct_1 = 0

    bar = tqdm(enumerate(train_dl), total=len(train_dl))
    for i, batch in bar:
        batch_inputs = batch[0].to(device)
        batch_labels_1 = batch[1].to(device)
        batch_labels_2 = batch[2].to(device)
        batch_labels_3 = batch[3].to(device)

        '''获取模型输出并计算损失'''
        out1, out2, out3 = model(batch_inputs)
        loss1 = criterion1(out1, batch_labels_1)
        loss2 = criterion2(out2, batch_labels_2.float())
        loss3 = criterion1(out3, batch_labels_3)
        loss = 0.33*loss1 + 0.33*loss2 + 0.33*loss3

        '''1.清空梯度 2.反向传播求梯度 3.优化参数'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''计算预测正确的罪名总数'''
        batch_preds_1 = out1.argmax(dim=-1)
        correct_1 = (batch_preds_1 == batch_labels_1).sum().item()
        total_correct_1 += correct_1

        '''计算总损失'''
        total_loss_1 += loss1.item()
        total_loss_2 += loss2.item()
        total_loss_3 += loss3.item()

        '''计算准确率和平均损失，并显示在tqdm中'''
        num_examples += len(batch_labels_1)
        accuracy_1 = total_correct_1 / num_examples
        avg_loss_1 = total_loss_1 / num_examples
        avg_loss_2 = total_loss_2 / num_examples
        avg_loss_3 = total_loss_3 / num_examples
        bar.set_description(f'epoch={epoch}')
        bar.set_postfix(accuracy_task1=accuracy_1, loss1=avg_loss_1, loss2=avg_loss_2, loss3=avg_loss_3)

        '''每300个batch，调整一次学习率'''
        if i % 300 == 0:
            if scheduler is not None:
                scheduler.step()

    '''返回当前epoch的平均损失和训练准确率'''
    return avg_loss1 + avg_loss_2 + avg_loss_3, accuracy_1
