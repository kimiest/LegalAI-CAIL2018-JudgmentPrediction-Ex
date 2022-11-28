import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

@torch.no_grad()
def valid_one_epoch(model, criterion1, criterion2, valid_dl, device, epoch):
    model.eval()
    num_examples = 0
    total_correct_1 = 0
    total_loss_1 = 0.0
    total_loss_2 = 0.0
    total_loss_3 = 0.0

    bar = tqdm(enumerate(valid_dl), total=len(valid_dl))
    for i, batch in bar:
        batch_inputs = batch[0].to(device)
        batch_labels_1 = batch[1].to(device)
        batch_labels_2 = batch[2].to(device)
        batch_labels_3 = batch[3].to(device)

        '''获取模型输出并计算损失'''
        out1, out2, out3 = model(batch_inputs)
        loss1 = criterion1(out, batch_labels_1)
        loss2 = criterion2(out2, batch_labels_2)
        loss3 = criterion1(out3, batch_labels_3)

        '''计算总损失'''
        total_loss_1 += loss1.item()
        total_loss_2 += loss2.item()
        total_loss_3 += loss3.item()

        '''计算预测正确的罪名总数'''
        batch_preds_1 = out1.argmax(dim=-1)
        correct_1 = (batch_preds_1 == batch_labels_1).sum().item()
        total_correct_1 += correct_1

        '''计算准确率和平均损失，并显示在tqdm中'''
        num_examples += len(batch_labels_1)
        accuracy_1 = total_correct_1 / num_examples
        avg_loss_1 = total_loss_1 / num_examples
        avg_loss_2 = total_loss_2 / num_examples
        avg_loss_3 = total_loss_3 / num_examples
        bar.set_description(f'epoch={epoch}')
        bar.set_postfix(accuracy_task1=accuracy_1, loss1=avg_loss_1, loss2=avg_loss_2, loss3=avg_loss_3)

    return avg_loss_1 + avg_loss_2 + avg_loss_3, accuracy_1


