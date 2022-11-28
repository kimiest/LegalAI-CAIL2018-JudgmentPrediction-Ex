from transformers import AutoModel
import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained('xcjthu/Lawformer')
        self.fc1 = nn.Linear(768, 196)
        self.fc2 = nn.Linear(768, 183)
        self.fc3 = nn.Linear(768, 14)

    def forward(self, batch_inputs):
        cls = self.bert(input_ids=batch_inputs['input_ids'].squeeze(),
                           attention_mask=batch_inputs['attention_mask'].squeeze()).pooler_output
        out1 = self.fc1(cls)
        out2 = torch.sigmoid(self.fc2(cls))
        out3 = self.fc3(cls)
        return out1, out2, out3

