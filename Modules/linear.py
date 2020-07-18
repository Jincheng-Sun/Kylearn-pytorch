import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, d_features, seq_length, d_hid, d_out):
        super(LinearClassifier, self).__init__()

        self.d_features = d_features
        self.maxpool = torch.nn.MaxPool1d(seq_length, stride=1, padding=0)
        self.fc1 = nn.Linear(d_features, d_hid)
        self.activation = nn.functional.leaky_relu
        self.fc2 = nn.Linear(d_hid, d_out)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.maxpool(x)
        x = x.view(-1, self.d_features)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x