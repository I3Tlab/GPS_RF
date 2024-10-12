import torch.nn as nn
class spatial_time_convertor_real(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, dropout=0):
        super(spatial_time_convertor_real, self).__init__()
        self.Linear1_1 = nn.Linear(in_channel, int(in_channel/2), bias)
        self.BN1d_1 = nn.BatchNorm1d(int(in_channel/2))
        self.Linear1_2 = nn.Linear(int(in_channel/2), int(in_channel/4), bias)
        self.BN1d_2 = nn.BatchNorm1d(int(in_channel / 4))
        self.Linear1_3 = nn.Linear(int(in_channel/4), int(in_channel/8), bias)
        self.BN1d_3 = nn.BatchNorm1d(int(in_channel / 8))

        self.rf_real = nn.Linear(int(in_channel/8),out_channel, bias)
        self.leaky_relu = nn.LeakyReLU()
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.out_channel = out_channel
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x_1 = self.Linear1_1(x)
        x_1 = self.leaky_relu(x_1)
        x_1 = self.Linear1_2(x_1)
        x_1 = self.leaky_relu(x_1)
        x_1 = self.Linear1_3(x_1)
        if self.dropout_rate  > 0:
            x_1 = self.dropout(x_1)
        rf_real = self.rf_real(x_1)
        return rf_real

class spatial_time_convertor_imag(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, dropout=0):
        super(spatial_time_convertor_imag, self).__init__()
        self.Linear1_1 = nn.Linear(in_channel, int(in_channel/2), bias)

        self.Linear1_2 = nn.Linear(int(in_channel/2), int(in_channel/4), bias)
        self.Linear1_3 = nn.Linear(int(in_channel/4), int(in_channel/8), bias)

        self.rf_imag_1 = nn.Linear(int(in_channel/8),int(in_channel/16), bias)
        self.rf_imag_2 = nn.Linear(int(in_channel / 16), int(in_channel / 32), bias)
        self.rf_imag_3 = nn.Linear(int(in_channel / 32), out_channel, bias)
        self.leaky_relu = nn.LeakyReLU()
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.out_channel = out_channel
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x_1 = self.Linear1_1(x)
        x_1 = self.sigmoid(x_1)

        x_1 = self.Linear1_2(x_1)
        x_1 = self.sigmoid(x_1)

        x_1 = self.Linear1_3(x_1)
        if self.dropout_rate  > 0:
            x_1 = self.dropout(x_1)

        rf_imag = self.rf_imag_1(x_1)
        rf_imag = self.rf_imag_2(rf_imag)
        rf_imag = self.rf_imag_3(rf_imag)
        return rf_imag