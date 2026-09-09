import numpy as np
import torch
import torch.nn as nn

from typing import List, Tuple, Union

class LSTMModel(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()

        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        # self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len = x.shape

        memory_usage = 0

        h_t = torch.zeros(bs, self.lstm.hidden_size, dtype=x.dtype, device=x.device)
        c_t = torch.zeros(bs, self.lstm.hidden_size, dtype=x.dtype, device=x.device)

        memory_usage += h_t.element_size() * h_t.nelement()
        memory_usage += c_t.element_size() * c_t.nelement()

        memory_usage += self.lstm.weight_ih.element_size() * self.lstm.weight_ih.nelement()
        memory_usage += self.lstm.weight_hh.element_size() * self.lstm.weight_hh.nelement()
        memory_usage += self.lstm.bias_ih.element_size() * self.lstm.bias_ih.nelement()
        memory_usage += self.lstm.bias_hh.element_size() * self.lstm.bias_hh.nelement()

        memory_usage += self.fc1.weight.element_size() * self.fc1.weight.nelement()
        memory_usage += self.fc1.bias.element_size() * self.fc1.bias.nelement()

        for t in range(seq_len):
            x_t = x[:, t].unsqueeze(-1)
            h_t, c_t = self.lstm(x_t, (h_t, c_t))

            out = self.fc1(h_t)

        return out, memory_usage

class Conv1DModel(nn.Module):
    def __init__(
            self, 
            in_channel: list[int],
            filters: List[int],
            kernel_sizes: List[int],
            strides: List[int],
            fc_input_dim: int,
            num_classes: int,
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv1d(
                in_channels=in_channel[i],
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i]
            ) for i in range(len(filters))]
        )
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, num_chan, seq_len = x.shape

        memory_usage = 0

        for i in range(len(self.convs)):
            x = self.relu(self.convs[i](x))

            memory_usage += self.convs[i].weight.element_size() * self.convs[i].weight.nelement()
            memory_usage += self.convs[i].bias.element_size() * self.convs[i].bias.nelement()
            memory_usage += x.element_size() * x.nelement()

        x = x.view(bs, -1)
        x = self.fc1(x)
        memory_usage += self.fc1.weight.element_size() * self.fc1.weight.nelement()
        memory_usage += self.fc1.bias.element_size() * self.fc1.bias.nelement()

        return x, memory_usage

if __name__ == "__main__":
    lstm_net = LSTMModel(
        input_dim=1,
        hidden_dim=32,
        num_classes=3
    )

    conv_net = Conv1DModel(
        in_channel=[1, 16], # 1, 48 -> 1, 46
        filters=[16, 16],
        kernel_sizes=[3, 3],
        strides=[1, 1],
        fc_input_dim=16 * 60,
        num_classes=3
    )

    dummy_input = torch.ones(1, 64) # batch_size, seq_len

    lstm_output, lstm_memory_usage = lstm_net(dummy_input)
    cnn_output, cnn_memory_usage = conv_net(dummy_input.unsqueeze(1)) # add channel dimension

    print(f"LSTM Memory Usage: {lstm_memory_usage} bytes")
    print(f"CNN Memory Usage: {cnn_memory_usage} bytes")