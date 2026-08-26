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

        h_t = torch.zeros(bs, self.lstm.hidden_size, dtype=x.dtype, device=x.device)
        c_t = torch.zeros(bs, self.lstm.hidden_size, dtype=x.dtype, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t].unsqueeze(-1)
            h_t, c_t = self.lstm(x_t, (h_t, c_t))

            out = self.fc1(h_t)

        return out

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

        for i in range(len(self.convs)):
            x = self.relu(self.convs[i](x))

        x = x.view(bs, -1)
        x = self.fc1(x)

        return x

if __name__ == "__main__":
    lstm_net = LSTMModel(
        input_dim=1,
        hidden_dim=32,
        num_classes=3
    )

    conv_net = Conv1DModel(
        in_channel=[1, 16], # 1, 48 -> 1, 46
        filters=[16, 32],
        kernel_sizes=[3, 3],
        strides=[1, 1],
        fc_input_dim=32 * 44,
        num_classes=3
    )

    dummy_input = torch.ones(1, 48) # batch_size, seq_len

    
