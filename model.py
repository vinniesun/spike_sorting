import torch
import torch.nn as nn
import snntorch as snn
from snntorch.surrogate import atan

from typing import Tuple, Union, List, Optional

from BRF.grad_functions import StepDoubleGaussianGrad

def step_double_gaussian():
    def inner(x):
        return StepDoubleGaussianGrad.apply(x)
    return inner

class LSTMCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool=True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.w_i = nn.Linear(input_dim, hidden_dim*4, bias=bias)
        self.w_h = nn.Linear(hidden_dim, hidden_dim*4, bias=bias)

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h_t = torch.zeros(batch_size, self.hidden_dim, device=self.w_i.weight.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=self.w_i.weight.device)

        return h_t, c_t

    def forward(self, x: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = self.w_i(x) + self.w_h(h_t)

        gate_i, gate_f, gate_g, gate_o = gates.chunk(4, dim=1)

        i_t = torch.sigmoid(gate_i)
        f_t = torch.sigmoid(gate_f)
        g_t = torch.tanh(gate_g)
        o_t = torch.sigmoid(gate_o)

        h_t = o_t * torch.tanh(c_t)
        c_t = f_t * c_t + i_t * g_t

        return h_t, c_t

class SpikingLSTMSpikeSorter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_classes: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # self.slstm = snn.SLSTM(
        #     input_size=input_dim,
        #     hidden_size=hidden_size,
        #     bias=True,
        #     threshold=0.3,
        #     spike_grad=step_double_gaussian(),
        #     learn_threshold=True,
        #     reset_mechanism="none"
        # )
        self.lstm = nn.LSTMCell(
            input_size=input_dim + hidden_size, # recurrence dim: input_dim + hidden_size. non-recurrent dim: input_dim
            hidden_size=hidden_size,
            bias=True
        )

        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.lif1 = snn.Leaky(
            beta=0.9,
            threshold=0.2,
            reset_mechanism="subtract",
            spike_grad=atan(),
            learn_beta=True,
            learn_threshold=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        # slstm_syn, slstm_mem = self.slstm.init_slstm()
        mem2 = self.lif1.reset_mem()

        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros_like(h_t)

        spk2_hist = []
        for i in range(seq_len):
            # spk1, slstm_syn, slstm_mem = self.slstm(x[:, i].unsqueeze(-1), slstm_syn, slstm_mem)
            current_input = torch.cat((x[:, i].unsqueeze(-1), h_t), dim=1) # concatenate input and previous hidden state
            h_t, c_t = self.lstm(current_input, (h_t, c_t))

            # curr = self.fc1(spk1)
            curr = self.fc1(h_t)
            spk2, mem2 = self.lif1(curr, mem2)

            spk2_hist.append(spk2)

        return torch.stack(spk2_hist, dim=0)

class SpikeDetector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        # self.threshold = threshold # for event density approach

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.lif1 = snn.Leaky(
            beta=0.9,
            threshold=1.0,
            reset_mechanism="subtract",
            learn_beta=True,
            learn_threshold=True,
        )
        self.fc2 = nn.Linear(hidden_size, output_dim)
        self.lif2 = snn.Leaky(
            beta=0.9,
            threshold=1.0,
            reset_mechanism="subtract",
            learn_beta=True,
            learn_threshold=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_features = x.shape

        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()

        spk2_hist, mem2_hist = [], []
        for i in range(seq_len):
            curr1 = self.fc1(x[:, i, :])
            spk1, mem1 = self.lif1(curr1, mem1)

            curr2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(curr2, mem2)

            spk2_hist.append(spk2)
            mem2_hist.append(mem2)

        return torch.stack(spk2_hist, dim=0), torch.stack(mem2_hist, dim=0)

if __name__ == "__main__":
    net = SpikingLSTMSpikeSorter(input_dim=1, hidden_size=20, num_classes=3)

    with torch.no_grad():
        x = torch.rand(32, 100) # batch_size=32, seq_len=100

        slstm_spk_hist, spk2_hist = net(x)

    print(slstm_spk_hist.shape) # should be (100, 32, 20)
    print(spk2_hist.shape) # should be (100, 32, 3)