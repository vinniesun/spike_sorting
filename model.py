import torch
import torch.nn as nn
import snntorch as snn
from snntorch.surrogate import atan
from einops import repeat

from neurons import ABRF, DBRF, DTLIF

from typing import Tuple, Union, List, Optional

# from BRF.grad_functions import StepDoubleGaussianGrad

raf_interval_to_b_mapping = {
    4: 2,
    5: 2,
    6: 2,
    7: 3,
    8: 3,
    9: 3,
    10: 4,
    11: 4,
    12: 4,
    13: 5,
    14: 5,
    15: 5,
    16: 6,
    17: 6,
    18: 6,
    19: 7,
    20: 7,
    21: 7,
    22: 8,
    23: 8,
    24: 8,
    25: 8,
}

@torch.jit.script
def step(x: torch.Tensor) -> torch.Tensor:
    #
    # x.gt(0.0).float()
    # is slightly faster (but less readable) than
    # torch.where(x > 0.0, 1.0, 0.0)
    #
    return x.gt(0.0).float()

@torch.jit.script
def gaussian(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    return (1 / (sigma * torch.sqrt(2 * torch.tensor(torch.pi)))) * torch.exp(
        -((x - mu) ** 2) / (2.0 * (sigma ** 2))
    )

class StepDoubleGaussianGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors

        p = 0.15
        scale = 6.
        len = 0.5

        sigma1 = len
        sigma2 = scale * len

        gamma = 0.5
        dfd = (1. + p) * gaussian(x, mu=0., sigma=sigma1) - 2. * p * gaussian(x, mu=0., sigma=sigma2)

        return grad_output * dfd * gamma

def step_double_gaussian():
    def inner(x):
        return StepDoubleGaussianGrad.apply(x)
    return inner

class SigmoidGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x.gt(0.0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        alpha = ctx.alpha

        sig = torch.sigmoid(alpha * x)
        dfd = alpha * sig * (1 - sig)

        return grad_output * dfd, None

def sigmoid_grad(alpha=1.0):
    def inner(x):
        return SigmoidGrad.apply(x, alpha)
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

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

"""
    Based on the paper: Long Short-Term Memory Spiking Networks and Their Applications
    (https://arxiv.org/pdf/2007.04779)
"""
class SpikingLSTMCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        surrogate_fn1=sigmoid_grad(),
        surrogate_fn2=atan(0.8),
        threshold1=1.0,
        threshold2=1.0,
        learn_threshold=False,
        bias: bool=True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.surrogate_fn1 = surrogate_fn1
        if surrogate_fn2 is None:
            self.surrogate_fn2 = surrogate_fn1
        else:
            self.surrogate_fn2 = surrogate_fn2

        if learn_threshold:
            self.threshold1 = nn.Parameter(torch.tensor(threshold1))
            self.threshold2 = nn.Parameter(torch.tensor(threshold2))
        else:
            self.register_buffer("threshold1", torch.tensor(threshold1))
            self.register_buffer("threshold2", torch.tensor(threshold2))

        self.w_i = nn.Linear(input_dim, hidden_dim*4, bias=bias)
        self.w_h = nn.Linear(hidden_dim, hidden_dim*4, bias=bias)

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h_t = torch.zeros(batch_size, self.hidden_dim, device=self.w_i.weight.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=self.w_i.weight.device)

        return h_t, c_t

    def forward(self, x: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = self.w_i(x) + self.w_h(h_t)

        gate_i, gate_f, gate_g, gate_o = gates.chunk(4, dim=1)

        i_t = self.surrogate_fn1(gate_i - self.threshold1)
        f_t = self.surrogate_fn1(gate_f - self.threshold1)
        g_t = self.surrogate_fn2(gate_g - self.threshold2)
        o_t = self.surrogate_fn1(gate_o - self.threshold1)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * c_t

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
        #     input_size=input_dim + hidden_size, # recurrence dim: input_dim + hidden_size. non-recurrent dim: input_dim
        #     hidden_size=hidden_size,
        #     bias=True, # Don't include bias cause we dont want membrane potential to change when input = 0
        #     threshold=0.5, # starting at 1.0 seems to high.
        #     spike_grad=atan(), # step_double_gaussian()/atan()
        #     learn_threshold=True,
        #     reset_mechanism="subtract"
        # )

        self.lstm = nn.LSTMCell(
            input_size=input_dim, # recurrence dim: input_dim + hidden_size. non-recurrent dim: input_dim
            hidden_size=hidden_size,
            bias=True
        )
        # self.lstm_reverse = nn.LSTMCell(
        #     input_size=input_dim, # recurrence dim: input_dim + hidden_size. non-recurrent dim: input_dim
        #     hidden_size=hidden_size,
        #     bias=True
        # )

        # self.lstm = SpikingLSTMCell(
        #     input_dim=input_dim,
        #     hidden_dim=hidden_size,
        #     surrogate_fn1=atan(0.8),
        #     surrogate_fn2=atan(0.8),
        #     threshold1=0.5,
        #     threshold2=0.5,
        #     learn_threshold=True,
        #     bias=True
        # )

        self.pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1) # padding=1 to keep the same length as input
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(hidden_size, num_classes)
        # self.fc1 = nn.Linear(hidden_size*2, num_classes)
        self.lif1 = snn.Leaky(
            # beta=0.9 * torch.ones(num_classes),
            # threshold=0.2 * torch.ones(num_classes),
            beta=0.9,
            threshold=0.2,
            reset_mechanism="subtract",
            spike_grad=atan(), # step_double_gaussian()/atan()
            learn_beta=True,
            learn_threshold=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)    # for nn.LSTMCell
        c_t = torch.zeros_like(h_t)                                         # for nn.LSTMCell

        mem2 = self.lif1.reset_mem()
        
        spk2_hist = []
        for i in range(seq_len):
            current_input = x[:, i].unsqueeze(-1) # only use the input, not the previous hidden state

            h_t, c_t = self.lstm(current_input, (h_t, c_t)) # for nn.LSTMCell
            curr = self.fc1(self.relu(h_t))
            spk2, mem2 = self.lif1(curr, mem2)

            spk2_hist.append(spk2)

        return torch.stack(spk2_hist, dim=0)

class RAFSpikingLSTMSpikeSorter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_classes: int,
        learn_raf_params: bool = True,
        learn_dtlif_params: bool = True,
        dt: float = 1/24000,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.learn_raf_params = learn_raf_params
        self.learn_dtlif_params = learn_dtlif_params
        self.dt = dt

        self.rafs, self.dtlif = self.init_feature_extractors()

        self.lstm = nn.LSTMCell(
            input_size=self.rafs.dual_omegas.shape[0] + self.dtlif.beta.shape[0], # recurrence dim: input_dim + hidden_size. non-recurrent dim: input_dim
            hidden_size=hidden_size,
            bias=True
        )

        self.pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1) # padding=1 to keep the same length as input
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(hidden_size, num_classes)
        # self.fc1 = nn.Linear(hidden_size*2, num_classes)
        self.lif1 = snn.Leaky(
            # beta=0.9 * torch.ones(num_classes),
            # threshold=0.2 * torch.ones(num_classes),
            beta=0.9,
            threshold=0.2,
            reset_mechanism="subtract",
            spike_grad=atan(), # step_double_gaussian()/atan()
            learn_beta=True,
            learn_threshold=True,
        )

    def init_feature_extractors(self):
        interval1 = torch.arange(start=4, end=20, step=2, dtype=torch.float32)
        interval2 = torch.arange(start=4, end=25, step=2, dtype=torch.float32)

        raf_omega_interval = torch.cartesian_prod(interval1, interval2)
        raf_omegas = torch.pi / (raf_omega_interval / 24000)

        raf_bs = torch.tensor([
            [
                raf_omegas[i, 0] / raf_interval_to_b_mapping[raf_omega_interval[i, 0].item()], 
                raf_omegas[i, 1] / raf_interval_to_b_mapping[raf_omega_interval[i, 1].item()]
            ] for i in range(raf_omegas.shape[0])
        ])

        initial_dv = 4.1667e-5
        
        k_threshold1 = 2
        k_threshold2 = 3.0  # original is 1.9
        threshold1 = k_threshold1 * initial_dv # original value: 6e-5
        threshold2 = k_threshold2 * initial_dv # original value: 7.8e-5
        raf_thresholds = torch.tensor([threshold1, threshold2], dtype=torch.float32)
        raf_thresholds = repeat(raf_thresholds, 't -> b t', b=raf_omegas.shape[0]).clone()

        raf_q_coeff = torch.tensor([1e-1, 1e-3], dtype=torch.float32)
        raf_q_coeff = repeat(raf_q_coeff, 't -> b t', b=raf_omegas.shape[0]).clone()
        
        betas = torch.linspace(start=0.01, end=1, steps=raf_omegas.shape[0], dtype=torch.float32)
        pos_thresholds = torch.ones(raf_omegas.shape[0], dtype=torch.float32) * 3.0
        neg_thresholds = torch.ones(raf_omegas.shape[0], dtype=torch.float32) * -3.0

        rafs = DBRF(
            input_dim=raf_omegas.shape[0],
            dual_omegas=raf_omegas,
            dual_bs=raf_bs,
            dual_threshold=raf_thresholds,
            dual_q_coeff=raf_q_coeff,
            dt=self.dt,
            learn_omega=self.learn_raf_params,
            learn_b=self.learn_raf_params,
            learn_dual_threshold=self.learn_raf_params
        )

        dtlif = DTLIF(
            beta=betas,
            pos_threshold=pos_thresholds,
            neg_threshold=neg_thresholds,
            learn_beta=self.learn_dtlif_params,
            learn_threshold=self.learn_dtlif_params,
            reset_mechanism="subtract"
        )

        return rafs, dtlif

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        hidden_states = self.rafs.init_hidden_state(batch_size=batch_size)
        dt_mem = self.dtlif.reset_mem()

        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)    # for nn.LSTMCell
        c_t = torch.zeros_like(h_t)                                         # for nn.LSTMCell

        mem2 = self.lif1.reset_mem()
        
        spk2_hist = []
        for i in range(seq_len):
            raf_curr = torch.clamp(x[:, i].unsqueeze(-1), min=-1.0, max=1.0)
            raf_spk, u, v, q, use_t1 = self.rafs(raf_curr, hidden_states)

            dt_spk, dt_mem = self.dtlif(x[:, i].unsqueeze(-1), dt_mem)

            combined_spks = torch.cat((raf_spk, dt_spk), dim=1) # Shape: (batch_size, # of RAF neurons + # of dtlif)

            h_t, c_t = self.lstm(combined_spks, (h_t, c_t)) # for nn.LSTMCell
            curr = self.fc1(self.relu(h_t))
            spk2, mem2 = self.lif1(curr, mem2)

            spk2_hist.append(spk2)

            hidden_states = raf_spk, u, v, q, use_t1

        return torch.stack(spk2_hist, dim=0)

class RAFAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size

        # encoder contains: 
        # 1. rafs that loops over the input sequence
        # 2. linear layer/lstm layer that generates the embedding from the raf's output spikes
        # 3. linear layer that generates the final latent embedding.
        self.encoder = nn.Linear(input_dim, hidden_size)

        # decoder contains:
        # 1. linear layer that decodes the final latent embedding
        # 2. linear layer that reconstructs the input sequence from the decoded embedding
        self.decoder = nn.Linear(hidden_size, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class DBRFDTLIFModel(nn.Module):
    def __init__(
            self, 
            dbrf_input_dim: int,
            dtlif_input_dim: int,
            dual_omegas: torch.Tensor,
            dual_bs: torch.Tensor,
            dual_threshold: torch.Tensor,
            dual_q_coeff: torch.Tensor,
            dt: float=1/24000,
            learn_dual_threshold: bool=False,
            num_classes: int=3,
            beta: Union[torch.Tensor, float]=0.9,
            pos_threshold: Union[torch.Tensor, float]=1.0,
            neg_threshold: Union[torch.Tensor, float]=-1.0,
            learn_beta: bool=False,
            learn_dtlif_threshold: bool=False,
            reset_mechanism: str="subtract"
    ):
        super().__init__()

        self.dbrf_input_dim = dbrf_input_dim
        self.dtlif_input_dim = dtlif_input_dim

        if dbrf_input_dim > 0:
            assert dbrf_input_dim == dual_omegas.shape[0], "input_dim does not match dual_omegas' first dimension"
            assert dbrf_input_dim == dual_bs.shape[0], "input_dim does not match dual_bs' first dimension"
            assert dbrf_input_dim == dual_threshold.shape[0], "input_dim does not match dual_threshold's first dimension"

            self.rafs = DBRF(
                input_dim=dbrf_input_dim,
                dual_omegas=dual_omegas,
                dual_bs=dual_bs,
                dual_threshold=dual_threshold,
                dual_q_coeff=dual_q_coeff,
                dt=dt,
                learn_omega=True,
                learn_b=True,
                learn_dual_threshold=learn_dual_threshold
            )

        if dtlif_input_dim > 0:
            if type(beta) is torch.Tensor:
                assert dtlif_input_dim == beta.shape[0], "input_dim does not match beta's first dimension"
            if type(pos_threshold) is torch.Tensor:
                assert dtlif_input_dim == pos_threshold.shape[0], "input_dim does not match pos_threshold's first dimension"
            if type(neg_threshold) is torch.Tensor:
                assert dtlif_input_dim == neg_threshold.shape[0], "input_dim does not match neg_threshold's first dimension"

            if type(beta) is float:
                beta = torch.tensor([beta] * dtlif_input_dim, dtype=torch.float32)
            if type(pos_threshold) is float:
                pos_threshold = torch.tensor([pos_threshold] * dtlif_input_dim, dtype=torch.float32)
            if type(neg_threshold) is float:
                neg_threshold = torch.tensor([neg_threshold] * dtlif_input_dim, dtype=torch.float32)

            self.dtlif = DTLIF(
                beta=beta,
                pos_threshold=pos_threshold,
                neg_threshold=neg_threshold,
                learn_beta=learn_beta,
                learn_threshold=learn_dtlif_threshold,
                reset_mechanism=reset_mechanism,
            )

        self.fc1 = nn.Linear(dbrf_input_dim + dtlif_input_dim, num_classes, bias=True)
        self.lif1 = snn.Leaky(beta=0.9, threshold=0.8, learn_beta=True, learn_threshold=True, spike_grad=step_double_gaussian(), reset_mechanism="subtract")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        bs, seq_len = x.shape

        if self.dbrf_input_dim > 0:
            hidden_states = self.rafs.init_hidden_state(batch_size=bs)

        if self.dtlif_input_dim > 0:
            dt_mem = self.dtlif.reset_mem()

        mem1 = self.lif1.reset_mem()

        spk_hist, mem_hist = [], []
        for i in range(seq_len):
            curr = torch.clamp(x[:, i].unsqueeze(-1), min=-1.0, max=1.0)      # Shape: (batch_size, 1). This is just to take into account of the polarity
            # curr = x[:, i].unsqueeze(-1)    # Shape: (batch_size, 1)            # This is taking into account of both polarity and magnitude

            if self.dbrf_input_dim > 0:
                raf_spk, u, v, q, use_t1 = self.rafs(curr, hidden_states) # Output Shape: (batch_size, # of RAF neurons)

            if self.dtlif_input_dim > 0:
                dt_spk, dt_mem = self.dtlif(x[:, i].unsqueeze(-1), dt_mem) # Output Shape: (batch_size, 1)

            if self.dbrf_input_dim > 0 and self.dtlif_input_dim > 0:
                combined_spks = torch.cat((raf_spk, dt_spk), dim=1) # Shape: (batch_size, # of RAF neurons + 1)
            elif self.dbrf_input_dim > 0:
                combined_spks = raf_spk
            elif self.dtlif_input_dim > 0:
                combined_spks = dt_spk

            out1 = self.fc1(combined_spks)
            spk1, mem1 = self.lif1(out1, mem1)

            # out2 = self.fc2(spk1)
            # spk2, mem2 = self.lif2(out2, mem2)

            spk_hist.append(spk1)
            mem_hist.append(mem1)

            hidden_states = raf_spk, u, v, q, use_t1

        return torch.stack(spk_hist), torch.stack(mem_hist)

if __name__ == "__main__":
    net = SpikingLSTMSpikeSorter(input_dim=1, hidden_size=20, num_classes=3)

    with torch.no_grad():
        x = torch.rand(32, 100) # batch_size=32, seq_len=100

        slstm_spk_hist, spk2_hist = net(x)

    print(slstm_spk_hist.shape) # should be (100, 32, 20)
    print(spk2_hist.shape) # should be (100, 32, 3)