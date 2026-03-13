import torch
from BRF.neurons import *
from BRF.grad_functions import *

class SimpleResRNN(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            pruning: bool = False,
            adaptive_omega_a: float = 5.,
            adaptive_omega_b: float = 10.,
            adaptive_b_offset_a: float = 0.,
            adaptive_b_offset_b: float = 1.,
            out_adaptive_tau_mem_mean: float = 20.,
            out_adaptive_tau_mem_std: float = 5.,
            n_last: int = 1,
            mask_prob: float = 0.,
            sub_seq_length: int = 0,
            hidden_bias: bool = False,
            output_bias: bool = False,
            label_last: bool = False,
            dt: float = 0.01,
    ) -> None:
        super(SimpleResRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.sub_seq_length = sub_seq_length

        self.label_last = label_last
        self.n_last = n_last

        self.mask_prob = mask_prob

        self.adaptive_omega_a = adaptive_omega_a
        self.adaptive_omega_b = adaptive_omega_b

        self.adaptive_b_offset_a = adaptive_b_offset_a
        self.adaptive_b_offset_b = adaptive_b_offset_b

        self.out_adaptive_tau_mem_mean = out_adaptive_tau_mem_mean
        self.out_adaptive_tau_mem_std = out_adaptive_tau_mem_std

        self.hidden = BRFCell(
            input_size=input_size + hidden_size,  # only input_size for non-recurrency
            layer_size=hidden_size,
            bias=hidden_bias,
            mask_prob=mask_prob,
            adaptive_omega=True,
            adaptive_omega_a=adaptive_omega_a,
            adaptive_omega_b=adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=adaptive_b_offset_a,
            adaptive_b_offset_b=adaptive_b_offset_b,
            dt=dt,
            pruning=pruning
        )

        self.out = LICell(
            input_size=hidden_size,
            layer_size=output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=out_adaptive_tau_mem_std,
            bias=output_bias,
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor]:

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = list()

        num_spikes = torch.tensor(0.).to(x.device)

        hidden_z = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_v = torch.zeros_like(hidden_z)
        hidden_q = torch.zeros_like(hidden_z)

        out_u = torch.zeros((batch_size, self.output_size)).to(x.device)

        for t in range(sequence_length):

            input_t = x[t]

            hidden = hidden_z, hidden_u, hidden_v, hidden_q

            hidden_z, hidden_u, hidden_v, hidden_q = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),  # input_t for non-recurrency
                hidden
            )

            # SOP
            num_spikes += hidden_z.sum()

            out_u = self.out(hidden_z, out_u)

            # Records outputs with sub_seq_length delay
            if t >= self.sub_seq_length:
                outputs.append(out_u)

        outputs = torch.stack(outputs)

        if self.label_last:
            outputs = outputs[-self.n_last:, :, :]

        return outputs, ((hidden_z, hidden_u), out_u), num_spikes