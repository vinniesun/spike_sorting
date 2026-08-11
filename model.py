import torch
import torch.nn as nn

import snntorch as snn

from typing import Union, Tuple, List, Optional

from feature_extractor_neurons import (
    DBRF,
    DTLIF,
    BMLIF
)

from BRF.grad_functions import StepDoubleGaussianGrad

def step_double_gaussian():
    def inner(x):
        return StepDoubleGaussianGrad.apply(x)
    return inner

class DBRFDTLIFModel(nn.Module):
    def __init__(
            self, 
            dbrf_input_dim: int,
            dtlif_input_dim: int,
            dual_omegas: torch.Tensor,
            dual_bs: torch.Tensor,
            dual_threshold: torch.Tensor,
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
            # curr = torch.clamp(x[:, i].unsqueeze(-1), min=-1.0, max=1.0)      # Shape: (batch_size, 1). This is just to take into account of the polarity
            curr = x[:, i].unsqueeze(-1)    # Shape: (batch_size, 1)            # This is taking into account of both polarity and magnitude

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