
import torch
import torch.nn as nn
import einops

from snntorch.surrogate import atan

from BRF.grad_functions import *
from typing import Tuple, Union, List, Optional

class BRF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        omegas: torch.Tensor,
        bs: torch.Tensor,
        threshold: torch.Tensor,
        dt: float = 1/24000,
        learn_omega: bool = False,
        learn_b: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim

        # assert self.input_dim == omegas.shape[0], "input_dim does not match omegas' first dimension"
        # assert self.input_dim == bs.shape[0], "input_dim does not match bs' first dimension"
        # assert self.input_dim == threshold.shape[0], "input_dim does not match threshold's first dimension"

        self.learn_omega = learn_omega
        if learn_omega:
            self.omegas = nn.Parameter(omegas)
        else:
            self.register_buffer('omegas', omegas)  # Dimension is (num_of_raf_neurons, 2)

        self.learn_b = learn_b
        if learn_b:
            self.bs = nn.Parameter(bs)
        else:
            self.register_buffer('bs', bs)  # Dimension is (num_of_raf_neurons, 2)

        self.learn_threshold = learn_threshold
        if learn_threshold:
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer('threshold', threshold)    # Dimension is (num_of_raf_neurons, 2)

        self.dt = dt
        
    def init_hidden_state(
        self,
        batch_size: int,
    ):
        hidden_z = torch.zeros((batch_size, self.input_dim), device=self.omegas.device, dtype=self.omegas.dtype)
        hidden_u = torch.zeros_like(hidden_z, device=self.omegas.device, dtype=self.omegas.dtype)
        hidden_v = torch.zeros_like(hidden_z, device=self.omegas.device, dtype=self.omegas.dtype)
        hidden_q = torch.zeros_like(hidden_z, device=self.omegas.device, dtype=self.omegas.dtype)

        return hidden_z, hidden_u, hidden_v, hidden_q

    def sustain_osc(
        self,
        omega: torch.Tensor
    ):
        return (-1 + torch.sqrt(1 - torch.square(self.dt * omega))) / self.dt

    def brf_dynamics(
        self, 
        x: torch.Tensor,  # injected current: input x weight
        u: torch.Tensor,  # membrane potential (real part)
        v: torch.Tensor,  # membrane potential (complex part)
        q: torch.Tensor,  # refractory period
        b: torch.Tensor,  # attraction to resting state
        omega: torch.Tensor,  # eigen ang. frequency of the neuron
        dt: Union[float, torch.Tensor],
    ):
        
        u_ = u + b * u * dt - omega * v * dt + x * dt
        v = v + omega * u * dt + b * v * dt
        # generate spike.
        # z = StepDoubleGaussianGrad.apply(u_ - self.threshold - q)
        z = StepDoubleGaussianGrad.apply(torch.abs(u_) - torch.abs(self.threshold) - q)

        q = q.mul(0.9) + z # Original Scale

        return z, u_, v, q

    def forward(
        self, 
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]=None,
    ):
        if state is None:
            batch_size = x.shape[0]
            state = self.init_hidden_state(batch_size)
            z, u, v, q = state
        else:
            batch_size = x.shape[0]
            z, u, v, q = state

        omega = torch.abs(self.omegas)
        p_omega = self.sustain_osc(omega)

        # divergence boundary
        b_offset = torch.abs(self.bs)
        b = p_omega - b_offset - q

        # The input sequence is looped outside of this class
        z, u, v, q = self.brf_dynamics(
            x=x,
            u=u,
            v=v,
            q=q,
            b=b,
            omega=omega,
            dt=self.dt,
        )

        return z, u, v, q

class RAF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        t1_t2_omegas: torch.Tensor,
        t1_t2_bs: torch.Tensor,
        threshold: torch.Tensor,
        dt: float = 1/24000,
        learn_omega: bool = False,
        learn_b: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim

        assert self.input_dim == t1_t2_omegas.shape[0], "input_dim does not match t1_t2_omegas' first dimension"
        assert self.input_dim == t1_t2_bs.shape[0], "input_dim does not match t1_t2_bs' first dimension"
        assert self.input_dim == threshold.shape[0], "input_dim does not match threshold's first dimension"

        self.learn_omega = learn_omega
        if learn_omega:
            # print("Learning omega")
            self.t1_t2_omegas = nn.Parameter(t1_t2_omegas)
        else:
            self.register_buffer('t1_t2_omegas', t1_t2_omegas)  # Dimension is (num_of_raf_neurons, 2)

        self.learn_b = learn_b
        if learn_b:
            # print("Learning b")
            self.t1_t2_bs = nn.Parameter(t1_t2_bs)
        else:
            self.register_buffer('t1_t2_bs', t1_t2_bs)  # Dimension is (num_of_raf_neurons, 2)

        self.learn_threshold = learn_threshold
        if learn_threshold:
            # print("Learning threshold")
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer('threshold', threshold)    # Dimension is (num_of_raf_neurons, 2)

        self.dt = dt
        
    def init_hidden_state(
        self,
        batch_size: int,
    ):
        hidden_z = torch.zeros((batch_size, self.input_dim), device=self.t1_t2_omegas.device, dtype=self.t1_t2_omegas.dtype)
        hidden_u = torch.zeros_like(hidden_z, device=self.t1_t2_omegas.device, dtype=self.t1_t2_omegas.dtype)
        hidden_v = torch.zeros_like(hidden_z, device=self.t1_t2_omegas.device, dtype=self.t1_t2_omegas.dtype)
        hidden_q = torch.zeros_like(hidden_z, device=self.t1_t2_omegas.device, dtype=self.t1_t2_omegas.dtype)

        # Initialise the use_t1 here.
        use_t1 = torch.zeros_like(hidden_z, device=self.t1_t2_omegas.device, dtype=torch.int64) # Dimension is (num_of_raf_neurons, 2)
        # if true, use t1_t2_omegas[:, 0], else use t1_t2_omegas[:, 1]. Same logic for t1_t2_bs and threshold
        # Must be of dtype int64 for gather operation later

        return hidden_z, hidden_u, hidden_v, hidden_q, use_t1

    def sustain_osc(
        self,
        omega: torch.Tensor
    ):
        return (-1 + torch.sqrt(1 - torch.square(self.dt * omega))) / self.dt

    def brf_dynamics(
        self, 
        x: torch.Tensor,  # injected current: input x weight
        u: torch.Tensor,  # membrane potential (real part)
        v: torch.Tensor,  # membrane potential (complex part)
        q: torch.Tensor,  # refractory period
        b: torch.Tensor,  # attraction to resting state
        omega: torch.Tensor,  # eigen ang. frequency of the neuron
        dt: Union[float, torch.Tensor],
        use_t1: torch.Tensor,
    ):
        
        u_ = u + b * u * dt - omega * v * dt + x * dt
        v = v + omega * u * dt + b * v * dt
        # generate spike.
        # select the correct threshold based on self.use_t1
        # z = StepDoubleGaussianGrad.apply(torch.abs(u_) - self.threshold.gather(-1, self.use_t1.unsqueeze(-1)).squeeze(-1) - q)
        current_threshold = einops.repeat(self.threshold, 'd t -> b d t', b=x.shape[0])
        z = StepDoubleGaussianGrad.apply(torch.abs(u_) - current_threshold.gather(-1, use_t1.unsqueeze(-1)).squeeze(-1) - q)

        # q = q.mul(0.9) + z # Original Scale
        q = q.mul(1e-2) + z

        # output z should retain its polarity (i.e. -1 or 1)
        # z = z * torch.sign(u_)

        return z, u_, v, q

    def forward(
        self, 
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]=None,
    ):
        if state is None:
            batch_size = x.shape[0]
            state = self.init_hidden_state(batch_size)
            z, u, v, q, use_t1 = state
        else:
            batch_size = x.shape[0]
            z, u, v, q, use_t1 = state
        # print("before use_t1: ", self.use_t1.shape)
        # print(f"before spike: {z.shape}, u: {u.shape}, v: {v.shape}, q: {q.shape}, omega: {self.t1_t2_omegas.shape}, b: {self.t1_t2_bs.shape}, input: {x.shape}")

        # self.t1_t2_omegas = einops.repeat(self.t1_t2_omegas, 'd t -> b d t', b=x.shape[0])
        # self.t1_t2_bs = einops.repeat(self.t1_t2_bs, 'd t -> b d t', b=x.shape[0])
        # self.threshold = einops.repeat(self.threshold, 'd t -> b d t', b=x.shape[0])
        # self.use_t1 = einops.repeat(self.use_t1, 'd -> b d', b=x.shape[0])

        # select omega and b based on self.use_t1
        current_t1_t2_omegas = einops.repeat(self.t1_t2_omegas, 'd t -> b d t', b=batch_size)
        omega = torch.abs(current_t1_t2_omegas.gather(-1, use_t1.unsqueeze(-1))).squeeze(-1)
        # omega = torch.abs(self.t1_t2_omegas.gather(-1, self.use_t1.unsqueeze(-1))).squeeze(-1)
        p_omega = self.sustain_osc(omega)

        current_t1_t2_bs = einops.repeat(self.t1_t2_bs, 'd t -> b d t', b=batch_size)
        b_offset = torch.abs(current_t1_t2_bs.gather(-1, use_t1.unsqueeze(-1))).squeeze(-1)
        # b_offset = torch.abs(self.t1_t2_bs.gather(-1, self.use_t1.unsqueeze(-1))).squeeze(-1)
        # divergence boundary
        b = p_omega - b_offset - q

        # The input sequence is looped outside of this class
        z, u, v, q = self.brf_dynamics(
            x=x,
            u=u,
            v=v,
            q=q,
            b=b,
            omega=omega,
            dt=self.dt,
            use_t1=use_t1,
        )

        # z = z.squeeze(-1)
        # u = u.squeeze(-1)
        # v = v.squeeze(-1)
        # q = q.squeeze(-1)

        ######### use_t1 is now a hidden state
        ######### when we select which omega, b and threshold to use, we just repeat them to include batch size dimension
        ######### Then the rest of the code remains unchanged, and we can remove those redundant helper methods
        # Update self.use_t1. Should update just use_t1 now
        # use_t1 = torch.remainder(use_t1 + z, 2).to(torch.int64)
        use_t1 = torch.remainder(use_t1 + torch.abs(z), 2).to(torch.int64) # To account for negative spikes
        # self.use_t1 = torch.remainder(self.use_t1 + z, 2).to(torch.int64)
        # print("after use_t1: ", self.use_t1.shape)
        # print(f"after spike: {z.shape}, u: {u.shape}, v: {v.shape}, q: {q.shape}, omega: {self.t1_t2_omegas.shape}, b: {self.t1_t2_bs.shape}, input: {x.shape},\
        #       p_omega: {p_omega.shape}, b: {b.shape}")
        
        return z, u, v, q, use_t1

# Used to be known DTLIF
class TwoThresholdLIF(nn.Module):
    def __init__(
        self,
        beta,
        pos_threshold,
        neg_threshold,
        learn_beta: bool=False,
        learn_threshold: bool=False,
        spike_grad=None,
        graded_spikes_factor=1.0,
        reset_mechanism="subtract"
    ):
        super().__init__()

        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta, dtype=torch.float32)
        if learn_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)
        
        if not isinstance(pos_threshold, torch.Tensor):
            pos_threshold = torch.as_tensor(pos_threshold, dtype=torch.float32)
        if not isinstance(neg_threshold, torch.Tensor):
            neg_threshold = torch.as_tensor(neg_threshold, dtype=torch.float32)
        if learn_threshold:
            self.pos_threshold = nn.Parameter(pos_threshold)
            self.neg_threshold = nn.Parameter(neg_threshold)
        else:
            self.register_buffer("pos_threshold", pos_threshold)
            self.register_buffer("neg_threshold", neg_threshold)

        if reset_mechanism == "subtract":
            self.state_function = self._base_sub
        elif reset_mechanism == "zero":
            self.state_function = self._base_zero
        elif reset_mechanism == "none":
            self.state_function = self._base_int
        else:
            raise ValueError("reset_mechanism must be one of 'subtract', 'zero', or 'none'")

        if spike_grad is None:
            self.spike_grad = atan()
        else:
            self.spike_grad = spike_grad

        self.graded_spikes_factor = graded_spikes_factor

        self._init_mem()

    def _init_mem(self):
        mem = torch.zeros(0)
        self.register_buffer("mem", mem, False)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.mem

    def mem_reset(self, mem):
        final_out = self.calc_dual_threshold(mem)

        # mem_shift = mem - final_out
        # reset = self.spike_grad(mem_shift).clone().detach()
        reset = self.spike_grad(torch.abs(final_out)).clone().detach()

        return reset

    def calc_dual_threshold(self, mem):
        pos_out = torch.where(mem > self.pos_threshold, self.pos_threshold, 0)
        neg_out = torch.where(mem < self.neg_threshold, self.neg_threshold, 0)
        return pos_out + neg_out

    def fire(self, mem):
        """Generates spike if mem > threshold.
        Returns spk."""
        
        final_out = self.calc_dual_threshold(mem)
        # mem_shift = mem - final_out
        # mem_shift1 = mem - self.pos_threshold
        # print(mem_shift, mem_shift1)
        # spk = self.spike_grad(mem_shift)
        spk = self.spike_grad(torch.abs(final_out))

        spk = spk * self.graded_spikes_factor
        spk = spk * torch.sign(final_out)

        return spk

    def _base_state_function(self, input_):
        base_fn = self.beta.clamp(0, 1) * self.mem + input_
        return base_fn

    def _base_sub(self, input_):
        final_out = self.calc_dual_threshold(self.mem)

        return self._base_state_function(input_) - self.reset * final_out

    def _base_zero(self, input_):
        self.mem = (1 - self.reset) * self.mem
        return self._base_state_function(input_)

    def _base_int(self, input_):
        return self._base_state_function(input_)

    def forward(self, x, mem=None):
        if mem is not None:
            self.mem = mem

        if not self.mem.shape == x.shape:
            self.mem = torch.zeros_like(x, device=self.mem.device)
        
        self.reset = self.mem_reset(self.mem)
        self.mem = self.state_function(x)

        spk = self.fire(self.mem)

        return spk, self.mem
    