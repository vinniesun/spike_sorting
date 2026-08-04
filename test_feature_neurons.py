import numpy as np
import torch
from einops import repeat
import matplotlib.pyplot as plt

import plotly.io as pio
import plotly.graph_objects as go
import plotly.tools as tls

from typing import Union

from feature_extractor_neurons import (
    DBRF,
    DTLIF,
    BMLIF
)

def plot_neuron_dynamic(
    seq_len: int,
    input_dim: int,
    batch_size: int,
    neuron: Union[DBRF, DTLIF, BMLIF],
    neuron_name: str,
    spk_train: torch.Tensor,
    plot_type: str="2D",
):
    if type(neuron) is DBRF:
        hidden_states = neuron.init_hidden_state(batch_size=spk_train.shape[0])
        out_spk_hist, u_hist, v_hist, q_hist, use_t1_hist = [], [], [], [], []

    with torch.no_grad():
        for i in range(seq_len):
            curr_input = spk_train[:, i]

            out_spk, u, v, q, use_t1 = neuron(
                curr_input.unsqueeze(-1),  # Shape: (batch_size, 1)
                hidden_states
            )

            out_spk_hist.append(out_spk)
            u_hist.append(u)
            v_hist.append(v)
            q_hist.append(q)
            use_t1_hist.append(use_t1)
            # print(f"Step {i}: input shape: {curr_input.shape}, out_spk shape: {out_spk.shape}, u shape: {u.shape}, v shape: {v.shape}, q shape: {q.shape}, use_t1 shape: {use_t1.shape}")

            hidden_states = out_spk, u, v, q, use_t1

    out_spk_hist = torch.stack(out_spk_hist, dim=0)  # Shape: (seq_len, batch_size, input_dim)
    u_hist = torch.stack(u_hist, dim=0)
    v_hist = torch.stack(v_hist, dim=0)
    q_hist = torch.stack(q_hist, dim=0)
    use_t1_hist = torch.stack(use_t1_hist, dim=0)
    # print(f"out_spk_hist shape: {out_spk_hist.shape}, u_hist shape: {u_hist.shape}, v_hist shape: {v_hist.shape}, q_hist shape: {q_hist.shape}, use_t1_hist shape: {use_t1_hist.shape}")

    if plot_type == "2D":
        fig, ax = plt.subplots(figsize=(12, 6))

        for i in range(input_dim):
            ax.plot(u_hist[:, 0, i].cpu().numpy(), label=f"u_{i+1}")

        ax.spines.top.set_linewidth(1.5)
        ax.spines.bottom.set_linewidth(1.5)
        ax.spines.left.set_linewidth(1.5)
        ax.spines.right.set_linewidth(1.5)
        ax.grid(True, linestyle='--', alpha=0.5)

        ax.legend(loc="upper left", fontsize=14, ncols=2)

        plt.tight_layout()
        plt.savefig(f"{neuron_name}_dynamic_plot.jpg", dpi=300)
        plt.close()

    elif plot_type == "3D":
        fig = go.Figure()

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=time_hist,
        #         y=u_hist,
        #         z=v_hist,
        #         mode='lines',
        #         name='RAF Dynamic',
        #         line=dict(color='cyan')
        #     )
        # )

        pio.write_html(fig, file=f"RAF_Output.html", auto_open=False)
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}. Must be '2D' or '3D'.")
    

if __name__ == "__main__":
    batch_size = 2
    seq_len = 48
    input_dim = 30 # number of RAF neurons
    dt = 1/24000

    plt.rcParams["font.sans-serif"] = "Arial"
    axis_font_size = 18

    raf_omegas1 = (torch.pi) / (torch.linspace(4, 24, steps=input_dim, dtype=torch.float32) / 24000) # original shape (30,). was 2*pi
    raf_omegas2 = (torch.pi) / (torch.linspace(4, 32, steps=input_dim, dtype=torch.float32) / 24000) # original shape (30,). was 2*pi
    raf_omegas = torch.stack((raf_omegas1, raf_omegas2), dim=-1) # stack to form (24, 2)
    raf_bs = raf_omegas / 8

    initial_dv = 4.1667e-5
    k_threshold1 = 1.5
    k_threshold2 = 1.95  # original is 1.9
    threshold1 = k_threshold1 * initial_dv # original value: 6e-5
    threshold2 = k_threshold2 * initial_dv # original value: 7.8e-5
    raf_thresholds = torch.tensor([threshold1, threshold2], dtype=torch.float32)
    raf_thresholds = repeat(raf_thresholds, 't -> b t', b=input_dim).clone()

    dbrf = DBRF(
        input_dim=input_dim,
        dual_omegas=raf_omegas,
        dual_bs=raf_bs,
        dual_threshold=raf_thresholds,
        dt=dt,
        learn_omega=False,
        learn_b=False,
        learn_dual_threshold=False
    )

    spike_train = torch.zeros((batch_size, seq_len), dtype=torch.float32)

    spike_train[0, 10] = 1.0
    spike_train[0, 20] = -1.0
    spike_train[0, 32] = 1.0

    spike_train[1, 7] = -1.0
    spike_train[1, 14] = 1.0
    spike_train[1, 22] = -1.0

    plot_neuron_dynamic(
        seq_len=seq_len,
        input_dim=input_dim,
        batch_size=batch_size,
        neuron=dbrf,
        neuron_name="DBRF",
        spk_train=spike_train,
        plot_type="2D"
    )