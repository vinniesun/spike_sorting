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

# high omegas have high b, low omegas have low b.
# omega {4~6}:      2
# omega {7~9}:      3
# omega {10~12}:    4
# omega {13~15}:    5
# omega {16~18}:    6
# omega {19~21}:    7
# omega {22+}:      8
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

def plot_neuron_dynamic(
    seq_len: int,
    input_dim: int,
    batch_size: int,
    selected_batch_no: int,
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
        fig, ax = plt.subplots(3, 1, figsize=(12, 12))
        time = torch.arange(seq_len) / 24000
        # print(f"Time: {time}")

        pos_idx = torch.where(spk_train[selected_batch_no] > 0)[0]
        neg_idx = torch.where(spk_train[selected_batch_no] < 0)[0]
        ax[0].eventplot(time[pos_idx], lineoffsets=1, colors='blue', linelengths=0.5)
        ax[0].eventplot(time[neg_idx], lineoffsets=0, colors='red', linelengths=0.5)

        for i in range(input_dim):
            # ax.plot(u_hist[:, 0, i].cpu().numpy(), v_hist[:, 0, i].cpu().numpy(), label=f"u_{i+1}")
            line, = ax[1].plot(time, u_hist[:, selected_batch_no, i].cpu().numpy(), label=f"u_{i+1}")
            color = line.get_color()  # grab the auto-assigned colour

            # ax.plot(use_t1_hist[:, 0, i].cpu().numpy() * 1e-5, label=f"use_t1_{i+1}")
            
            pos_idx = torch.where(out_spk_hist[:, selected_batch_no, i] > 0)[0]
            neg_idx = torch.where(out_spk_hist[:, selected_batch_no, i] < 0)[0]
            # print(f"Neuron {i}: {out_spk_hist[:, selected_batch_no, i]}, {pos_idx}, {neg_idx}, {time[pos_idx]}, {time[neg_idx]}")

            ax[2].eventplot(time[pos_idx], label=f"out_spk_{i+1}", lineoffsets=i+0.15, colors=color, linelengths=0.3)
            ax[2].eventplot(time[neg_idx], lineoffsets=i-0.15, colors=color, linelengths=0.3)

        # ax[1].axhline(y=neuron.dual_threshold[0, 0].item(), color='r', linestyle='--')
        # ax[1].axhline(y=-neuron.dual_threshold[0, 0].item(), color='r', linestyle='--')

        # ax[1].axhline(y=neuron.dual_threshold[0, 1].item(), color='g', linestyle='--')
        # ax[1].axhline(y=-neuron.dual_threshold[0, 1].item(), color='g', linestyle='--')

        # ax.spines.top.set_linewidth(1.5)
        # ax.spines.bottom.set_linewidth(1.5)
        # ax.spines.left.set_linewidth(1.5)
        # ax.spines.right.set_linewidth(1.5)
        ax[0].minorticks_on()
        ax[0].grid(True, linestyle='--', alpha=0.5, which='both')
        ax[2].minorticks_on()
        ax[2].grid(True, linestyle='--', alpha=0.5, which='both')

        ax[0].set_xlim(time[0], time[-1])
        ax[1].set_xlim(time[0], time[-1])
        ax[2].set_xlim(time[0], time[-1])

        ax[1].legend(loc="lower left", fontsize=14, ncols=2)
        ax[2].legend(loc="lower left", fontsize=14, ncols=2)

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
    selected_batch_no = 0
    seq_len = 46
    input_dim = 10 # number of RAF neurons
    dt = 1/24000

    plt.rcParams["font.sans-serif"] = "Arial"
    axis_font_size = 18

    torch.set_printoptions(sci_mode=False)

    # raf_omegas1 = (torch.pi) / (torch.linspace(4, 24, steps=input_dim, dtype=torch.float32) / 24000) # original shape (30,). was 2*pi
    # raf_omegas2 = (torch.pi) / (torch.linspace(4, 32, steps=input_dim, dtype=torch.float32) / 24000) # original shape (30,). was 2*pi

    # raf_omegas1 = (torch.pi) / (
    #     torch.tensor([
    #         4, 
    #         5, 
    #         7, 
    #         7, 
    #         7, 
    #         9, 
    #         9, 
    #         9, 
    #         12, 
    #         15, 
    #     ]) / 24000
    # )
    # raf_omegas2 = (torch.pi) / (
    #     torch.tensor([
    #         5, 
    #         5, 
    #         8, 
    #         12, 
    #         16, 
    #         8, 
    #         12, 
    #         16, 
    #         13, 
    #         21,
    #     ]) / 24000
    # )
    # raf_omegas = torch.stack((raf_omegas1, raf_omegas2), dim=-1) # stack to form (24, 2)

    interval1 = torch.arange(start=4, end=20, dtype=torch.float32)
    interval2 = torch.arange(start=4, end=25, dtype=torch.float32)

    raf_omega_interval = torch.cartesian_prod(interval1, interval2)
    raf_omegas = torch.pi / (raf_omega_interval / 24000)

    # raf_bs = raf_omegas / 8 

    raf_bs = torch.tensor([
        [
            raf_omegas[i, 0] / raf_interval_to_b_mapping[raf_omega_interval[i, 0].item()], 
            raf_omegas[i, 1] / raf_interval_to_b_mapping[raf_omega_interval[i, 1].item()]
        ] for i in range(raf_omegas.shape[0])
    ])

    idx = torch.randperm(raf_omegas.shape[0])[:input_dim]
    # raf_omegas = raf_omegas[idx]
    # raf_bs = raf_bs[idx]
    raf_omegas = raf_omegas[25:25+input_dim]
    raf_bs = raf_bs[:input_dim]

    initial_dv = 4.1667e-5

    k_threshold1 = 2
    k_threshold2 = 3.5  # original is 1.9
    threshold1 = k_threshold1 * initial_dv # original value: 6e-5
    threshold2 = k_threshold2 * initial_dv # original value: 7.8e-5
    raf_thresholds = torch.tensor([threshold1, threshold2], dtype=torch.float32)
    raf_thresholds = repeat(raf_thresholds, 't -> b t', b=input_dim).clone()

    # raf_thresholds = torch.tensor(
    #     [
    #         [initial_dv * 2, initial_dv * 3.5],
    #         [initial_dv * 2, initial_dv * 3.5],
    #         [initial_dv * 2, initial_dv * 3.5],
    #         [initial_dv * 2, initial_dv * 3.5],
    #         [initial_dv * 2, initial_dv * 3.5],
    #         [initial_dv * 2, initial_dv * 3.5],
    #         [initial_dv * 2, initial_dv * 3.5],
    #         [initial_dv * 2, initial_dv * 3.5],
    #         [initial_dv * 2, initial_dv * 3.5],
    #         [initial_dv * 2, initial_dv * 3.5],
    #     ]
    # )

    ########### 
    # 
    # Need different q coefficient for each neuron.
    # neurons with higher frequency (i.e. smaller period) should have smaller q coefficient, so that they have small refractory periods.
    # The inverse is true for neurons with lower frequency (i.e. larger period), they should have larger q coefficient, so that they have large refractory period.
    #
    ###########
    raf_q_coeff = torch.tensor([1e-1, 1e-3], dtype=torch.float32)
    raf_q_coeff = repeat(raf_q_coeff, 't -> b t', b=input_dim).clone()
    # raf_q_coeff = torch.tensor(
    #     [
    #         [1e-2, 1e-3],
    #         [1e-2, 1e-3],
    #         [1e-2, 1e-2],
    #         [1e-2, 1e-2],
    #         [1e-2, 1e-2],
    #         [1e-2, 1e-2],
    #         [1e-2, 1e-2],
    #         [1e-2, 1e-2],
    #         [1e-2, 1e-2],
    #         [1e-2, 1e-2],
    #     ]
    # )

    dbrf = DBRF(
        input_dim=input_dim,
        dual_omegas=raf_omegas,
        dual_bs=raf_bs,
        dual_threshold=raf_thresholds,
        dual_q_coeff=raf_q_coeff,
        dt=dt,
        learn_omega=False,
        learn_b=False,
        learn_dual_threshold=False
    )

    # spike_train = torch.zeros((batch_size, seq_len), dtype=torch.float32)

    # spike_train[0, 19] = 1.0
    # spike_train[0, 20] = 2.0
    # spike_train[0, 21] = 1.0
    # spike_train[0, 23] = -3.0
    # spike_train[0, 24] = -3.0
    # spike_train[0, 25] = -1.0
    # spike_train[0, 31] = 1.0
    # spike_train[0, 35] = 1.0
    # spike_train[0, 43] = 1.0

    # spike_train[1, 16] = -1.0
    # spike_train[1, 19] = 2.0
    # spike_train[1, 20] = 3.0
    # spike_train[1, 23] = -2.0
    # spike_train[1, 24] = -2.0
    # spike_train[1, 25] = -1.0
    # spike_train[1, -2] = 1.0

    spike_train = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, -1.0, -2.0, -3.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, -3.0, -3.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    spike_train = torch.clamp(spike_train, min=-1.0, max=1.0)

    plot_neuron_dynamic(
        seq_len=seq_len,
        input_dim=input_dim,
        batch_size=batch_size,
        selected_batch_no=selected_batch_no,
        neuron=dbrf,
        neuron_name="DBRF",
        spk_train=spike_train,
        plot_type="2D"
    )
