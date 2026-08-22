import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.io as pio

from utils import (
    load_dataset_intracortical,
    leaky_integrate_neuron,
    generate_event_stream_dm,
    generate_event_stream_lif,
    reconstruction_lif,
    calc_rmse,
    dv_to_lif_spike_gen
)

def lif_reconstruct_dv(spikes, dt, tau, lif_threshold=1.0):
    """
    Reconstruct an estimate of dv from a signed spike train.
    spikes: array of shape (N,), values in {-1, 0, 1}
    dt: timestep
    tau: LIF time constant
    lif_threshold: amplitude scale for each spike
    """
    alpha = np.exp(-dt / tau)
    beta = 1.0 - alpha

    x = spikes.astype(np.float64) * lif_threshold
    dv_hat = np.zeros_like(x, dtype=np.float64)

    u = 0.0
    for n, s in enumerate(x):
        u = alpha * u + beta * s
        dv_hat[n] = u

    return dv_hat

def integrate_dv_to_v(dv_hat, dt, V0=0.0):
    """
    Recover V from dv_hat using cumulative integration.
    """
    V_hat = np.zeros_like(dv_hat, dtype=np.float64)
    V_hat[0] = V0
    for n in range(1, len(dv_hat)):
        V_hat[n] = V_hat[n-1] + dv_hat[n-1] * dt

    return V_hat

def lif_reconstruct_signal(spikes, dt, tau, V0=0.0, lif_threshold=1.0):
    dv_hat = lif_reconstruct_dv(spikes, dt=dt, tau=tau, lif_threshold=lif_threshold)
    V_hat = integrate_dv_to_v(dv_hat, dt=dt, V0=V0)
    return dv_hat, V_hat

def compression_ratio(filtered_signal, spike_train):
    idx = np.where(spike_train != 0)[0]
    tdr_fs = filtered_signal.shape[0] * 12
    tdr_dm = idx.shape[0] * (np.ceil(np.log2(10000)) + 1) if idx.shape[0] > 0 else 1

    return tdr_fs / tdr_dm

if __name__ == "__main__":
    filepath = "./intracortical_dataset/"

    dm_threshold = np.array([0.2])
    lif_threshold = 0.2
    reset_mechanism = "none" # "none", "subtract", "zero"

    for difficulty in ["Difficult1", "Difficult2", "Easy1", "Easy2"]:
        for noise_level in ["005", "01", "015", "02"]:
            filename = f"C_{difficulty}_noise{noise_level}.mat"

            signal, spike_class_label, spike_times, sampling_interval, \
            sampling_rate, spike_pulse_1ms_idx_length, spike_classes, \
            filtered_signal = load_dataset_intracortical(filepath, filename)

            dv_u_hist, dv_spk_hist, dv_time_lif = dv_to_lif_spike_gen(
                signal=filtered_signal,
                lif_threshold=lif_threshold,
                sampling_interval=sampling_interval,
                lif_tau=4*sampling_interval,
                reset_mechanism=reset_mechanism
            )

            reconstructed_signal = reconstruction_lif(dv_spk_hist, time_step=sampling_interval, reconstruct_tau=6*sampling_interval, alpha=0.8, order=2)
            # reconstructed_signal = lif_reconstruct_signal(dv_spk_hist, dt=sampling_interval, tau=8*sampling_interval, V0=0.0, lif_threshold=lif_threshold)[1]
            rmse = calc_rmse(filtered_signal, reconstructed_signal, spike_times)

            print(f"filename: {filename}, lif_threshold: {lif_threshold}, rmse: {rmse:.4f}")

            # generate dm spike train for comparison
            event_stream = generate_event_stream_dm(filtered_signal, dm_threshold, -dm_threshold)
            dm_spike_train = np.zeros_like(signal)
            dm_spike_train[event_stream[:, 0].astype(int)] = event_stream[:, 1] - event_stream[:, 2]
            dm_spike_train = np.clip(dm_spike_train, -1, 1)

            # generate lif spike train for comparison
            lif_spike_train = generate_event_stream_lif(filtered_signal, sampling_interval, uth=lif_threshold, lif_tau=sampling_interval, if_reconstruct=False)

            # Compare the spike trains
            fig, ax = plt.subplots(2, 1 , figsize=(12, 6), sharex=True)

            start_idx, end_idx = spike_times[0] - 100, spike_times[0] + 100

            time = np.arange(filtered_signal[start_idx:end_idx].shape[0]) / 24000
            ax[0].plot(time, filtered_signal[start_idx:end_idx], color="blue", label="Filtered Signal")
            ax[0].plot(time, reconstructed_signal[start_idx:end_idx], color="red", label="Reconstructed Signal")

            # plot dm spikes
            dm_on_idx = np.where(dm_spike_train[start_idx:end_idx] > 0)[0]
            dm_off_idx = np.where(dm_spike_train[start_idx:end_idx] < 0)[0]
            ax[1].eventplot(time[dm_on_idx], lineoffsets=0.15, colors="red", label="DM On Spikes", linelengths=0.3)
            ax[1].eventplot(time[dm_off_idx], lineoffsets=-0.15, colors="green", label="DM Off Spikes", linelengths=0.3)

            # plot lif spikes
            lif_on_idx = np.where(lif_spike_train[start_idx:end_idx] > 0)[0]
            lif_off_idx = np.where(lif_spike_train[start_idx:end_idx] < 0)[0]
            ax[1].eventplot(time[lif_on_idx], lineoffsets=0.75, colors="orange", label="LIF On Spikes", linelengths=0.3)
            ax[1].eventplot(time[lif_off_idx], lineoffsets=0.45, colors="purple", label="LIF Off Spikes", linelengths=0.3)

            dv_on_idx = np.where(dv_spk_hist[start_idx:end_idx] > 0)[0]
            dv_off_idx = np.where(dv_spk_hist[start_idx:end_idx] < 0)[0]
            ax[1].eventplot(time[dv_on_idx], lineoffsets=1.35, colors="cyan", label="DV On Spikes", linelengths=0.3)
            ax[1].eventplot(time[dv_off_idx], lineoffsets=1.05, colors="magenta", label="DV Off Spikes", linelengths=0.3)

            ax[0].set_xlim([time[0], time[-1]])
            ax[1].set_xlim([time[0], time[-1]])

            ax[0].minorticks_on()
            ax[0].grid(True, linestyle='--', alpha=0.5, which='both')
            ax[1].minorticks_on()
            ax[1].grid(True, linestyle='--', alpha=0.5, which='both')

            ax[0].legend(loc="best")
            ax[1].legend(loc="best")

            plt.tight_layout()
            plt.savefig(f"./compare_spike_trains/{difficulty}_noise{noise_level}.jpg", dpi=300)
            plt.close()
            # time = np.arange(24000) / 24000
            # for i in range(0, len(filtered_signal), 24000):
            #     start, end = i, i + 24000

            #     fig = go.Figure()

            #     fig.add_trace(
            #         go.Scatter(
            #             x=time,
            #             y=filtered_signal[start:end] + 4,
            #             name="Filtered Signal",
            #             mode="lines",
            #             line=dict(color="blue")
            #         )
            #     )

            #     dm_here = dm_spike_train[start:end]
            #     pos_idx = np.where(dm_here > 0)[0]
            #     neg_idx = np.where(dm_here < 0)[0]
            #     fig.add_trace(
            #         go.Scatter(
            #             x=time[pos_idx],
            #             y=dm_here[pos_idx] * 0.3 - 0.15,
            #             name="DM On",
            #             mode="markers",
            #             marker=dict(
            #                 symbol="line-ns-open",
            #                 color="red",
            #                 size=20,
            #                 line=dict(width=2),
            #             ),
            #         )
            #     )
            #     fig.add_trace(
            #         go.Scatter(
            #             x=time[neg_idx],
            #             y=dm_here[neg_idx] * 0.3 + 0.15,
            #             name="DM Off",
            #             mode="markers",
            #             marker=dict(
            #                 symbol="line-ns-open",
            #                 color="red",
            #                 size=20,
            #                 line=dict(width=2),
            #             ),
            #         )
            #     )

            #     lif_here = lif_spike_train[start:end]
            #     pos_idx = np.where(lif_here > 0)[0]
            #     neg_idx = np.where(lif_here < 0)[0]
            #     fig.add_trace(
            #         go.Scatter(
            #             x=time[pos_idx],
            #             y=lif_here[pos_idx] * 0.3 + 0.85,
            #             name="LIF On",
            #             mode="markers",
            #             marker=dict(
            #                 symbol="line-ns-open",
            #                 color="orange",
            #                 size=20,
            #                 line=dict(width=2),
            #             ),
            #         )
            #     )
            #     fig.add_trace(
            #         go.Scatter(
            #             x=time[neg_idx],
            #             y=lif_here[neg_idx] * 0.3 + 1.15,
            #             name="LIF Off",
            #             mode="markers",
            #             marker=dict(
            #                 symbol="line-ns-open",
            #                 color="orange",
            #                 size=20,
            #                 line=dict(width=2),
            #             ),
            #         )
            #     )

            #     dv_here = dv_spk_hist[start:end]
            #     pos_idx = np.where(dv_here > 0)[0]
            #     neg_idx = np.where(dv_here < 0)[0]
            #     fig.add_trace(
            #         go.Scatter(
            #             x=time[pos_idx],
            #             y=dv_here[pos_idx] * 0.3 + 1.85,
            #             name="DV On",
            #             mode="markers",
            #             marker=dict(
            #                 symbol="line-ns-open",
            #                 color="cyan",
            #                 size=20,
            #                 line=dict(width=2),
            #             ),
            #         )
            #     )
            #     fig.add_trace(
            #         go.Scatter(
            #             x=time[neg_idx],
            #             y=dv_here[neg_idx] * 0.3 + 2.15,
            #             name="DV Off",
            #             mode="markers",
            #             marker=dict(
            #                 symbol="line-ns-open",
            #                 color="cyan",
            #                 size=20,
            #                 line=dict(width=2),
            #             ),
            #         )
            #     )

            #     pio.write_html(fig, file=f"./compare_spike_trains/{difficulty}_noise{noise_level}_{i}s.html", auto_open=False)
