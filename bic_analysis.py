import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator

from datetime import datetime

import torch

from tqdm import tqdm

from utils import (
    load_dataset_intracortical,
    generate_event_stream_lif,
    reconstruction_lif,
    calc_rmse
)

if __name__ == "__main__":
    SEED = 1337 # 1234, 1337, 5673
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # for lif, the threshold is: 0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0
    thresholds = np.array([0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]) # to do 0.6

    filepath = "./intracortical_dataset/"

    bic_complete = {}
    for threshold in thresholds:
        bic_complete[threshold] = {"005": [], "01": [], "015": [], "02": []}

    for threshold in tqdm(thresholds, desc="LIF thresholds"):
        for difficulty in ["Easy1", "Easy2", "Difficult1", "Difficult2"]:
            for gt_noise_level in ["005", "01", "015", "02"]:
                filename = f"C_{difficulty}_noise{gt_noise_level}.mat"

                signal, spike_class_label, spike_times, sampling_interval, sampling_rate, spike_pulse_1ms_idx_length, spike_classes, filtered_signal = load_dataset_intracortical(filepath, filename)

                spike_train, event_stream = generate_event_stream_lif(filtered_signal, sampling_interval, uth=threshold, lif_tau=sampling_interval, if_reconstruct=True)
                reconstructed_signal = reconstruction_lif(event_stream, time_step=sampling_interval, reconstruct_tau=10*sampling_interval, alpha=0.8, order=2)

                rmse = calc_rmse(filtered_signal, reconstructed_signal, spike_times)

                ns = filtered_signal.shape[0]
                mse = rmse ** 2
                # kappa = event_stream.shape[0]
                kappa = np.sum(np.abs(spike_train)) # number of spikes in the spike train
                bic = ns * np.log(mse) + kappa * np.log(ns)
                bic_complete[threshold][gt_noise_level].append(bic)

                # print(f"filename: {filename}, threshold: {threshold}, ns: {ns}, kappa: {kappa}, kappa_sum: {np.sum(np.abs(spike_train))}, rmse: {rmse:.4f}, bic: {bic:.4f}")

    bic_005, bic_01, bic_015, bic_02 = [], [], [], []
    for threshold in thresholds:
        bic_005.append(np.mean(bic_complete[threshold]["005"]))
        bic_01.append(np.mean(bic_complete[threshold]["01"]))
        bic_015.append(np.mean(bic_complete[threshold]["015"]))
        bic_02.append(np.mean(bic_complete[threshold]["02"]))

    plt.rcParams["font.sans-serif"] = "Arial"
    axis_font_size = 18

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(thresholds, bic_005, marker='o', color="r", label=r"$\sigma_{005}$")
    ax.plot(thresholds, bic_01, marker='o', color="g", label=r"$\sigma_{01}$")
    ax.plot(thresholds, bic_015, marker='o', color="cyan", label=r"$\sigma_{015}$")
    ax.plot(thresholds, bic_02, marker='o', color="purple", label=r"$\sigma_{02}$")

    ax.spines.top.set_linewidth(1.5)
    ax.spines.bottom.set_linewidth(1.5)
    ax.spines.left.set_linewidth(1.5)
    ax.spines.right.set_linewidth(1.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.set_xlabel(r"$\tau$", fontsize=axis_font_size)
    ax.set_ylabel("BIC", fontsize=axis_font_size)
    ax.tick_params(labelsize=axis_font_size)
    ax.legend(loc="best", fontsize=14)
    plt.tight_layout()
    plt.savefig("final_figures/fig_bic_plots.jpg", dpi=600)
    plt.savefig("final_figures/fig_bic_plots.eps", format='eps', bbox_inches='tight')
    plt.close()
