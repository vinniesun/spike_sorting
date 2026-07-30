import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator

import torch

from scipy.signal import ellip, lfilter, butter, find_peaks
from scipy.io import loadmat

from datetime import datetime

import h5py 
import math
from typing import Tuple, Union, List, Optional

from tqdm import tqdm

from utils import (
    get_threshold_reset_counts,
    generate_event_stream_dm,
    generate_event_stream_lif,
    leaky_integrate_neuron,
    lif_neuron,
    load_dataset_intracortical
)

if __name__ == "__main__":
    # for lif, the threshold is: 0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0
    thresholds = np.array([1.8])

    filepath = "./intracortical_dataset/"
    for difficulty in ["Easy1"]:
        for gt_noise_level in ["005"]:
            filename = f"C_{difficulty}_noise{gt_noise_level}.mat"

            signal, spike_class_label, spike_times, sampling_interval, sampling_rate, spike_pulse_1ms_idx_length, spike_classes, filtered_signal = load_dataset_intracortical(filepath, filename)

            spike_train = generate_event_stream_lif(filtered_signal, sampling_interval, uth=thresholds, lif_tau=sampling_interval, if_reconstruct=False)

            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            time = np.arange(filtered_signal[400:600].shape[0]) / 24000
            ax[0].plot(time, filtered_signal[400:600], color="blue", label="Filtered Signal")
            on_idx = np.where(spike_train[400:600] > 0)[0]
            off_idx = np.where(spike_train[400:600] < 0)[0]
            ax[1].eventplot(time[on_idx], lineoffsets=1, colors="red", label="On Spikes", linelengths=0.3)
            ax[1].eventplot(time[off_idx], lineoffsets=0, colors="green", label="Off Spikes", linelengths=0.3)

            plt.tight_layout()
            plt.savefig(f"lif_signal_check.jpg", dpi=300)
            plt.close()