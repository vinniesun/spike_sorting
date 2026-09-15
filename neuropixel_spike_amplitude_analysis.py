import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import snntorch as snn
import snntorch.functional as SF
from scipy.signal import find_peaks

import random
import os
from datetime import datetime
from tqdm import tqdm
import argparse

from utils import (
    IntracorticalDataset,
    load_dataset_intracortical,
    train_test_split_spike_detection,
    create_training_dataset_spike_detection,
    dv_to_lif_spike_gen,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spike Detection on Neuropixel")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed") # 1337, 5673, 1234
    parser.add_argument("--sampling_rate", type=int, default=30000, help="Sampling rate for Neuropixel Dataset")
    parser.add_argument("--window_size", type=int, default=240, help="Window size for Neuropixel Dataset in seconds")
    parser.add_argument("--label_window_size", type=int, default=3, help="Window size for labeling spikes")
    parser.add_argument("--lif_threshold", type=float, default=0.1, help="LIF threshold for spike detection")
    parser.add_argument("--lif_tau", type=int, default=1, help="LIF tau for spike detection")
    parser.add_argument("--examine_window_size", type=int, default=8, help="Window size for examining spikes")
    parser.add_argument("--skip_forward_window_size", type=int, default=12, help="Window size for skipping forward in spike detection")
    parser.add_argument("--spike_detection_threshold", type=int, default=4, help="Threshold for spike detection in terms of number of events")
    parser.add_argument("--reset_mechanism", type=str, default="subtract", choices=["none", "subtract", "zero"], help="Reset mechanism for LIF neuron")
    
    args = parser.parse_args()

    filename = "AL036_2020-03-11_stripe240_NatIm_g0_t0.imec0.ap.bin"
    filepath = "./Neuropixel/AL036_2020-03-11/AL036_2020-03-11/"
    complete_filename = filepath + filename

    sampling_rate = args.sampling_rate
    sampling_interval = 1 / sampling_rate
    num_rows = 385
    num_channels = 384
    window_size = args.window_size # 240 seconds worth of data

    raw_data = []
    for i in range(num_channels):
        raw_data_channel = np.load(f"{filepath}raw_data_filtered_channel_{i}.npy")
        raw_data.append(raw_data_channel)
    raw_data = np.array(raw_data)
    raw_data /= 1e3 # shift from mV to V

    for i in tqdm(range(num_channels), desc="Calculating Spike Amplitudes"):
        curr_channel = raw_data[i, :]

        abs_threshold = np.median(np.abs(curr_channel))

        data_up = np.copy(curr_channel)
        data_up[data_up < abs_threshold] = 0
        peak, _ = find_peaks(data_up)

        data_down = np.copy(curr_channel)
        data_down[data_down > -abs_threshold] = 0
        trough, _ = find_peaks(np.abs(data_down))

        median_peak = np.median(data_up[peak])
        median_trough = np.median(data_down[trough])
        spik_amplitude = (median_peak - median_trough) / 2

        with open(f"Neuropixel_Spike_Amplitudes.txt", "a") as f:
            f.write(f"Channel {i}: Spike Amplitude = {spik_amplitude:.2f}\n")