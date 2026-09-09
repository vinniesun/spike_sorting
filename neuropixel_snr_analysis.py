import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import snntorch as snn
import snntorch.functional as SF

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

    raw_data_copy = raw_data.copy()

    NEO = np.power(raw_data,2) - np.multiply(np.hstack((np.zeros((num_channels,1)),raw_data))[:,0:-1],np.hstack((raw_data,np.zeros((num_channels,1))))[:,0:-1])
    NEO = np.power(NEO,2) - np.multiply(np.hstack((np.zeros((num_channels,1)),NEO))[:,0:-1],np.hstack((NEO,np.zeros((num_channels,1))))[:,0:-1])

    multiplier = 40
    THD = np.mean(NEO,axis=1).reshape(-1,1)  * multiplier
    SPD = np.where(NEO > THD, 1, 0)

    refractory = 1e-3 / sampling_interval
    extend = 1e-3 / sampling_interval
    spike_times_GT = np.empty(num_channels, dtype=object)
    for i in tqdm(range(num_channels), desc="Calculating SNRs"):
        spks, p2p, idx = [], [], np.array([])

        checkpoint = 0
        spk_list = np.where(SPD[i] == 1)[0]
        spk_list_updated = np.array([])
        for j in range(np.size(spk_list)):
            if (spk_list[j] > checkpoint):
                spk_list_updated = np.append(spk_list_updated,spk_list[j])
                checkpoint = spk_list_updated[-1] + refractory
        spike_times_GT[i] = spk_list_updated.astype(int)

        for spk in spike_times_GT[i]:
            start = int(max(0,spk - extend))
            end = int(min(sampling_rate * window_size - 1, spk + extend))
            spks.append(raw_data_copy[i,start:end+1])
            p2p.append(np.max(raw_data_copy[i,start:end+1]) - np.min(raw_data_copy[i,start:end+1]))
            idx = np.concatenate((idx, np.arange(start,end)))

        noise = np.delete(raw_data_copy[i], idx.astype(int))

        noise_intensity = np.std(noise)
        sig_intensity = np.mean(np.array(p2p))
        snr = 20 * np.log10(sig_intensity / noise_intensity)

        with open(f"Neuropixel_SNRs.txt", "a") as f:
            f.write(f"Channel {i}: SNR = {snr:.2f} dB\n")
