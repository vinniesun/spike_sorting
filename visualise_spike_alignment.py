import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import snntorch.functional as SF
import random
import os
from datetime import datetime
from tqdm import tqdm
import copy

from utils import (
    IntracorticalDataset,
    load_dataset_intracortical,
    dv_to_lif_spike_gen,
    train_test_split_spike_sorting,
    generate_event_stream_dm,
)

if __name__ == "__main__":
    BATCH_SIZE = 256 # 128 or 64
    NUM_EPOCHS= 120 # 60 epochs seems to work for lstm + lif model. slstm + lif seems to need more epochs.

    filepath = "./intracortical_dataset/"
    
    encoder = "dm" # "dm" or "dv"
    reset_mechanism = "subtract" # "none", "subtract", "zero"
    train_test_split_ratio = 0.7 # 70% training, 30% testing
    encoder_threshold = 0.2
    lif_tau = 1 * (1/24000)
    detection_window_size = 8
    sorting_window_size = 48 # 2ms
    
    DETECTION_IDX = f"Intracortical_Spike_Detection_IDX.txt"
    PLOT_PATH = f"compare_spike_alignment/"

    # parse the detection_idx file to get the detection idx and spike time idx for each file
    # The reason for this is that other papers only do spike sorting on the detected spikes,
    # not the entire signal. 
    detection_idx_results_when_detected = {}
    detection_idx_results_label_spike_time = {}
    with open(DETECTION_IDX, "r") as f:
        detection_idx_lines = f.readlines()
    
    line_no = 0
    while line_no < len(detection_idx_lines):
        line = detection_idx_lines[line_no]
        if "Filename: " in line:
            filename = line.split("Filename: ")[1][:-2]
            detection_idx_results_when_detected[filename] = []
            detection_idx_results_label_spike_time[filename] = []
        else:
            temp = line.split(",")
            detected_idx = int(temp[0].split("Detection idx: ")[1])
            spike_time_idx = int(temp[1].split(" Spike time idx: ")[1])
            detection_idx_results_when_detected[filename].append(detected_idx)
            detection_idx_results_label_spike_time[filename].append(spike_time_idx)
    
        line_no += 1

    for difficulty in ["Difficult1"]:
        for noise_level in ["005", "01", "015", "02"]:
            filename = f"C_{difficulty}_noise{noise_level}.mat"

            signal, spike_class_label, spike_times, sampling_interval, \
            sampling_rate, spike_pulse_1ms_idx_length, spike_classes, \
            filtered_signal = load_dataset_intracortical(filepath, filename)

            on_threshold = encoder_threshold
            off_threshold = -encoder_threshold
            event_stream = generate_event_stream_dm(filtered_signal, on_threshold, off_threshold)
            dm_spike_train = np.zeros_like(signal)
            dm_spike_train[event_stream[:, 0].astype(int)] = event_stream[:, 1] - event_stream[:, 2]

            dv_u_hist, dv_spike_train, dv_time_lif = dv_to_lif_spike_gen(
                signal=filtered_signal,
                lif_threshold=0.4,
                sampling_interval=sampling_interval,
                lif_tau=lif_tau,
                reset_mechanism=reset_mechanism
            )

            # visualise the original spike alignment vs the current spike alignment
            # use axvline to show where's the start and end of the window post alignment.
            detected_spikes = np.array(detection_idx_results_label_spike_time[filename])
            detected_spike_times = np.array(detection_idx_results_when_detected[filename])
            all_spike_signals = {i: [] for i in spike_classes}
            all_spk_trains = {i: [] for i in spike_classes}
            for i in range(len(spike_times)):
                if i in detected_spikes:
                    idx = np.where(detected_spikes == i)[0][0]

                    filtered_signal_segment = filtered_signal[spike_times[i] - 70:spike_times[i] + 70]
                    dm_spk_train_segment = dm_spike_train[spike_times[i] - 70:spike_times[i] + 70]
                    dv_spk_train_segment = dv_spike_train[spike_times[i] - 70:spike_times[i] + 70]

                    time = np.arange(filtered_signal_segment.shape[0])

                    # time that's been zero aligned
                    dm_start_time = 70 - 23
                    dm_end_time = 70 + 23
                    dv_start_time = 70 + (detected_spike_times[idx] - spike_times[i]) - detection_window_size
                    dv_end_time = 70 + (detected_spike_times[idx] - spike_times[i]) + sorting_window_size - detection_window_size

                    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

                    ax[0].plot(time, filtered_signal_segment)
                    ax[0].axvline(x=dm_start_time, color="red", linestyle="--", label="DM Start Time")
                    ax[0].axvline(x=dm_end_time, color="green", linestyle="--", label="DM End Time")
                    ax[0].axvline(x=dv_start_time, color="orange", linestyle="--", label="DV Start Time")
                    ax[0].axvline(x=dv_end_time, color="purple", linestyle="--", label="DV End Time")
                    ax[0].legend(loc="best")
                    ax[0].minorticks_on()
                    ax[0].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")

                    ax[1].eventplot(time[np.where(dm_spk_train_segment > 0)[0]], color="green", lineoffsets=0.0, linelengths=0.4, label="DM ON")
                    ax[1].eventplot(time[np.where(dm_spk_train_segment < 0)[0]], color='black', lineoffsets=-0.4, linelengths=0.4, label="DM OFF")
                    ax[1].axvline(x=dm_start_time, color="red", linestyle="--", label="DM Start Time")
                    ax[1].axvline(x=dm_end_time, color="green", linestyle="--", label="DM End Time")
                    ax[1].axvline(x=dv_start_time, color="orange", linestyle="--", label="DV Start Time")
                    ax[1].axvline(x=dv_end_time, color="purple", linestyle="--", label="DV End Time")
                    ax[1].legend(loc="best")
                    ax[1].minorticks_on()
                    ax[1].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")

                    ax[2].eventplot(time[np.where(dv_spk_train_segment > 0)[0]], color="red", lineoffsets=0.0, linelengths=0.4, label="DV ON")
                    ax[2].eventplot(time[np.where(dv_spk_train_segment < 0)[0]], color='blue', lineoffsets=-0.4, linelengths=0.4, label="DV OFF")
                    ax[2].axvline(x=dm_start_time, color="red", linestyle="--", label="DM Start Time")
                    ax[2].axvline(x=dm_end_time, color="green", linestyle="--", label="DM End Time")
                    ax[2].axvline(x=dv_start_time, color="orange", linestyle="--", label="DV Start Time")
                    ax[2].axvline(x=dv_end_time, color="purple", linestyle="--", label="DV End Time")
                    ax[2].legend(loc="best")
                    ax[2].minorticks_on()
                    ax[2].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")

                    plt.tight_layout()
                    plt.savefig(f"{PLOT_PATH}{filename[:-4]}_spike_alignment_{i}.jpg")
                    plt.close()

