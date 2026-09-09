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
    """
        Dataset downloaded from: https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595?file=21819066
    """

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

    TRAINING_LOG_PATH = "./spike_detection_training_log"
    if not os.path.exists(TRAINING_LOG_PATH):
        os.makedirs(TRAINING_LOG_PATH)
    TRAINING_LOG_NAME = f"{TRAINING_LOG_PATH}/training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    DETECTION_IDX = f"Intracortical_Spike_Detection_IDX.txt"

    SEED = args.seed # 1337, 5673, 1234
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    filename = "AL036_2020-03-11_stripe240_NatIm_g0_t0.imec0.ap.bin"
    filepath = "./Neuropixel/AL036_2020-03-11/AL036_2020-03-11/"
    complete_filename = filepath + filename
    ks_filepath = "./Neuropixel/AL036_2020-03-11/AL036_2020-03-11/ks"

    sampling_rate = parser.sampling_rate
    sampling_interval = 1 / sampling_rate
    num_rows = 385
    num_channels = 384
    window_size = parser.window_size # 240 seconds worth of data
    
    total_samples = int(window_size * sampling_rate)
    
    reset_mechanism = parser.reset_mechanism # "none", "subtract", "zero"

    label_window_size = parser.label_window_size
    lif_threshold = parser.lif_threshold
    lif_tau = parser.lif_tau * (1/30000)

    examine_window_size = parser.examine_window_size # 1ms
    skip_forward_window_size = parser.skip_forward_window_size # 0.5ms
    spike_detection_threshold = parser.spike_detection_threshold # number of events to be exceeded to be classified as an AP.

    with open(TRAINING_LOG_NAME, "a") as f:
        f.write(f"Seed Number: {SEED}\nevent density threshold: {spike_detection_threshold}\nexamin_window_size: {examine_window_size}\n")
        f.write(f"skip_forward_window_size: {skip_forward_window_size}\nlabel_window_size: {label_window_size}\n")
        f.write(f"lif_threshold: {lif_threshold}\nlif_tau: {lif_tau}\nreset_mechanism: {reset_mechanism}\n")

    raw_data = []
    for i in range(num_channels):
        raw_data_channel = np.load(f"{filepath}raw_data_filtered_channel_{i}.npy")
        raw_data.append(raw_data_channel)
    raw_data = np.array(raw_data)
    raw_data /= 1e3 # shift from mV to V
    print(f"loaded data is of shape: {raw_data.shape}")

    gt_spike_times = np.load(f"{ks_filepath}/spike_times.npy")      # This gives index of when the spike occurs in the raw data. shape (number of spikes)
    spike_clusters = np.load(f"{ks_filepath}/spike_clusters.npy")   # this gives the cluster id for each spike. shape (number of spikes)
    pc_features_ind = np.load(f"{ks_filepath}/pc_feature_ind.npy")  # this gives the channels responsible for each cluster group. shape (clusters, channels)

    # trim gt_spike_times and spike_clusters to window_size
    gt_spike_times = gt_spike_times[gt_spike_times < total_samples]
    spike_clusters = spike_clusters[:len(gt_spike_times)]

    # generate dv for all channels
    dv_all_channels = []
    for i in tqdm(range(num_channels), desc="Generating DV for all channels"):
        dv_u_hist, dv_spike_train, dv_time_lif = dv_to_lif_spike_gen(
            signal=raw_data[i],
            lif_threshold=lif_threshold,
            sampling_interval=sampling_interval,
            lif_tau=lif_tau,
            reset_mechanism=reset_mechanism
        )

        dv_all_channels.append(dv_spike_train)

    dv_all_channels = np.array(dv_all_channels)

    # plot some dv_spike_train to verify
    for i in tqdm(range(10), desc="Plotting DV spike trains"):
        spike_time = gt_spike_times[i]
        cluster_id = spike_clusters[i]
        affected_channels = pc_features_ind[cluster_id]

        for ac in affected_channels:
            fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            signal = raw_data[ac, spike_time-50:spike_time+50]
            curr_spk_train = dv_all_channels[ac, spike_time-50:spike_time+50]
            time = np.arange(signal.shape[0])

            ax[0].plot(signal)
            ax[0].axvline(x=50, color='r', linestyle='--')

            pos_idx = np.where(curr_spk_train > 0)[0]
            neg_idx = np.where(curr_spk_train < 0)[0]
            ax[1].eventplot(pos_idx, lineoffsets=1, colors='r', linelengths=0.5, label="On Events")
            ax[1].eventplot(neg_idx, lineoffsets=0, colors='b', linelengths=0.5, label="Off Events")

            ax[1].legend(loc="best")

            ax[0].set_xlim(time[0], time[-1])
            ax[1].set_xlim(time[0], time[-1])

            plt.tight_layout()
            plt.savefig(f"Neuropixel_example_dv_train/channel_{ac}_cluster_{cluster_id}.png")
            plt.close()
