import argparse
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import (
    dv_to_lif_spike_gen,
)

def plot_example_dv_trains(raw_data, dv_all_channels, gt_spike_times, spike_clusters, pc_features_ind):
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

def neo_labeling(raw_signal, neo_multiplier=80, gt_window=3, num_channels=384):
    neo = np.power(raw_signal, 2) - \
          np.multiply(
              np.hstack((np.zeros((num_channels, 1)), raw_signal))[:, :-1],
              np.hstack((raw_signal, np.zeros((num_channels, 1))))[:, :-1]
          )
    neo = np.power(neo, 2) - \
          np.multiply(
              np.hstack((np.zeros((num_channels, 1)), neo))[:, :-1],
              np.hstack((neo, np.zeros((num_channels, 1))))[:, :-1]
          )
    thd = np.mean(neo, axis=1).reshape(-1, 1) * neo_multiplier # shape: (384, 1)
    
    # Original
    label = np.where(neo > thd, 1, 0) # (384, 300000)
    
    # New
    # spd_label = np.zeros_like(raw_signal)
    # for i in range(num_channels):
    #     spk_idx = np.where(neo[i, :] > thd[i, :])[0]
    #     spk_idx += 24
    #     spk_idx = spk_idx.astype(int)
    #     for idx in spk_idx:
    #         spd_label[i, idx-gt_window:idx+gt_window+1] = 1

    return label

if __name__ == "__main__":
    """
        Dataset downloaded from: https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595?file=21819066
    """

    parser = argparse.ArgumentParser(description="Spike Detection on Neuropixel")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed") # 1337, 5673, 1234
    parser.add_argument("--sampling_rate", type=int, default=30000, help="Sampling rate for Neuropixel Dataset")
    parser.add_argument("--window_size", type=int, default=200, help="Window size for Neuropixel Dataset in seconds")
    parser.add_argument("--label_window_size", type=int, default=4, help="Window size for labeling spikes")
    parser.add_argument("--skip_forward_window_size", type=int, default=15, help="Window size for skipping forward in spike detection")
    parser.add_argument("--reset_mechanism", type=str, default="subtract", choices=["none", "subtract", "zero"], help="Reset mechanism for LIF neuron")
    parser.add_argument("--chosen_channel", type=int, default=0, help="Channel to detect spikes")
    
    args = parser.parse_args()

    TRAINING_LOG_PATH = "./spike_detection_training_log"
    if not os.path.exists(TRAINING_LOG_PATH):
        os.makedirs(TRAINING_LOG_PATH)
    TRAINING_LOG_NAME = f"{TRAINING_LOG_PATH}/training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    DETECTION_IDX = f"Neuropixel_Spike_Detection_IDX.txt"

    SEED = args.seed # 1337, 5673, 1234
    random.seed(SEED)
    np.random.seed(SEED)

    filename = "AL036_2020-03-11_stripe240_NatIm_g0_t0.imec0.ap.bin"
    filepath = "./Neuropixel/AL036_2020-03-11/AL036_2020-03-11/"
    complete_filename = filepath + filename
    ks_filepath = "./Neuropixel/AL036_2020-03-11/AL036_2020-03-11/ks"

    sampling_rate = args.sampling_rate
    sampling_interval = 1 / sampling_rate
    num_rows = 385
    num_channels = 384
    window_size = args.window_size # 240 seconds worth of data
    
    total_samples = int(window_size * sampling_rate)
    reset_mechanism = args.reset_mechanism # "none", "subtract", "zero"
    label_window_size = args.label_window_size
    skip_forward_window_size = args.skip_forward_window_size # 0.5ms

    with open(TRAINING_LOG_NAME, "a") as f:
        f.write(f"Detection Channel: {args.chosen_channel}\n\n")

    spike_amplitudes = []
    with open("Neuropixel_Spike_Amplitudes.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        _, amp = line.split(" = ")
        spike_amplitudes.append(float(amp))
    spike_amplitudes = np.array(spike_amplitudes)
    spike_amplitude = spike_amplitudes[args.chosen_channel]

    raw_data = np.array(np.load(f"{filepath}raw_data_filtered_channel_{args.chosen_channel}.npy"))
    raw_data /= 1e3 # shift from mV to V

    gt_spike_times = np.load(f"{ks_filepath}/spike_times.npy")      # This gives index of when the spike occurs in the raw data. shape (number of spikes)
    spike_clusters = np.load(f"{ks_filepath}/spike_clusters.npy")   # this gives the cluster id for each spike. shape (number of spikes)
    pc_features_ind = np.load(f"{ks_filepath}/pc_feature_ind.npy")  # this gives the channels responsible for each cluster group. shape (clusters, channels)

    # trim gt_spike_times and spike_clusters to window_size
    gt_spike_times = gt_spike_times[gt_spike_times < total_samples]
    # need another version for ground truth based on NEO
    # gt_spike_times = neo_labeling(raw_data, neo_multiplier=80) # neo_multiplier: 40, 60, 70, 80
    spike_clusters = spike_clusters[:len(gt_spike_times)]
    print(gt_spike_times.shape)

    for lif_threshold_multiplier in [0.1, 0.2, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6]:
        for lif_tau_multiplier in [1, 2, 3, 4, 5, 6]:
            for examine_window_size in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
                spike_detection_thresholds = [i for i in range(examine_window_size+1)]
                for spike_detection_threshold in spike_detection_thresholds:
                    with open(TRAINING_LOG_NAME, "a") as f:
                        f.write(f"\tLIF Threshold Multiplier: {lif_threshold_multiplier}\n")
                        f.write(f"\tLIF Tau Multiplier: {lif_tau_multiplier}\n")
                        f.write(f"\tExamine Window Size: {examine_window_size}\n")
                        f.write(f"\tSpike Detection Threshold: {spike_detection_threshold}\n")
                    lif_tau = lif_tau_multiplier * (1 / sampling_rate)
                    lif_threshold = spike_amplitude * lif_threshold_multiplier

                    dv_u_hist, spike_train, dv_time_lif = dv_to_lif_spike_gen(
                        signal=raw_data,
                        lif_threshold=lif_threshold,
                        sampling_interval=sampling_interval,
                        lif_tau=lif_tau,
                        reset_mechanism=reset_mechanism
                    )

                    tp, fp, fn = 0, 0, 0
                    i, k = examine_window_size, 0
                    while i <= spike_train.shape[0] and k < gt_spike_times.shape[0]:
                        current_true_label_window_start, current_true_label_window_end = gt_spike_times[k] - 15, gt_spike_times[k] + 15

                        count = np.count_nonzero(spike_train[i-examine_window_size:i])
                        ap_detected =  (count > spike_detection_threshold)

                        within_window = current_true_label_window_start <= i <= current_true_label_window_end

                        if i > current_true_label_window_end:
                            k += 1
                        elif ap_detected and within_window: # ap is detected and is within the true label window
                            with open(DETECTION_IDX, "a") as f:
                                f.write(f"Detection idx: {i-1}, Spike time idx: {k}\n") # i-1 because of array indexing does not include the ith element
                            tp += 1
                            k += 1
                            i += skip_forward_window_size # skip forward by 0.5ms to avoid multi-counting the same AP
                        elif ap_detected and not within_window: # ap is detected but is outside the true label window
                            fp += 1
                            i += 1
                        elif ((i == current_true_label_window_end) and not ap_detected): # ap is not detected and we are at the end of the true label window
                            fn += 1
                            k += 1
                            i += 1
                        else: # everywhere else
                            i += 1

                    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                    with open(TRAINING_LOG_NAME, "a") as f:
                        f.write(f"\t\tNumber of APs in label: {gt_spike_times.shape[0]}\n")
                        f.write(f"\t\tAccuracy: {accuracy:.4f}. TP: {tp}, FP: {fp}, FN: {fn}\n")


    # # generate dv for all channels and plot some of them to verify
    # dv_all_channels = []
    # for i in tqdm(range(num_channels), desc="Generating DV for all channels"):
    #     dv_u_hist, dv_spike_train, dv_time_lif = dv_to_lif_spike_gen(
    #         signal=raw_data[i],
    #         lif_threshold=lif_threshold[i],
    #         sampling_interval=sampling_interval,
    #         lif_tau=lif_tau,
    #         reset_mechanism=reset_mechanism
    #     )
    #     dv_all_channels.append(dv_spike_train)

    # dv_all_channels = np.array(dv_all_channels)
    # plot_example_dv_trains(raw_data, dv_all_channels, gt_spike_times, spike_clusters, pc_features_ind)


