import argparse
import copy
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *
from utils import (
    IntracorticalDataset,
    dv_to_lif_spike_gen,
    generate_event_stream_dm,
    generate_event_stream_lif,
    load_dataset_intracortical,
    train_test_split_spike_sorting,
)


if __name__ == "__main__":
    """
        Dataset downloaded from: https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595?file=21819066
    """
    parser = argparse.ArgumentParser(description="Spike Detection on Neuropixel")
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed"
    )  # 1337, 5673, 1234
    parser.add_argument(
        "--encoder_th",
        type=float,
        default=0.1,
        help="Encoder threshold for the DM or DV encoder",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="dm",
        choices=["dm", "dv", "lif", "none"],
        help="Encoder type for the DM or DV encoder",
    )
    
    args = parser.parse_args()
    
    TRAINING_LOG_PATH = "./spike_sorting_training_log"
    if not os.path.exists(TRAINING_LOG_PATH):
        os.makedirs(TRAINING_LOG_PATH)
    
    SEED = args.seed  # 1337, 5673, 1234
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    filepath = "./intracortical_dataset/"
    
    encoder = args.encoder  # "dm" or "dv"
    reset_mechanism = "subtract"  # "none", "subtract", "zero"
    train_test_split_ratio = 0.7  # 70% training, 30% testing
    encoder_threshold = args.encoder_th
    lif_tau = 1 * (1 / 24000)
    detection_window_size = 8
    sorting_window_size = 32  # 2ms
    visualisation_window = 50

    DETECTION_IDX = "Intracortical_Spike_Detection_IDX.txt"
    VISUALISATION_DIR = "./encoder_alignment_plots/"
    if not os.path.exists(VISUALISATION_DIR):
        os.makedirs(VISUALISATION_DIR)
    
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
    
    (
        complete_train_data,
        complete_train_labels,
        complete_test_data,
        complete_test_labels,
    ) = [], [], {}, {}
    for difficulty in ["Difficult2", "Difficult1", "Easy1", "Easy2"]:
        for noise_level in ["005", "01", "015", "02"]:
            filename = f"C_{difficulty}_noise{noise_level}.mat"
    
            MODEL_FILENAME_ACC = f"./intracortical_weights/spike_sorting_kmeans_{filename[:-4]}_best_model_acc.pth"
            # MODEL_FILENAME_LOSS = f"./intracortical_weights/spike_sorting_ae_{filename[:-4]}_best_model_loss.pth"
            MODEL_FILENAME_LOSS = f"./intracortical_weights/spike_sorting_ae_{difficulty}_best_model_loss.pth"
            # MODEL_FILENAME_ACC = "./intracortical_weights/spike_sorting_kmeans_best_model_acc.pth"
            # MODEL_FILENAME_LOSS = "./intracortical_weights/spike_sorting_ae_best_model_loss.pth"
    
            current_file_recon_path = VISUALISATION_DIR + filename[:-4] + "/"
            if not os.path.exists(current_file_recon_path):
                os.makedirs(current_file_recon_path)
    
            (
                signal,
                spike_class_label,
                spike_times,
                sampling_interval,
                sampling_rate,
                spike_pulse_1ms_idx_length,
                spike_classes,
                filtered_signal,
            ) = load_dataset_intracortical(filepath, filename)
    
            on_threshold = 0.2
            off_threshold = -0.2
            event_stream = generate_event_stream_dm(
                filtered_signal, on_threshold, off_threshold
            )
            spike_train_dm = np.zeros_like(signal)
            spike_train_dm[event_stream[:, 0].astype(int)] = (
                event_stream[:, 1] - event_stream[:, 2]
            )
            dv_u_hist, spike_train_dv, dv_time_lif = dv_to_lif_spike_gen(
                signal=filtered_signal,
                lif_threshold=0.4,
                sampling_interval=sampling_interval,
                lif_tau=lif_tau,
                reset_mechanism=reset_mechanism,
            )
            # for lif, the threshold is: 0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0
            spike_train_lif = generate_event_stream_lif(
                filtered_signal,
                time_step=sampling_interval,
                uth=1.2,
                lif_tau=sampling_interval,
                if_reconstruct=False
            )

            detected_spikes = np.array(detection_idx_results_label_spike_time[filename])
            detected_spike_times = np.array(detection_idx_results_when_detected[filename])
            all_spike_signals = {i: [] for i in spike_classes}
            all_spk_trains = {i: [] for i in spike_classes}
            for i in range(len(spike_times)):
                if i in detected_spikes:
                    idx = np.where(detected_spikes == i)[0][0]
                    signal_of_interest = filtered_signal[spike_times[idx] - visualisation_window : spike_times[idx] + visualisation_window]
                    spike_train_of_interest_dm = spike_train_dm[spike_times[idx] - visualisation_window : spike_times[idx] + visualisation_window]
                    spike_train_of_interest_dv = spike_train_dv[spike_times[idx] - visualisation_window : spike_times[idx] + visualisation_window]
                    spike_train_of_interest_lif = spike_train_lif[spike_times[idx] - visualisation_window : spike_times[idx] + visualisation_window]

                    idx_of_spike_detector = detected_spike_times[idx] - spike_times[i] + visualisation_window
                    idx_of_spike_label = spike_times[i] - spike_times[i] + visualisation_window
    
                    # plot the signal about the label spike time
                    # generate a vertical line for when spike is detected
                    # plot the corresponding spike train
                    fig, ax = plt.subplots(4, 1, figsize=(10, 12))
    
                    ax[0].plot(signal_of_interest, color='b', label="V(t)")
                    ax[0].axvline(x=idx_of_spike_label, color='r', linestyle='--', label="Label Spike Time")
                    ax[0].axvline(x=idx_of_spike_detector, color='g', linestyle='--', label="Detected Spike Time")

                    ax[1].stem(spike_train_of_interest_dm, linefmt='b-', markerfmt='go', basefmt='r-', label=r"$S_{dm}(t)$")
                    ax[1].axvline(x=idx_of_spike_label, color='r', linestyle='--', label="Label Spike Time")
                    ax[1].axvline(x=idx_of_spike_detector, color='g', linestyle='--', label="Detected Spike Time")

                    ax[2].stem(spike_train_of_interest_dv, linefmt='b-', markerfmt='go', basefmt='r-', label=r"$S_{dv}(t)$")
                    ax[2].axvline(x=idx_of_spike_label, color='r', linestyle='--', label="Label Spike Time")
                    ax[2].axvline(x=idx_of_spike_detector, color='g', linestyle='--', label="Detected Spike Time")

                    ax[3].stem(spike_train_of_interest_lif, linefmt='b-', markerfmt='go', basefmt='r-', label=r"$S_{lif}(t)$")
                    ax[3].axvline(x=idx_of_spike_label, color='r', linestyle='--', label="Label Spike Time")
                    ax[3].axvline(x=idx_of_spike_detector, color='g', linestyle='--', label="Detected Spike Time")

                    ax[0].legend(loc="upper right")
                    ax[1].legend(loc="upper right")
                    ax[2].legend(loc="upper right")
                    ax[3].legend(loc="upper right")

                    ax[0].minorticks_on()
                    ax[0].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
                    ax[1].minorticks_on()
                    ax[1].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
                    ax[2].minorticks_on()
                    ax[2].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
                    ax[3].minorticks_on()
                    ax[3].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
    
                    plt.tight_layout()
                    plt.savefig(f"{current_file_recon_path}train_{i}_class_{spike_class_label[i].item()}.jpg", dpi=300)
                    plt.close()
                else:
                    signal_of_interest = filtered_signal[spike_times[idx] - visualisation_window : spike_times[idx] + visualisation_window]
                    spike_train_of_interest_dm = spike_train_dm[spike_times[idx] - visualisation_window : spike_times[idx] + visualisation_window]
                    spike_train_of_interest_dv = spike_train_dv[spike_times[idx] - visualisation_window : spike_times[idx] + visualisation_window]
                    spike_train_of_interest_lif = spike_train_lif[spike_times[idx] - visualisation_window : spike_times[idx] + visualisation_window]
                    
                    idx_of_spike_label = spike_times[i] - spike_times[i] + visualisation_window
                    
                    # plot the signal about the label spike time
                    # generate a vertical line for when spike is detected
                    # plot the corresponding spike train
                    fig, ax = plt.subplots(4, 1, figsize=(10, 12))
                    
                    ax[0].plot(signal_of_interest, color='b', label="V(t)")
                    ax[0].axvline(x=idx_of_spike_label, color='r', linestyle='--', label="Label Spike Time")
                    
                    ax[1].stem(spike_train_of_interest_dm, linefmt='b-', markerfmt='go', basefmt='r-', label=r"$S_{dm}(t)$")
                    ax[1].axvline(x=idx_of_spike_label, color='r', linestyle='--', label="Label Spike Time")
                    
                    ax[2].stem(spike_train_of_interest_dv, linefmt='b-', markerfmt='go', basefmt='r-', label=r"$S_{dv}(t)$")
                    ax[2].axvline(x=idx_of_spike_label, color='r', linestyle='--', label="Label Spike Time")
                    
                    ax[3].stem(spike_train_of_interest_lif, linefmt='b-', markerfmt='go', basefmt='r-', label=r"$S_{lif}(t)$")
                    ax[3].axvline(x=idx_of_spike_label, color='r', linestyle='--', label="Label Spike Time")
                    
                    ax[0].legend(loc="upper right")
                    ax[1].legend(loc="upper right")
                    ax[2].legend(loc="upper right")
                    ax[3].legend(loc="upper right")
                    
                    ax[0].minorticks_on()
                    ax[0].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
                    ax[1].minorticks_on()
                    ax[1].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
                    ax[2].minorticks_on()
                    ax[2].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
                    ax[3].minorticks_on()
                    ax[3].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
                    
                    plt.tight_layout()
                    plt.savefig(f"{current_file_recon_path}train_{i}_class_{spike_class_label[i].item()}.jpg", dpi=300)
                    plt.close()
    