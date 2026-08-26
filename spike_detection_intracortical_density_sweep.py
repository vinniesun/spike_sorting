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

from utils import (
    IntracorticalDataset,
    load_dataset_intracortical,
    train_test_split_spike_detection,
    create_training_dataset_spike_detection,
    dv_to_lif_spike_gen,
)

from model import SpikeDetector

if __name__ == "__main__":
    """
        Dataset downloaded from: https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595?file=21819066
    """
    TRAINING_LOG_PATH = "./spike_detection_training_log"
    if not os.path.exists(TRAINING_LOG_PATH):
        os.makedirs(TRAINING_LOG_PATH)
    TRAINING_LOG_NAME = f"{TRAINING_LOG_PATH}/training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    SEED = 5673 # 1337, 5673, 1234
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    filepath = "./intracortical_dataset/"
    reset_mechanism = "subtract" # "none", "subtract", "zero"

    label_window_size = 3
    lif_threshold = 0.4
    lif_tau = 1 * (1/24000)

    examine_window_size = 8 # 1ms
    skip_forward_window_size = 12 # 0.5ms
    spike_detection_thresholds = [i for i in range(0, 10)] # number of events to be exceeded to be classified as an AP.

    for spike_detection_threshold in spike_detection_thresholds:
        with open(TRAINING_LOG_NAME, "a") as f:
            f.write(f"Seed Number: {SEED}\nevent density threshold: {spike_detection_threshold}\nexamin_window_size: {examine_window_size}\n")
            f.write(f"skip_forward_window_size: {skip_forward_window_size}\nlabel_window_size: {label_window_size}\n")
            f.write(f"lif_threshold: {lif_threshold}\nlif_tau: {lif_tau}\nreset_mechanism: {reset_mechanism}\n")
        for difficulty in ["Difficult1", "Difficult2", "Easy1", "Easy2"]:
            for noise_level in ["005", "01", "015", "02"]:
                filename = f"C_{difficulty}_noise{noise_level}.mat"

                with open(TRAINING_LOG_NAME, "a") as f:
                    f.write(f"Filename: {filename}.\n")

                signal, spike_class_label, spike_times, sampling_interval, \
                sampling_rate, spike_pulse_1ms_idx_length, spike_classes, \
                filtered_signal = load_dataset_intracortical(filepath, filename)

                dv_u_hist, spike_train, dv_time_lif = dv_to_lif_spike_gen(
                    signal=filtered_signal,
                    lif_threshold=lif_threshold,
                    sampling_interval=sampling_interval,
                    lif_tau=lif_tau,
                    reset_mechanism=reset_mechanism
                )

                ############ Verify spike window and gt spike time and spike signal matches

                # train_spike_train, train_spike_labels, \
                # test_spike_train, test_spike_labels, \
                # train_spike_times, test_spike_times = train_test_split_spike_detection(
                #     spike_train=spike_train,
                #     spike_times=spike_times,
                #     split_ratio=train_test_split_ratio,
                #     label_window=label_window_size
                # )

                # fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
                # time = np.arange(filtered_signal[train_spike_times[0]-48:train_spike_times[0]+48].shape[0]) # / 24000

                # # ax[0].plot(time, filtered_signal[train_spike_times[0]-48:train_spike_times[0]+48], color="blue", label=r"V(t)")
                # # ax[0].plot(time, train_spike_labels[train_spike_times[0]-48:train_spike_times[0]+48], color="red", label=r"GT")
                # # pos_idx = np.where(train_spike_train[train_spike_times[0]-48:train_spike_times[0]+48] > 0)[0]
                # # neg_idx = np.where(train_spike_train[train_spike_times[0]-48:train_spike_times[0]+48] < 0)[0]
                # # ax[0].eventplot(time[pos_idx], color="green", lineoffsets=2.0, linelengths=0.4)
                # # ax[0].eventplot(time[neg_idx], color='black', lineoffsets=2.0, linelengths=0.4)
                # # ax[0].legend(loc="lower left")
                # special_time = np.arange(filtered_signal[23290:23330].shape[0])
                # ax[0].plot(special_time, filtered_signal[23290:23330], color="blue", label=r"V(t)")
                # ax[0].plot(special_time, train_spike_labels[23290:23330], color="red", label=r"GT")
                # ax[0].plot(special_time, dv_u_hist[23290:23330], color="orange", label=r"u(t)")
                # pos_idx = np.where(spike_train[23290:23330] > 0)[0]
                # neg_idx = np.where(spike_train[23290:23330] < 0)[0]
                # ax[0].eventplot(special_time[pos_idx], color="green", lineoffsets=2.0, linelengths=0.4)
                # ax[0].eventplot(special_time[neg_idx], color='black', lineoffsets=2.0, linelengths=0.4)
                # ax[0].legend(loc="lower left")

                # ax[1].plot(time, filtered_signal[train_spike_times[3]-48:train_spike_times[3]+48], color="blue", label=r"V(t)")
                # ax[1].plot(time, train_spike_labels[train_spike_times[3]-48:train_spike_times[3]+48], color="red", label=r"\hat{y}(t)")
                # pos_idx = np.where(train_spike_train[train_spike_times[3]-48:train_spike_times[3]+48] > 0)[0]
                # neg_idx = np.where(train_spike_train[train_spike_times[3]-48:train_spike_times[3]+48] < 0)[0]
                # ax[1].eventplot(time[pos_idx], color="green", lineoffsets=2.0, linelengths=0.4)
                # ax[1].eventplot(time[neg_idx], color='black', lineoffsets=2.0, linelengths=0.4)
                # ax[1].legend(loc="lower left")

                # ax[2].plot(time, filtered_signal[train_spike_times[8]-48:train_spike_times[8]+48], color="blue", label=r"V(t)")
                # ax[2].plot(time, train_spike_labels[train_spike_times[8]-48:train_spike_times[8]+48], color="red", label=r"\hat{y}(t)")
                # pos_idx = np.where(train_spike_train[train_spike_times[8]-48:train_spike_times[8]+48] > 0)[0]
                # neg_idx = np.where(train_spike_train[train_spike_times[8]-48:train_spike_times[8]+48] < 0)[0]
                # ax[2].eventplot(time[pos_idx], color="green", lineoffsets=2.0, linelengths=0.4)
                # ax[2].eventplot(time[neg_idx], color='black', lineoffsets=2.0, linelengths=0.4)
                # ax[2].legend(loc="lower left")

                # ax[3].plot(time, filtered_signal[train_spike_times[23]-48:train_spike_times[23]+48], color="blue", label=r"V(t)")
                # ax[3].plot(time, train_spike_labels[train_spike_times[23]-48:train_spike_times[23]+48], color="red", label=r"\hat{y}(t)")
                # pos_idx = np.where(train_spike_train[train_spike_times[23]-48:train_spike_times[23]+48] > 0)[0]
                # neg_idx = np.where(train_spike_train[train_spike_times[23]-48:train_spike_times[23]+48] < 0)[0]
                # ax[3].eventplot(time[pos_idx], color="green", lineoffsets=2.0, linelengths=0.4)
                # ax[3].eventplot(time[neg_idx], color='black', lineoffsets=2.0, linelengths=0.4)
                # ax[3].legend(loc="lower left")

                # plt.tight_layout()
                # plt.savefig(f"./verify_spike_det_labeling/{difficulty}_noise{noise_level}.jpg", dpi=300)
                # plt.close()
                ############ End of verification

                tp, fp, fn = 0, 0, 0
                i, k = examine_window_size, 0
                while i <= spike_train.shape[0] and k < spike_times.shape[0]:
                    # if difficulty == "Difficult1" and noise_level == "005":
                    #     output = f"idx: {i:,} k: {k}, spike time: {spike_times[k]:,}, density of spikes in window: {np.count_nonzero(spike_train[i-examine_window_size:i])}, "
                    current_true_label_window_start, current_true_label_window_end = spike_times[k] - label_window_size, spike_times[k] + (3*label_window_size)

                    count = np.count_nonzero(spike_train[i-examine_window_size:i])
                    ap_detected =  (count > spike_detection_threshold)

                    within_window = current_true_label_window_start <= i <= current_true_label_window_end

                    if i > current_true_label_window_end:
                        k += 1
                    elif ap_detected and within_window: # ap is detected and is within the true label window
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
                    # print(f"Idx: {i:,}", end="\033[K\r") 
                    # if difficulty == "Difficult1" and noise_level == "005":
                    #     with open(TRAINING_LOG_NAME, "a") as f:
                    #         output += f"ap_detected: {ap_detected}, tp: {tp}, fp: {fp}, fn: {fn}, current window start: {current_true_label_window_start:,}, current window end: {current_true_label_window_end:,}\n"
                    #         f.write(output)

                    # if i == 661:
                    #     print(ap_detected, np.count_nonzero(spike_train[i-examine_window_size:i]), spike_detection_threshold, \
                    #           repr(spike_detection_threshold), repr(np.count_nonzero(spike_train[i-examine_window_size:i])))

                accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                with open(TRAINING_LOG_NAME, "a") as f:
                    f.write(f"Number of APs in label: {spike_times.shape[0]}\n")
                    f.write(f"Accuracy: {accuracy:.4f}. TP: {tp}, FP: {fp}, FN: {fn}\n")

        with open(TRAINING_LOG_NAME, "a") as f:
            f.write(f"\n\n")
                