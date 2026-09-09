import numpy as np
import matplotlib.pyplot as plt

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
    generate_event_stream_dm,
)

from model import SpikeDetector

if __name__ == "__main__":
    """
        Dataset downloaded from: https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595?file=21819066
    """

    filepath = "./intracortical_dataset/"
    reset_mechanism = "subtract" # "none", "subtract", "zero"

    label_window_size = 3
    lif_threshold = 0.4
    lif_tau = 1 * (1/24000)

    examine_window_size = 8 # 1ms
    skip_forward_window_size = 12 # 0.5ms
    spike_detection_threshold = 4 # number of events to be exceeded to be classified as an AP.

    dm_threshold = 0.2

    DETECTION_IDX = f"Intracortical_Spike_Detection_IDX.txt"

    detection_idx_results = {}
    # parse the detection_idx file to get the detection idx and spike time idx for each file
    with open(DETECTION_IDX, "r") as f:
        detection_idx_lines = f.readlines()

    line_no = 0
    while line_no < len(detection_idx_lines):
        line = detection_idx_lines[line_no]
        if "Filename: " in line:
            filename = line.split("Filename: ")[1][:-2]
            detection_idx_results[filename] = []
        else:
            temp = line.split(",")
            detected_idx = int(temp[0].split("Detection idx: ")[1])
            spike_time_idx = int(temp[1].split(" Spike time idx: ")[1])
            detection_idx_results[filename].append((detected_idx, spike_time_idx))

        line_no += 1

    for difficulty in ["Difficult1", "Difficult2", "Easy1", "Easy2"]:
        for noise_level in ["005", "01", "015", "02"]:
            filename = f"C_{difficulty}_noise{noise_level}.mat"

            if not os.path.exists(f"./spike_alignment/{filename[:-4]}/"):
                os.makedirs(f"./spike_alignment/{filename[:-4]}/")

            signal, spike_class_label, spike_times, sampling_interval, \
            sampling_rate, spike_pulse_1ms_idx_length, spike_classes, \
            filtered_signal = load_dataset_intracortical(filepath, filename)

            dv_u_hist, dv_spike_train, dv_time_lif = dv_to_lif_spike_gen(
                signal=filtered_signal,
                lif_threshold=lif_threshold,
                sampling_interval=sampling_interval,
                lif_tau=lif_tau,
                reset_mechanism=reset_mechanism
            )

            event_stream = generate_event_stream_dm(filtered_signal, dm_threshold, -dm_threshold)
            dm_spike_train = np.zeros_like(signal)
            dm_spike_train[event_stream[:, 0].astype(int)] = event_stream[:, 1] - event_stream[:, 2]

            ############ Verify spike window and gt spike time and spike signal matches
            for detected_idx, spike_time_idx in tqdm(detection_idx_results[filename], desc=f"Plotting spike alignment for {filename}"):
                # start_time, end_time = spike_times[detected_idx] - 8, spike_times[detected_idx] + 32
                start_time, end_time = detected_idx - 8, detected_idx + 32

                # shifted_detected_idx = detected_idx - start_time # this should be zero based
                shifted_detected_idx = 8

                fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

                time = np.arange(filtered_signal[start_time:end_time].shape[0]) # / 24000

                ax[0].plot(time, filtered_signal[start_time:end_time], color="blue", label=r"V(t)")

                dm_pos_idx = np.where(dm_spike_train[start_time:end_time] > 0)[0]
                dm_neg_idx = np.where(dm_spike_train[start_time:end_time] < 0)[0]

                dv_pos_idx = np.where(dv_spike_train[start_time:end_time] > 0)[0]
                dv_neg_idx = np.where(dv_spike_train[start_time:end_time] < 0)[0]

                ax[1].eventplot(time[dm_pos_idx], color="green", lineoffsets=2.0, linelengths=0.4, label="DM ON")
                ax[1].eventplot(time[dm_neg_idx], color='black', lineoffsets=2.0, linelengths=0.4, label="DM OFF")
                ax[1].eventplot(time[dv_pos_idx], color="red", lineoffsets=0.0, linelengths=0.4, label="DV ON")
                ax[1].eventplot(time[dv_neg_idx], color='blue', lineoffsets=0.0, linelengths=0.4, label="DV OFF")
                ax[1].axvline(x=shifted_detected_idx, color="orange", linestyle="--", label="Detected Spike Time")
                ax[1].legend(loc="lower left")

                ax[0].set_xlim(time[0], time[-1])
                ax[1].set_xlim(time[0], time[-1])

                ax[0].minorticks_on()
                ax[1].minorticks_on()

                ax[0].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
                ax[1].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")

                plt.tight_layout()
                plt.savefig(f"./spike_alignment/{filename[:-4]}/spike_time_idx_{spike_time_idx}.jpg", dpi=300)
                plt.close()
            ############ End of verification

            