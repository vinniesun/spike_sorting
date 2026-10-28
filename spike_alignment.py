import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import (
    dv_to_lif_spike_gen,
    find_alignment_point,
    generate_event_stream_dm,
    load_dataset_intracortical,
)

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

    alignment_window_half_dim = 12 # 0.5ms

    DETECTION_IDX = "Intracortical_Spike_Detection_IDX.txt"
    VISUALISATION_DIR = "./post_alignment_plots/"
    if not os.path.exists(VISUALISATION_DIR):
        os.makedirs(VISUALISATION_DIR)

    detection_idx_results_when_detected = {}
    detection_idx_results_label_spike_time = {}
    with open(DETECTION_IDX, "r") as f:
        detection_idx_lines = f.readlines()
    
    line_no = 0
    while line_no < len(detection_idx_lines):
        line = detection_idx_lines[line_no]
        if "Filename: " in line:
            filename = line.split("Filename: ")[1][:-1]
            detection_idx_results_when_detected[filename] = []
            detection_idx_results_label_spike_time[filename] = []
        else:
            temp = line.split(",")
            detected_idx = int(temp[0].split("Detection idx: ")[1])
            spike_time_idx = int(temp[1].split(" Spike time idx: ")[1])
            detection_idx_results_when_detected[filename].append(detected_idx)
            detection_idx_results_label_spike_time[filename].append(spike_time_idx)
    
        line_no += 1

    for difficulty in ["Difficult1", "Difficult2"]:
        for noise_level in ["005", "01", "015", "02"]:
            filename = f"C_{difficulty}_noise{noise_level}.mat"

            current_file_recon_path = VISUALISATION_DIR + filename[:-4] + "/"
            if not os.path.exists(current_file_recon_path):
                os.makedirs(current_file_recon_path)

            signal, spike_class_label, spike_times, sampling_interval, \
            sampling_rate, spike_pulse_1ms_idx_length, spike_classes, \
            filtered_signal = load_dataset_intracortical(filepath, filename)

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

            detected_spikes = np.array(detection_idx_results_label_spike_time[filename])
            detected_spike_times = np.array(detection_idx_results_when_detected[filename])
            all_spike_signals = {i: [] for i in spike_classes}
            all_spk_trains = {i: [] for i in spike_classes}
            for i in tqdm(range(len(spike_times))):
                if i in detected_spikes:
                    idx = np.where(detected_spikes == i)[0][0]

                    # First get the detection index 
                    idx_of_spike_detector = detected_spike_times[idx]
                    idx_of_spike_label = spike_times[i]

                    # Then extract the window to align
                    # We want the detected index point to sit right in the middle of the window
                    for_alignment_window = spike_train_dm[idx_of_spike_detector - alignment_window_half_dim : idx_of_spike_detector + alignment_window_half_dim + 1]

                    # Find the alignment point
                    alignment_point = find_alignment_point(for_alignment_window)

                    # Get the relative index of idx_of_spike_detector with respect to for_alignment_window
                    relative_idx = alignment_point - alignment_window_half_dim

                    # Get the new aligned index of the spike detector
                    new_idx_of_spike_detector = idx_of_spike_detector + relative_idx

                    # draw the aligned window
                    filtered_signal_aligned = filtered_signal[new_idx_of_spike_detector - 36 : new_idx_of_spike_detector + 36]
                    spk_train_dm_aligned = spike_train_dm[new_idx_of_spike_detector - 36 : new_idx_of_spike_detector + 36]
                    spk_train_dv_aligned = spike_train_dv[new_idx_of_spike_detector - 36 : new_idx_of_spike_detector + 36]

                    fig, ax = plt.subplots(3, 1, figsize=(10, 9))

                    ax[0].plot(filtered_signal_aligned, color='b', label="V(t)")
                    ax[0].axvline(x=36 - relative_idx, color='r', linestyle='--', label="Spk Detected Idx")
                    ax[0].axvline(x=36, color='g', linestyle='--', label="Aligned Idx")
                    
                    ax[1].stem(spk_train_dm_aligned, linefmt='b-', markerfmt='go', basefmt='r-', label=r"$S_{dm}(t)$")
                    ax[1].axvline(x=36 - relative_idx, color='r', linestyle='--', label="Spk Detected Idx")
                    ax[1].axvline(x=36, color='g', linestyle='--', label="Aligned Idx")
                    
                    ax[2].stem(spk_train_dv_aligned, linefmt='b-', markerfmt='go', basefmt='r-', label=r"$S_{dv}(t)$")
                    ax[2].axvline(x=36 - relative_idx, color='r', linestyle='--', label="Spk Detected Idx")
                    ax[2].axvline(x=36, color='g', linestyle='--', label="Aligned Idx")

                    ax[0].legend(loc="upper right")
                    ax[1].legend(loc="upper right")
                    ax[2].legend(loc="upper right")

                    ax[0].set_xlim([0, filtered_signal_aligned.shape[0] - 1])
                    ax[1].set_xlim([0, filtered_signal_aligned.shape[0] - 1])
                    ax[2].set_xlim([0, filtered_signal_aligned.shape[0] - 1])

                    ax[0].minorticks_on()
                    ax[0].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
                    ax[1].minorticks_on()
                    ax[1].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
                    ax[2].minorticks_on()
                    ax[2].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")

                    plt.tight_layout()
                    plt.savefig(f"{current_file_recon_path}train_{i}_class_{spike_class_label[i].item()}.jpg", dpi=300)
                    plt.close()
            