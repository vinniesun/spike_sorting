"""
    When a neuron fires a spike, the waveform can be picked up by multiple channels.
    The goal for this script is to find all of the channels that's associated with 
    that particular neuron's activity within a time window, and based on some metric,
    determine which channel is the best channel to represent that neuron's activity.
    Then we can save the spike train of that channel for spike sorting.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def threshold_calculation(input_signal, threshold_multiplier):
    abs_threshold = np.median(np.abs(input_signal))

    data_up = np.copy(input_signal)
    data_up[data_up < abs_threshold] = 0
    peak, _ = find_peaks(data_up)

    data_down = np.copy(input_signal)
    data_down[data_down > -abs_threshold] = 0
    valley, _ = find_peaks(abs(data_down))

    median_peak = np.median(data_up[peak])
    median_valley = np.median(data_down[valley])
    spike_amplitude = (median_peak - median_valley) / 2

    delta_mod_threshold = threshold_multiplier*spike_amplitude

    return delta_mod_threshold

def generate_spike_train(filtered_signal, threshold_multiplier, num_channels=384, gt_window=3) -> Tuple[np.ndarray, np.ndarray]:
    complete_event_stream = [] # (384)
    for i in range(filtered_signal.shape[0]):
        input_signal = filtered_signal[i, :]

        threshold = threshold_calculation(input_signal, threshold_multiplier)
        on_threshold = threshold
        off_threshold = -threshold

        event_stream = generate_event_stream_dm(input_signal, on_threshold, off_threshold, bin_width=1)
        complete_event_stream.append(event_stream)

    spike_train = np.zeros_like(filtered_signal)
    for i, event_stream in enumerate(complete_event_stream):
        spike_train[i, event_stream[:, 0].astype(int)] = event_stream[:, 3] - event_stream[:, 4]

    # neo = np.power(filtered_signal, 2) - \
    #       np.multiply(
    #           np.hstack((np.zeros((num_channels, 1)), filtered_signal))[:, :-1],
    #           np.hstack((filtered_signal, np.zeros((num_channels, 1))))[:, :-1]
    #       )
    # neo = np.power(neo, 2) - \
    #       np.multiply(
    #           np.hstack((np.zeros((num_channels, 1)), neo))[:, :-1],
    #           np.hstack((neo, np.zeros((num_channels, 1))))[:, :-1]
    #       )
    # thd = np.mean(neo, axis=1).reshape(-1, 1) * neo_multiplier # shape: (384, 1)

    # # Original
    # # spd_label = np.where(neo > thd, 1, 0) # (384, 300000)

    # # New
    # spd_label = np.zeros_like(filtered_signal)
    # for i in range(num_channels):
    #     spk_idx = np.where(neo[i, :] > thd[i, :])[0]
    #     spk_idx += 24
    #     spk_idx = spk_idx.astype(int)
    #     for idx in spk_idx:
    #         spd_label[i, idx-gt_window:idx+gt_window+1] = 1

    return spike_train

def get_threshold_reset_counts(input_signal, last_reset_voltage, off_threshold, on_threshold, pulse, num_threshold_reset):
    if pulse == 1:
        bound = on_threshold
    else:
        bound = off_threshold
    
    num_threshold_reset += 1
    last_reset_voltage += bound

    if input_signal - last_reset_voltage > on_threshold:
        pulse = 1
        num_threshold_reset, last_reset_voltage = get_threshold_reset_counts(input_signal, last_reset_voltage, off_threshold, on_threshold, pulse, num_threshold_reset)
    elif input_signal - last_reset_voltage < off_threshold:
        pulse = -1
        num_threshold_reset, last_reset_voltage = get_threshold_reset_counts(input_signal, last_reset_voltage, off_threshold, on_threshold, pulse, num_threshold_reset)

    return num_threshold_reset, last_reset_voltage

def generate_event_stream_dm(input_signal, on_threshold, off_threshold, bin_width=1):
    last_reset_voltage = 0
    length_of_signal = input_signal.shape[0]

    event_stream = []
    for i in range(0, length_of_signal-bin_width+1, bin_width):
        event_queue = []
        signal_considered = input_signal[i:i+bin_width]
        num_on_pulses, num_off_pulses = [], []

        for j in range(signal_considered.shape[0]):
            num_threshold_reset = 0
            if signal_considered[j] - last_reset_voltage > on_threshold:
                pulse = 1
                num_threshold_reset, last_reset_voltage = get_threshold_reset_counts(signal_considered[j], last_reset_voltage, off_threshold, on_threshold, pulse, num_threshold_reset)
                num_on_pulses.append(num_threshold_reset)
                num_off_pulses.append(0)
            elif signal_considered[j] - last_reset_voltage < off_threshold:
                pulse = -1
                num_threshold_reset, last_reset_voltage = get_threshold_reset_counts(signal_considered[j], last_reset_voltage, off_threshold, on_threshold, pulse, num_threshold_reset)
                num_on_pulses.append(0)
                num_off_pulses.append(num_threshold_reset)
            else:
                num_on_pulses.append(0)
                num_off_pulses.append(0)
        if sum(num_on_pulses) + sum(num_off_pulses) > 0:
            event_queue.append([i, 1, 1, sum(num_on_pulses), sum(num_off_pulses)])
        if len(event_queue) > 0:
            event_stream.append(event_queue)

    return np.array(event_stream).squeeze(axis=1)

if __name__ == "__main__":
    filename = "AL036_2020-03-11_stripe240_NatIm_g0_t0.imec0.ap.bin"
    filepath = "./Neuropixel/AL036_2020-03-11/AL036_2020-03-11/ks"
    raw_filepath = "./Neuropixel/AL036_2020-03-11/AL036_2020-03-11/"

    raw_data = []
    for i in range(384):
        raw_data_channel = np.load(f"{raw_filepath}raw_data_filtered_channel_{i}.npy")
        raw_data.append(raw_data_channel)
    raw_data = np.array(raw_data)

    spike_train, spike_label = generate_spike_train(
        raw_data, 
        threshold_multiplier=0.2
    )