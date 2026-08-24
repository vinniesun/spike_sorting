import numpy as np
from scipy.io import loadmat
from scipy.signal import ellip, lfilter, butter

import torch
from torch.utils.data import Dataset

from typing import List, Tuple

class IntracorticalDataset(Dataset):
    def __init__(self, spikes: torch.Tensor, labels: torch.Tensor):
        self.spikes = spikes
        self.labels = labels

    def __len__(self):
        return self.spikes.shape[0]

    def __getitem__(self, idx):
        return self.spikes[idx], self.labels[idx]

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
            event_queue.append([i, sum(num_on_pulses), sum(num_off_pulses)])
        if len(event_queue) > 0:
            event_stream.append(event_queue)

    return np.array(event_stream).squeeze(axis=1)

def generate_event_stream_lif(filtered_signal, time_step, uth, lif_tau, if_reconstruct=False):
    spike_lif, time_lif, u_lif = lif_neuron(filtered_signal.T, time_step, uth, lif_tau)

    on_counts = np.zeros(filtered_signal.T.size)
    off_counts = np.zeros(filtered_signal.T.size)

    on_time = np.where(spike_lif > 0)
    off_time = np.where(spike_lif < 0)

    on_counts[np.array(on_time, dtype=int)] = 1
    off_counts[np.array(off_time, dtype=int)] = 1

    spike_train = on_counts - off_counts

    if if_reconstruct:
        return spike_train.T, spike_lif
    else:
        return spike_train.T

def leaky_integrate_neuron(U, time_step=1e-3, I=0, R=5, Urest=0, tau=5e-3):
    # tau = R*C
    U += (time_step/tau)*(-(U) + I*R) - Urest
    return U

def lif_neuron(filtered_signal, time_step=1e-3, uth=0.8, lif_tau=5e-3):
    U_trace = []
    spike_rec = []
    u = 0
    urest = 0
    time_lif = np.linspace(0, filtered_signal.shape[0], filtered_signal.shape[0], dtype=float)
    for step in range(filtered_signal.shape[0]):
        U_trace.append(u)
        u = leaky_integrate_neuron(u, time_step=time_step, I=filtered_signal[step], Urest=urest, tau=lif_tau)  # solve next step of U
        if u >= uth:
            urest = uth
            spike_rec.append(float(1))
        elif u<= -uth:
            urest = -uth
            spike_rec.append(float(-1))
        else:
            urest = 0
            spike_rec.append(float(0))

    return np.array(spike_rec), time_lif, U_trace

def load_dataset_intracortical(filepath: str, filename: str):
    complete_path = filepath + filename

    raw_data = loadmat(complete_path)
    # print(raw_data.keys())
    # print(raw_data["spike_class"].shape, raw_data["spike_class"]) # spike_class[0, 0] gives the spike class, and spike_times[0, 0] give the location for when that spike class occurs
    # print(raw_data["OVERLAP_DATA"].shape)
    # print(raw_data["data"].shape)
    # print(raw_data["startData"].shape)

    signal = raw_data["data"].squeeze() # shape (seq_len)
    spike_class_label = raw_data["spike_class"].squeeze()[0].squeeze()    # shape (num_of_spikes)
    spike_times = np.array(raw_data["spike_times"][0, 0].squeeze()) # shape (num_of_spikes)
    sampling_interval = raw_data["samplingInterval"][0, 0] * 1e-3
    sampling_rate = 1 / (sampling_interval) # 24kHz
    spike_pulse_1ms_idx_length = int(1e-3 / sampling_interval)

    spike_classes = np.unique(spike_class_label) # label is (1, 2, 3)

    spike_times += 24 # shift forward by 1ms

    order = 2
    rp = 0.1
    rs = 40
    wn = [300, 5000]
    normalised_wn = [(2*w) / (sampling_rate) for w in wn]
    b, a = ellip(order, rp, rs, normalised_wn, btype="bandpass")
    filtered_signal = lfilter(b, a, signal)

    return signal, spike_class_label, spike_times, sampling_interval, sampling_rate, spike_pulse_1ms_idx_length, spike_classes, filtered_signal

def reconstruction_lif(lif_data, time_step=1e-3, reconstruct_tau=0.05, alpha=0.4, order=2):
    step_total = lif_data.shape[0]
    u_rec = []
    u = 0

    # for step in range(step_total):
    #     decay = alpha * math.exp(-(time_step/reconstruct_tau))
    #     u_rec.append(u)
    #     u = (u + lif_data[step]) * decay
    
    rp = 0.1
    rs = 40
    cut_off_freq = 5000 # 5000 for intracortical. Try reducing this to make the reconstructed signal smoother.
    # A 2nd-order Butterworth filter has a relatively gentle roll-off. If high-frequency spike artifacts remain, you can increase the order:
    # to 4 or 6
    b, a = butter(order, 2*cut_off_freq/(1/time_step), btype='low')
    # b, a = ellip(order, rp, rs, 2*cut_off_freq/(1/time_step), btype="low")
    # b, a = bessel(order, 2*cut_off_freq/(1/time_step), btype='low')
    u_rec = lfilter(b, a, lif_data)

    return np.array(u_rec)

def calc_rmse(data, reconstructed_signal, spikeTimeGT):
    data_spk = np.array([])
    reconstructed_spk = np.array([])

    for i in spikeTimeGT[spikeTimeGT < np.size(data)]:
        evaluation_window = [i-12, i+48]
        reconstructed_spk_section = reconstructed_signal[evaluation_window[0]:evaluation_window[1]]

        data_spk_section = data[evaluation_window[0] - 1:evaluation_window[1] - 1]#Since the reconstructed signal is delayed by 1 sample
        # print("123", data_spk.shape, data_spk_section.shape)
        data_spk = np.concatenate((data_spk,data_spk_section))
        reconstructed_spk = np.concatenate((reconstructed_spk,reconstructed_spk_section))
    rmse = np.sqrt(np.mean((data_spk - reconstructed_spk)**2))
    # print(rmse)

    return rmse 

def train_test_split_spike_sorting(spike_classes, all_spk_trains, all_spike_signals, train_test_split_ratio):
    train_spk_train, test_spk_train = [], []
    train_signal, test_signal = [], []
    train_label, test_label = [], []
    for spike_class in spike_classes:
        idx = np.arange(len(all_spk_trains[spike_class]))
        np.random.shuffle(idx)
        train_idx = idx[:int(train_test_split_ratio * len(idx))]
        test_idx = idx[int(train_test_split_ratio * len(idx)):]
        for i in train_idx:
            train_spk_train.append(all_spk_trains[spike_class][i])
            train_signal.append(all_spike_signals[spike_class][i])
            train_label.append(spike_class)
        for i in test_idx:
            test_spk_train.append(all_spk_trains[spike_class][i])
            test_signal.append(all_spike_signals[spike_class][i])
            test_label.append(spike_class)

    return train_spk_train, test_spk_train, train_signal, test_signal, train_label, test_label

def reconstruct_DDM(event_counts, spike_amplitude):
    # print("sdafsdf", event_counts.shape)
    sig_length = np.shape(event_counts)[1]
    reconstructed_signal =  np.zeros(sig_length)
    reconstructed_signal[0] = 0
    for i in range(1,sig_length):
        current_value = reconstructed_signal[i-1]
        current_value = current_value + event_counts[0][i-1] * spike_amplitude
        current_value = current_value - event_counts[1][i-1] * spike_amplitude
        reconstructed_signal[i] = current_value
    return reconstructed_signal

def train_test_split_spike_sorting(
    spike_classes, 
    all_spk_trains, 
    all_spike_signals, 
    train_test_split_ratio
):
    train_spk_train, test_spk_train = [], []
    train_signal, test_signal = [], []
    train_label, test_label = [], []
    for spike_class in spike_classes:
        idx = np.arange(len(all_spk_trains[spike_class]))
        np.random.shuffle(idx)
        train_idx = idx[:int(train_test_split_ratio * len(idx))]
        test_idx = idx[int(train_test_split_ratio * len(idx)):]
        for i in train_idx:
            train_spk_train.append(all_spk_trains[spike_class][i])
            train_signal.append(all_spike_signals[spike_class][i])
            train_label.append(spike_class)
        for i in test_idx:
            test_spk_train.append(all_spk_trains[spike_class][i])
            test_signal.append(all_spike_signals[spike_class][i])
            test_label.append(spike_class)

    return train_spk_train, test_spk_train, train_signal, test_signal, train_label, test_label

def train_test_split_spike_detection(
    spike_train,
    spike_times,
    split_ratio=0.8,
    label_window=3
):
    # assuming spike times have already been shifted forward
    split_num = int(spike_times.shape[0] * split_ratio)

    train_spike_times = spike_times[:split_num]
    test_spike_times = spike_times[split_num:]

    spike_labels = np.zeros_like(spike_train) # shape (seq_len)

    for spike_time in spike_times:
        spike_labels[spike_time - label_window: spike_time + (3*label_window)] = 1

    train_spike_train = spike_train[:spike_times[split_num] + 24]
    train_spike_labels = spike_labels[:spike_times[split_num] + 24]

    test_spike_train = spike_train[spike_times[split_num] + 24:]
    test_spike_labels = spike_labels[spike_times[split_num] + 24:]
    test_spike_times -= train_spike_train.shape[0] # shift the test spike times to start from 0

    return train_spike_train, train_spike_labels, test_spike_train, test_spike_labels, train_spike_times, test_spike_times

def create_training_dataset_spike_detection(
    spike_train,
    spike_labels,
    spike_times,
    max_length=240,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    spike_samples = []
    spike_samples_labels = []
    for i in range(spike_times.shape[0]):
        spike_samples.append(torch.tensor(spike_train[spike_times[i] + 24 - max_length:spike_times[i] + 24], dtype=torch.float32))
        spike_samples_labels.append(torch.tensor(spike_labels[spike_times[i] + 24 - max_length:spike_times[i] + 24], dtype=torch.long))

    return spike_samples, spike_samples_labels

def reset_mech(reset_mechanism: str, u: float, lif_threshold) -> float:
    if reset_mechanism == "none":
        u_rest = 0
    elif reset_mechanism == "subtract":
        u_rest = lif_threshold
    elif reset_mechanism == "zero":
        u_rest = u
    else:
        raise ValueError(f"Invalid reset_mechanism: {reset_mechanism}. Must be one of 'none', 'subtract', or 'zero'.")

    return u_rest

def dv_to_lif_spike_gen(
    signal,
    lif_threshold,
    sampling_interval=1/24000,
    lif_tau=1/24000,
    reset_mechanism="none"
):
    length_of_signal = signal.shape[0]
    # dm specific variable
    last_reset_voltage = 0

    # lif specific variable
    u = 0
    u_rest = 0
    time_lif = np.linspace(0, length_of_signal, length_of_signal, dtype=np.float32)

    u_hist, spk_hist = [], []
    for i in range(length_of_signal):
        u_hist.append(u)

        dv = signal[i] - last_reset_voltage

        u = leaky_integrate_neuron(u, time_step=sampling_interval, I=dv, Urest=u_rest, tau=lif_tau)

        if u >= lif_threshold:
            # u_rest = 0 # not resetting seems to work the best
            # u_rest = 0 if reset_mechanism == "none" else (lif_threshold if reset_mechanism == "subtract" else u)
            u_rest = reset_mech(reset_mechanism, u, lif_threshold)
            spk_hist.append(float(1))
        elif u <= -lif_threshold:
            # u_rest = -0 # not resetting seems to work the best
            u_rest = reset_mech(reset_mechanism, -u, -lif_threshold)
            spk_hist.append(float(-1))
        else:
            u_rest = 0
            spk_hist.append(float(0))

        last_reset_voltage = signal[i]

    return np.array(u_hist), np.array(spk_hist), time_lif