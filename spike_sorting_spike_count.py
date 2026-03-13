import numpy as np
import random
import os
import subprocess
import shutil
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator

import plotly.io as pio
import plotly.graph_objects as go
import plotly.tools as tls

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import snntorch as snn
import snntorch.functional as SF

from einops import repeat

from scipy.signal import ellip, lfilter, butter, find_peaks
from scipy.io import loadmat

import h5py 
import math
from typing import Tuple, Union, List, Optional

from tqdm import tqdm

from BRF.neurons import RAF, DTLIF
from BRF.grad_functions import StepDoubleGaussianGrad

def step_double_gaussian():
    def inner(x):
        return StepDoubleGaussianGrad.apply(x)
    return inner

class IntracorticalDataset(Dataset):
    def __init__(self, spikes: torch.Tensor, labels: torch.Tensor):
        self.spikes = spikes
        self.labels = labels

    def __len__(self):
        return self.spikes.shape[0]

    def __getitem__(self, idx):
        return self.spikes[idx], self.labels[idx]

class Model(nn.Module):
    def __init__(
            self, 
            in_channel: list[int],
            filters: List[int],
            kernel_sizes: List[int],
            strides: List[int],
            fc_input_dim: int,
            num_classes: int,
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv1d(
                in_channels=in_channel[i],
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i]
            ) for i in range(len(filters))]
        )
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, num_chan, seq_len = x.shape

        for i in range(len(self.convs)):
            x = self.relu(self.convs[i](x))

        x = x.view(bs, -1)
        x = self.fc1(x)

        return x

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
    U = (U) + (time_step/tau)*(-(U) + I*R) - Urest
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

def load_dataset(filepath: str, filename: str):
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

    order = 2
    rp = 0.1
    rs = 40
    wn = [300, 5000]
    normalised_wn = [(2*w) / (sampling_rate) for w in wn]
    b, a = ellip(order, rp, rs, normalised_wn, btype="bandpass")
    filtered_signal = lfilter(b, a, signal)

    return signal, spike_class_label, spike_times, sampling_interval, sampling_rate, spike_pulse_1ms_idx_length, spike_classes, filtered_signal

def train_test_split(spike_classes, all_spk_trains, all_spike_signals, train_test_split_ratio):
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

def plot_sorted_spike_signals(signals: List, sorted_class: int):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for signal in signals:
        ax.plot(signal)
    ax.set_title(f"Sorted Spike Signals for Class {sorted_class}")
    plt.savefig(f"Sorted_Spike_Signals_Class_{sorted_class}.jpg")
    plt.close()

def plot_single_spike_signal_with_dm(input_signal: np.ndarray, spike_train: np.ndarray, sorted_class: int, spk_train_filtered: Optional[np.ndarray]=None, specify_name: Optional[str]=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(input_signal, color="red")
    # ax.stem(spike_train*0.1, markerfmt=" ")
    if spk_train_filtered is None:
        pos_loc = np.where(spike_train > 0)[0]
        neg_loc = np.where(spike_train < 0)[0]
        ax.eventplot(pos_loc, lineoffsets=0, linelengths=0.5, colors="blue")
        ax.eventplot(neg_loc, lineoffsets=0, linelengths=0.5, colors="green")
    else:
        pos_loc = np.where(spike_train > 0)[0]
        neg_loc = np.where(spike_train < 0)[0]
        ax.eventplot(pos_loc, lineoffsets=0.5, linelengths=0.5, colors="blue")
        ax.eventplot(neg_loc, lineoffsets=0.5, linelengths=0.5, colors="green")

        pos_loc_filt = np.where(spk_train_filtered > 0)[0]
        neg_loc_filt = np.where(spk_train_filtered < 0)[0]
        ax.eventplot(pos_loc_filt, lineoffsets=-0.5, linelengths=0.5, colors="blue")
        ax.eventplot(neg_loc_filt, lineoffsets=-0.5, linelengths=0.5, colors="green")
    plt.grid(visible=True, which="major", axis="both", alpha=0.5, color="gray")
    ax.minorticks_on()
    plt.grid(visible=True, which="minor", axis="both", alpha=0.4, color="lightgray")
    if specify_name is not None:
        if "train" in specify_name:
            plt.savefig(f"{TRAIN_SAMPLES_OUTPUT_PATH}{specify_name}.jpg")
        else:
            plt.savefig(f"{TEST_SAMPLES_OUTPUT_PATH}{specify_name}.jpg")
    else:
        plt.savefig(f"Single_Spike_Signal_with_DM_Class_{sorted_class}.jpg")
    plt.close()

def calc_t1_t2_interval_filter(spk_train: np.ndarray, sampling_interval, spike_pulse_1ms_idx_length, spk_train_id=None, idle_threshold=8):
    spk_train_new = spk_train.copy()
    i = 0
    start_of_spike = True
    first_idx = None
    second_idx = None
    third_idx = None
    change_in_polarity = 0
    t1, t2 = None, None
    idle_count = 0
    found_t1_t2 = False
    while i < len(spk_train):
        # tqdm.write(f"idx {i}: idle count: {idle_count}")
        # print(f"idx {i}: idle count: {idle_count}, {first_idx}, {second_idx}, {third_idx}, {t1}, {t2}, {change_in_polarity}, {found_t1_t2}")
        if found_t1_t2:
            spk_train_new[i] = 0
        else:
            if spk_train[i] != 0:
                # 1. Check if it's the start of a spike pulse
                if start_of_spike:
                    first_idx = i
                    start_of_spike = False
                else:
                    # 2. Check for idleness (This only applies before the start of t1 is found)
                    # 2.1 Check if t1 has been found
                    # if idle_count >= idle_threshold and t1 == 0.0:
                    #     spk_train_new[first_idx] = 0
                    #     idle_count = 0
                    #     first_idx = i
                    # 3. Check if duration has exceeded the 1ms pulse width
                    ######## Need to account for class 1, where the spike pulse is shorter, and the 2ms window may include noise.
                    ######## One way to account for this is to check for idleness of the final polarity, if there's long enough idleness,
                    ######## We just use the first spike of that polarity to calculate t2
                    ######## i.e. the idleness tracker kicks in again when change_in_polarity == 2
                    if i - first_idx > spike_pulse_1ms_idx_length:
                        if t1 is None and first_idx is not None and second_idx is not None:
                            t1 = (second_idx - first_idx) * sampling_interval
                        if t2 is None and second_idx is not None and third_idx is not None:
                            t2 = (third_idx - second_idx) * sampling_interval
                            spk_train_new[i] = 0
                            found_t1_t2 = True
                    # 3.1 otherwise check for change in polarity
                    else:
                        if change_in_polarity == 0:
                            # 3.1.1 Look for first change in polarity
                            if np.sign(spk_train[i]) != np.sign(spk_train[first_idx]):
                                second_idx = i
                                change_in_polarity += 1
                            else:
                                spk_train_new[i] = 0
                        elif change_in_polarity == 1:
                            # 3.1.2 Look for second change in polarity and calculate t1
                            if np.sign(spk_train[i]) != np.sign(spk_train[second_idx]):
                                t1 = (second_idx - first_idx) * sampling_interval
                                third_idx = i
                                change_in_polarity += 1
                            else:
                                spk_train_new[second_idx] = 0
                                second_idx = i
                        elif change_in_polarity == 2:
                            # 3.1.3 Look for third change in polarity and calculate t2
                            if np.sign(spk_train[i]) != np.sign(spk_train[third_idx]):
                                t2 = (third_idx - second_idx) * sampling_interval
                                change_in_polarity += 1
                                found_t1_t2 = True
                                spk_train_new[i] = 0
                                # print("Found t1 and t2")
                            else:
                                if idle_count >= idle_threshold:
                                    t2 = (third_idx - second_idx) * sampling_interval
                                    found_t1_t2 = True
                                    spk_train_new[i] = 0
                                else:
                                    spk_train_new[third_idx] = 0
                                    third_idx = i
                idle_count = 0
            else:
                if not start_of_spike:
                    # print("No activity...")
                    idle_count += 1
                if idle_count >= idle_threshold:
                    if first_idx is not None and change_in_polarity == 0:
                        # print("Resetting 1...")
                        spk_train_new[first_idx] = 0
                        start_of_spike = True
                        first_idx = None
                        second_idx = None
                        third_idx = None
                        change_in_polarity = 0
                    elif third_idx is not None and change_in_polarity == 2:
                        t2 = (third_idx - second_idx) * sampling_interval
                        found_t1_t2 = True
                    idle_count = 0
            # print(f"Index: {i}, spk_train content: {spk_train[i]}, 1st spk index: {first_idx}, \
            #       2nd spk index: {second_idx}, 3rd spk index: {third_idx}, # of changes in polarity: {change_in_polarity}, \
            #       t1: {t1}, t2: {t2}, idleness: {idle_count}, start of spike: {start_of_spike}")
        i += 1

    if first_idx is not None and second_idx is not None and t1 is None:
        t1 = (second_idx - first_idx) * sampling_interval

    if second_idx is not None and third_idx is not None and t2 is None:
        t2 = (third_idx - second_idx) * sampling_interval

    if spk_train_id is not None:
        if t1 is None:
            print(f"Spike train ID {spk_train_id} has t1 as None")
        if t2 is None:
            print(f"Spike train ID {spk_train_id} has t2 as None")

    return t1, t2, spk_train_new

def plot_histogram(values, counts, spk_class, t1_or_t2="t1"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.bar(values, counts, color='blue', width=1e-5)
    ax.set_xlabel("delta t-1 (seconds)")
    ax.set_title(f"Channel {i+1}")
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(f'./interspike_interval_histogram_{spk_class}_{t1_or_t2}.jpg')

def plot_density(xs: List, ys: List, size: List, spike_class: int, is_test_set: bool=False):
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.scatter(xs, ys, s=size, alpha=0.5)
    ax.set_xlabel("t1")
    ax.set_ylabel("t2")
    ax.set_xlim([0, max(xs)])
    ax.set_ylim([0, max(ys)])
    plt.tight_layout()
    if is_test_set:
        plt.savefig(f"./t1_t2_density_spike_class_{spike_class}_test.jpg")
    else:
        plt.savefig(f"./t1_t2_density_spike_class_{spike_class}.jpg")

def plot_density_complete(xs: List, ys: List, size: List, spike_classes: List, is_test_set: bool=False):
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    scatter = ax.scatter(xs, ys, s=size, alpha=0.5, c=spike_classes, cmap='berlin')
    ax.set_xlabel("t1")
    ax.set_ylabel("t2")
    ax.set_xlim([0, max(xs)])
    ax.set_ylim([0, max(ys)])
    ax.legend(*scatter.legend_elements(), loc="best")
    plt.tight_layout()
    if is_test_set:
        plt.savefig(f"./t1_t2_density_complete_test.jpg", dpi=2000)
    else:
        plt.savefig(f"./t1_t2_density_complete.jpg", dpi=2000)

def train(
    net,
    test_net,
    train_loader,
    test_loader,
    optimiser,
    loss_fn,
    acc_fn,
    scheduler=None,
):
    best_acc = 0.0
    best_loss = float('inf')
    # for epoch in tqdm(range(NUM_EPOCHS)):
    for epoch in range(NUM_EPOCHS):
        net.train()
        curr_loss = 0.0
        for data, label in train_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            pred = net(data.unsqueeze(1))

            loss = loss_fn(pred, label)
            curr_loss += loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        correct_samples, total_samples = 0, 0
        for data, label in train_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            pred = net(data.unsqueeze(1))

            correct, total = acc_fn(pred, label)
            correct_samples += correct
            total_samples += total

        train_acc = correct_samples / total_samples
        # tqdm.write(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Accuracy: {train_acc:.4f}, Loss: {curr_loss:.4f}")

        if train_acc > best_acc:
            torch.save(net.state_dict(), MODEL_FILENAME)
            best_acc = train_acc
        # if curr_loss < best_loss:
        #     torch.save(net.state_dict(), MODEL_FILENAME)
        #     best_loss = curr_loss
        # torch.save(net.state_dict(), MODEL_FILENAME)

        # test(test_net, test_loader, acc_fn)

        if scheduler is not None:
            scheduler.step()

def test(
    net,
    test_loader,
    acc_fn,
    final_test: bool=False,
):
    net.load_state_dict(torch.load(MODEL_FILENAME, weights_only=True))
    net.to(DEVICE)
    net.eval()
    correct_samples, total_samples = 0, 0
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            pred = net(data.unsqueeze(1))

            correct, total = acc_fn(pred, label)
            correct_samples += correct
            total_samples += total

    test_acc = correct_samples / total_samples
    if final_test:
        # tqdm.write(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"\t\tFinal Test Accuracy: {test_acc:.4f}")

def classification_acc_fn(output, label):
    _, pred = output.max(1)
    correct = pred.eq(label).sum().item()

    return correct, label.shape[0]

def first_to_spike_acc_function(spk_out):
    spk_time = (
        spk_out.transpose(0, -1)
        * (torch.arange(0, spk_out.size(0)).detach().to(spk_out.device) + 1)
    ).transpose(0, -1)

    first_spike_time = torch.zeros_like(spk_time[0])
    for step in range(spk_time.size(0)):
        first_spike_time += (
            spk_time[step] * ~first_spike_time.bool()
        )  # mask out subsequent spikes

    # Override element 0 (no spike) with shadow spike at final time step,
    # then offset by -1 s.t. first_spike is at t=0
    first_spike_time += ~first_spike_time.bool() * (spk_time.size(0))
    first_spike_time -= 1  # fix offset

    # take idx of torch.min, see if it matches targets
    _, idx = first_spike_time.min(1)

    return idx

def plot_test_samples(net, test_spikes_tensor: torch.Tensor, test_labels_tensor: torch.Tensor, test_signal: List):
    net.load_state_dict(torch.load(MODEL_FILENAME, weights_only=True))
    net.to("cpu")
    net.eval()

    with torch.no_grad():
        for i in tqdm(range(test_spikes_tensor.shape[0])):
            data = test_spikes_tensor[i].unsqueeze(0)
            label = test_labels_tensor[i]

            spk_out, mem_out = net(data)
            pred = first_to_spike_acc_function(spk_out)

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.plot(test_signal[i], color="red")
            # ax.stem(spike_train*0.1, markerfmt=" ")
            data = data.squeeze(0)
            pos_loc = np.where(data > 0)[0]
            neg_loc = np.where(data < 0)[0]
            ax.eventplot(pos_loc, lineoffsets=0, linelengths=0.5, colors="blue")
            ax.eventplot(neg_loc, lineoffsets=0, linelengths=0.5, colors="green")

            plt.grid(visible=True, which="major", axis="both", alpha=0.5, color="gray")
            ax.minorticks_on()
            plt.grid(visible=True, which="minor", axis="both", alpha=0.4, color="lightgray")
            plt.savefig(f"{PREDICTION_OUTPUT_PATH}signal_{i}_predicted_{pred.detach().item()}_actual_{label.item()}.jpg")
            plt.close()

def plot_train_samples(net, train_spikes_tensor: torch.Tensor, train_labels_tensor: torch.Tensor, train_signal: List):
    net.load_state_dict(torch.load(MODEL_FILENAME, weights_only=True))
    net.to("cpu")
    net.eval()

    with torch.no_grad():
        for i in tqdm(range(train_spikes_tensor.shape[0])):
            data = train_spikes_tensor[i].unsqueeze(0)
            label = train_labels_tensor[i]

            spk_out, mem_out = net(data)
            pred = first_to_spike_acc_function(spk_out)

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.plot(train_signal[i], color="red")
            # ax.stem(spike_train*0.1, markerfmt=" ")
            data = data.squeeze(0)
            pos_loc = np.where(data > 0)[0]
            neg_loc = np.where(data < 0)[0]
            ax.eventplot(pos_loc, lineoffsets=0, linelengths=0.5, colors="blue")
            ax.eventplot(neg_loc, lineoffsets=0, linelengths=0.5, colors="green")

            plt.grid(visible=True, which="major", axis="both", alpha=0.5, color="gray")
            ax.minorticks_on()
            plt.grid(visible=True, which="minor", axis="both", alpha=0.4, color="lightgray")
            plt.savefig(f"{TRAINING_PRED_OUTPUT_PATH}signal_{i}_predicted_{pred.detach().item()}_actual_{label.item()}.jpg")
            plt.close()

def clean_images(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Cleaned up folder: {folder_path}")
    os.makedirs(folder_path, exist_ok=True)

def reconstruct_DDM(event_counts, spike_amplitude):
    # print("sdafsdf", event_counts.shape)
    sig_length = np.shape(event_counts)[1]
    reconstructed_signal =  np.zeros(sig_length)
    reconstructed_signal[0] = 0
    for i in range(1,sig_length):
        current_value = reconstructed_signal[i-1]
        current_value = current_value + event_counts[0][i-1] * spike_amplitude
        current_value = current_value - event_counts[1][i-1] * spike_amplitude
        reconstructed_signal[i] = current_value[0]
    return reconstructed_signal

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
    cut_off_freq = 5000 # 5000 for intracortical
    b, a = butter(order, 2*cut_off_freq/(1/time_step), btype='low')
    # b, a = ellip(order, rp, rs, 2*cut_off_freq/(1/time_step), btype="low")
    # b, a = bessel(order, 2*cut_off_freq/(1/time_step), btype='low')
    u_rec = lfilter(b, a, lif_data)

    return np.array(u_rec)

if __name__ == "__main__":
    """
        Dataset downloaded from: https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595?file=21819066
    """
    BATCH_SIZE = 256
    NUM_EPOCHS= 30
    MODEL_FILENAME = f"./intracortical_best_model.pth"
    TEST_SAMPLES_OUTPUT_PATH = "./test_samples_output/"
    # clean_images(TEST_SAMPLES_OUTPUT_PATH)
    TRAIN_SAMPLES_OUTPUT_PATH = "./train_samples_output/"
    # clean_images(TRAIN_SAMPLES_OUTPUT_PATH)
    PREDICTION_OUTPUT_PATH = "./prediction_plots/"
    # clean_images(PREDICTION_OUTPUT_PATH)
    TRAINING_PRED_OUTPUT_PATH = "./training_prediction_plots/"
    # clean_images(TRAINING_PRED_OUTPUT_PATH)

    SEED = 1337 # 1234, 1337, 5673
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    use_dm = "none"
    use_reconstructed = True
    reconstruct_dm = False
    # for dm, the threshold is: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
    # for lif, the threshold is: 0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0
    thresholds = np.array([3.0]) # to do 0.6
    print(f"Current threshold is: {thresholds}")
    window_size = 70
    train_test_split_ratio = 0.5

    # gt_noise_level = "015" # 005, 01, 015, 02
    # difficulty = "Difficult2"   # Difficult1, Difficult2, Easy1, Easy2
    filepath = "./intracortical_dataset/"
    for difficulty in ["Easy1", "Easy2", "Difficult1", "Difficult2"]:
        for gt_noise_level in ["005", "01", "015", "02"]:
            filename = f"C_{difficulty}_noise{gt_noise_level}.mat"
            print(f"\tCurrent filename is: {filename}")

            signal, spike_class_label, spike_times, sampling_interval, sampling_rate, spike_pulse_1ms_idx_length, spike_classes, filtered_signal = load_dataset(filepath, filename)

            on_threshold = thresholds
            off_threshold = -thresholds

            if use_dm == "dm":
                event_stream = generate_event_stream_dm(filtered_signal, on_threshold, off_threshold)
                spike_train = np.zeros_like(signal)
                spike_train[event_stream[:, 0].astype(int)] = event_stream[:, 3] - event_stream[:, 4]
            elif use_dm == "lif":
                spike_train = generate_event_stream_lif(filtered_signal, sampling_interval, uth=thresholds, lif_tau=sampling_interval, if_reconstruct=False)
            elif use_dm == "none":
                if use_reconstructed:
                    if reconstruct_dm:
                        event_stream = generate_event_stream_dm(filtered_signal, on_threshold, off_threshold)
                        spike_train = np.zeros((filtered_signal.shape[0], 2))
                        spike_train[event_stream[:, 0].astype(int), 0] = event_stream[:, 3]
                        spike_train[event_stream[:, 0].astype(int), 1] = event_stream[:, 4]

                        spike_train = reconstruct_DDM(spike_train.T, thresholds)
                        print("\tFinished DM Reconstruction.")
                    else:
                        spike_train, event_stream = generate_event_stream_lif(filtered_signal, sampling_interval, uth=thresholds, lif_tau=sampling_interval, if_reconstruct=True)
                        spike_train = reconstruction_lif(event_stream, time_step=sampling_interval, reconstruct_tau=10*sampling_interval, alpha=0.8, order=2)
                        print("\tFinished LIF Reconstruction.")
                else:
                    spike_train = filtered_signal

            all_spike_signals = {i: [] for i in spike_classes}
            all_spk_trains = {i: [] for i in spike_classes}
            for i in range(len(spike_times)):
                all_spike_signals[spike_class_label[i]].append(filtered_signal[spike_times[i]:spike_times[i] + window_size])
                all_spk_trains[spike_class_label[i]].append(spike_train[spike_times[i]:spike_times[i] + window_size])

            ################ Plot to verify ################
            # for spike_class in spike_classes:
                # plot_sorted_spike_signals(all_spike_signals[spike_class], spike_class)
            # plot_single_spike_signal_with_dm(all_spike_signals[spike_classes[2]][0], all_spk_trains[spike_classes[2]][0], spike_classes[2])

            # Split into train & test set by class
            train_spk_train, test_spk_train, train_signal, test_signal, train_label, test_label = train_test_split(spike_classes, all_spk_trains, all_spike_signals, train_test_split_ratio)

            
        print("")
