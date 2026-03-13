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
from torch.utils.data import DataLoader, Dataset, TensorDataset
import snntorch as snn
import snntorch.functional as SF
from snntorch.surrogate import atan

from einops import repeat, rearrange

from scipy.signal import ellip, lfilter, butter, find_peaks
from scipy.io import loadmat
# from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import h5py 
import math
from typing import Tuple, Union, List, Optional

from tqdm import tqdm

from BRF.neurons import RAF, DTLIF, BRF
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
            input_dim: int,
            t1_t2_omegas: torch.Tensor,
            t1_t2_bs: torch.Tensor,
            threshold: torch.Tensor,
            dt: float=1/24000,
            learn_threshold: bool=False,
            num_classes: int=3,
            beta: float=0.9,
            pos_threshold: float=1.0,
            neg_threshold: float=-1.0,
            reset_mechanism: str="subtract"
    ):
        super().__init__()

        self.rafs = RAF(
            input_dim=input_dim,
            t1_t2_omegas=t1_t2_omegas,
            t1_t2_bs=t1_t2_bs,
            threshold=threshold,
            dt=dt,
            learn_omega=True,
            learn_b=True,
            learn_threshold=learn_threshold
        )
        self.dtlif = DTLIF(
            beta=beta,
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
            reset_mechanism=reset_mechanism,
        )

        self.fc1 = nn.Linear(input_dim + 1, num_classes, bias=True)
        self.lif1 = snn.Leaky(beta=0.9, threshold=0.8, learn_beta=True, learn_threshold=True, spike_grad=step_double_gaussian(), reset_mechanism="subtract")
        # self.fc2 = nn.Linear(30, num_classes)
        # self.lif2 = snn.Leaky(beta=0.9, threshold=0.8, reset_mechanism="subtract")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        bs, seq_len = x.shape
        hidden_states = self.rafs.init_hidden_state(batch_size=bs)
        dt_mem = self.dtlif.reset_mem()

        mem1 = self.lif1.reset_mem()
        # mem2 = self.lif2.reset_mem()

        raf_spk_hist, raf_u_hist = [], []

        spk_hist, mem_hist = [], []
        for i in range(seq_len):
            curr = torch.clamp(x[:, i].unsqueeze(-1), min=-1.0, max=1.0)    # Shape: (batch_size, 1)
            raf_spk, u, v, q, use_t1 = self.rafs(curr, hidden_states) # Output Shape: (batch_size, # of RAF neurons)

            dt_spk, dt_mem = self.dtlif(x[:, i].unsqueeze(-1), dt_mem) # Output Shape: (batch_size, 1)

            combined_spks = torch.cat((raf_spk, dt_spk), dim=1) # Shape: (batch_size, # of RAF neurons + 1)

            out1 = self.fc1(combined_spks)
            spk1, mem1 = self.lif1(out1, mem1)

            # out2 = self.fc2(spk1)
            # spk2, mem2 = self.lif2(out2, mem2)

            spk_hist.append(spk1)
            mem_hist.append(mem1)

            raf_spk_hist.append(raf_spk)
            raf_u_hist.append(u)

            hidden_states = raf_spk, u, v, q, use_t1

        return torch.stack(spk_hist), torch.stack(mem_hist), torch.stack(raf_spk_hist), torch.stack(raf_u_hist),

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

def generate_spike_and_label_dm(filtered_signal, event_stream, spike_time, use_gt=1, gt_window=3, spd_at_factor=4):
    # spike_train = np.zeros_like(filtered_signal)
    # spike_train[event_stream[:, 0].astype(int)] = event_stream[:, 3] - event_stream[:, 4]
    spike_train = np.zeros((filtered_signal.shape[0], 2))
    spike_train[event_stream[:, 0].astype(int), 0] = event_stream[:, 3]
    spike_train[event_stream[:, 0].astype(int), 1] = event_stream[:, 4]

    spike_time_label = np.zeros(filtered_signal.size)
    if use_gt == 1:
        spike_time += 24
        spike_time = spike_time.astype(int).flatten()
        for st in spike_time:
            spike_time_label[st - gt_window:st + gt_window + 1] = 1

    return spike_train, spike_time_label

def create_dataset_for_training(spike_train, spike_time_label, backward_window=23, forward_window=23):
    data, label = [], []

    for i in range(backward_window, spike_time_label.shape[0] - forward_window):
        data.append(spike_train[i - backward_window:i + forward_window + 1])
        label.append(spike_time_label[i])

    data = np.array(data)
    label = np.array(label)
    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).long()

    val, count = torch.unique(label, return_counts=True)
    desired_samples_per_unique = torch.min(count)

    selected_indices = []
    for v in val:
        idx = torch.where(label == v)[0]
        num_samples = min(desired_samples_per_unique, len(idx))
        selected_indices.extend(idx[:num_samples].tolist())

    data = data[selected_indices, :]
    label = label[selected_indices]

    ind = torch.randperm(label.shape[0])
    data = data[ind, :]
    label = label[ind]

    return data, label

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

def calculate_tsne_with_sign(net1, net2, train_loader):
    net1.eval()
    net2.eval()
    complete_combined_features_net1 = []
    complete_combined_features_net2 = []
    complete_label = []
    with torch.no_grad():
        for data, label in tqdm(train_loader):
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            complete_label.append(label)

            # Net1
            spk_out, mem_out, raf_spk, raf_u = net1(data) # spk_out/mem_out shape is (seq_len, bs, num_classes), raf_spk/raf_u shape is (seq_len, bs, hidden_dim)
            # raf_u_sign = torch.sign(raf_u)  # shape is (seq_len, bs, num_classes)

            # combined_feature = torch.cat((raf_spk, raf_u_sign), dim=0)  # shape is (seq_len * 2, bs, hidden_dim)

            # complete_combined_features_net1.append(combined_feature)
            complete_combined_features_net1.append(raf_spk)

            # Net2
            spk_out, mem_out, raf_spk, raf_u = net2(data) # spk_out/mem_out shape is (seq_len, bs, num_classes), raf_spk/raf_u shape is (seq_len, bs, hidden_dim)
            # raf_u_sign = torch.sign(raf_u)  # shape is (seq_len, bs, num_classes)

            # combined_feature = torch.cat((raf_spk, raf_u_sign), dim=0)  # shape is (seq_len * 2, bs, hidden_dim)

            # complete_combined_features_net2.append(combined_feature)
            complete_combined_features_net2.append(raf_spk)

    complete_combined_features_net1 = torch.cat(complete_combined_features_net1, dim=1)  # shape is (seq_len * 2, bs, hidden_dim)
    seq_len, total_num_samples, hidden_dim = complete_combined_features_net1.shape
    # complete_combined_features = complete_combined_features.permute(1, 2, 0).reshape(total_num_samples, -1) # shape is (bs, hidden_dim, seq_len * 2)
    complete_combined_features_net1 = complete_combined_features_net1.permute(1, 2, 0)
    # complete_combined_features_net1 = rearrange(complete_combined_features_net1, 'b n t -> (b n ) t')

    complete_combined_features_net2 = torch.cat(complete_combined_features_net2, dim=1)  # shape is (seq_len * 2, bs, hidden_dim)
    seq_len, total_num_samples, hidden_dim = complete_combined_features_net2.shape
    # complete_combined_features = complete_combined_features.permute(1, 2, 0).reshape(total_num_samples, -1) # shape is (bs, hidden_dim, seq_len * 2)
    complete_combined_features_net2 = complete_combined_features_net2.permute(1, 2, 0)
    # complete_combined_features_net2 = rearrange(complete_combined_features_net2, 'b n t -> (b n ) t')

    # torch.set_printoptions(profile="full")
    # print(complete_combined_features[1])
    # torch.set_printoptions(profile="default")

    complete_label = torch.cat(complete_label, dim=0)

    tsne = TSNE(n_components=2, perplexity=29, random_state=42)
    tsne_results1, tsne_results2 = [], []
    tsne_labels1, tsne_labels2 = [], []
    for i in tqdm(range(total_num_samples)):
        if torch.all(complete_combined_features_net1[i] == 0):
            continue
        else:
            tsne_results1.append(tsne.fit_transform(complete_combined_features_net1[i]))
            tsne_labels1.append(np.ones(hidden_dim) * complete_label[i].cpu().numpy())
        if torch.all(complete_combined_features_net2[i] == 0):
            continue
        else:
            tsne_results2.append(tsne.fit_transform(complete_combined_features_net2[i]))
            tsne_labels2.append(np.ones(hidden_dim) * complete_label[i].cpu().numpy())
    # tsne_results1 = tsne.fit_transform(complete_combined_features_net1.cpu().numpy())
    # tsne_results2 = tsne.fit_transform(complete_combined_features_net2.cpu().numpy())

    # tsne_results1 = torch.cat(tsne_results1, dim=0)
    # tsne_results2 = torch.cat(tsne_results2, dim=0)
    tsne_results1 = np.concatenate(tsne_results1, axis=0)
    tsne_results2 = np.concatenate(tsne_results2, axis=0)
    tsne_labels1 = np.concatenate(tsne_labels1, axis=0)
    tsne_labels2 = np.concatenate(tsne_labels2, axis=0)
    print("Checking dimension of tsne output: ", tsne_results1.shape, tsne_results2.shape, tsne_labels1.shape, tsne_labels2.shape)

    unique_labels = np.unique(complete_label.cpu().numpy())
    colors = [plt.cm.viridis(float(i)/max(unique_labels)) for i in unique_labels]

    # Plot net1
    plt.figure(figsize=(12, 16))
    for i, unique in enumerate(unique_labels):
        idx = (tsne_labels1 == unique)
        plt.scatter(tsne_results1[idx, 0], tsne_results1[idx, 1], color=colors[i], label=f"Class {int(unique)}")
    # plt.scatter(tsne_results1[:, 0], tsne_results1[:, 1], c=tsne_labels1, cmap='viridis')
    # plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Feature 1', fontsize=20)
    plt.ylabel('t-SNE Feature 2', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='best', fontsize=20)
    plt.savefig(filename[:-4] + f"spike_sorting_tsne_plot_no_loading.png", bbox_inches='tight')
    plt.savefig(filename[:-4] + f"spike_sorting_tsne_plot_no_loading.eps", format='eps', bbox_inches='tight')
    plt.close()

    # Plot net2
    plt.figure(figsize=(12, 16))
    for i, unique in enumerate(unique_labels):
        idx = (tsne_labels2 == unique)
        plt.scatter(tsne_results2[idx, 0], tsne_results2[idx, 1], color=colors[i], label=f"Class {int(unique)}")
    # plt.scatter(tsne_results2[:, 0], tsne_results2[:, 1], c=tsne_labels2, cmap='viridis')
    # plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Feature 1', fontsize=20)
    plt.ylabel('t-SNE Feature 2', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='best', fontsize=20)
    plt.savefig(filename[:-4] + f"spike_sorting_tsne_plot_loaded_model.png", bbox_inches='tight')
    plt.savefig(filename[:-4] + f"spike_sorting_tsne_plot_loaded_model.eps", format='eps', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    """
        Dataset downloaded from: https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595?file=21819066
    """
    BATCH_SIZE = 64 # original is 32
    NUM_EPOCHS= 30
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

    use_dm = "dm"
    # for dm, the threshold is: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
    # for lif, the threshold is: 0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0
    thresholds = np.array([0.2])
    print(f"Current threshold is: {thresholds}")
    train_test_split_ratio = 0.5

    # gt_noise_level = "015" # 005, 01, 015, 02
    # difficulty = "Difficult2"   # Difficult1, Difficult2, Easy1, Easy2
    filepath = "./intracortical_dataset/"
    difficulty = "Easy1"
    gt_noise_level = "005"
    filename = f"C_{difficulty}_noise{gt_noise_level}.mat"
    print(f"\tCurrent filename is: {filename}")
    MODEL_FILENAME = f"./spike_sorting_best_model.pth"

    signal, spike_class_label, spike_times, sampling_interval, sampling_rate, spike_pulse_1ms_idx_length, spike_classes, filtered_signal = load_dataset(filepath, filename)

    on_threshold = thresholds
    off_threshold = -thresholds

    event_stream = generate_event_stream_dm(filtered_signal, on_threshold, off_threshold)
    spike_train = np.zeros_like(signal)
    spike_train[event_stream[:, 0].astype(int)] = event_stream[:, 3] - event_stream[:, 4]

    all_spike_signals = {i: [] for i in spike_classes}
    all_spk_trains = {i: [] for i in spike_classes}
    for i in range(len(spike_times)):
        # all_spike_signals[spike_class_label[i]].append(filtered_signal[spike_times[i] - 23:spike_times[i] + 23 + 1])
        # all_spk_trains[spike_class_label[i]].append(spike_train[spike_times[i] - 23:spike_times[i] + 23 + 1])
        all_spike_signals[spike_class_label[i]].append(filtered_signal[spike_times[i]:spike_times[i] + 70])
        all_spk_trains[spike_class_label[i]].append(spike_train[spike_times[i]:spike_times[i] + 70])

    train_spk_train, test_spk_train, train_signal, test_signal, train_label, test_label = train_test_split(spike_classes, all_spk_trains, all_spike_signals, train_test_split_ratio)

    ######## Setup Training & Test Tensors ########
    training_spikes_tensor = torch.tensor(np.array(train_spk_train), dtype=torch.float32) # train_spk_train or filtered_spk_trains
    training_labels_tensor = torch.tensor(train_label, dtype=torch.long) - 1   # Offset by 1 to start from 0

    test_spikes_tensor = torch.tensor(np.array(test_spk_train), dtype=torch.float32)    # test_spk_train or filtered_spk_trains_test
    test_labels_tensor = torch.tensor(test_label, dtype=torch.long) - 1         # Offset by 1 to start from 0

    train_dataset = IntracorticalDataset(training_spikes_tensor, training_labels_tensor)
    test_dataset = IntracorticalDataset(test_spikes_tensor, test_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    raf_omegas1 = (torch.pi) / (torch.linspace(4, 24, steps=30, dtype=torch.float32) / 24000) # shape (24,). was 2*pi
    raf_omegas2 = (torch.pi) / (torch.linspace(4, 32, steps=30, dtype=torch.float32) / 24000) # shape (24,). was 2*pi
    raf_omegas = torch.stack((raf_omegas1, raf_omegas2), dim=-1) # stack to form (24, 2)
    raf_bs = raf_omegas / 8
    # raf_thresholds = torch.ones_like(raf_omegas) * 7.5e-5
    initial_dv = 4.1667e-5
    k_threshold1 = 1.5
    k_threshold2 = 1.95  # original is 1.9
    threshold1 = k_threshold1 * initial_dv # original value: 6e-5
    threshold2 = k_threshold2 * initial_dv # original value: 7.8e-5
    raf_thresholds = torch.tensor([threshold1, threshold2], dtype=torch.float32)
    raf_thresholds = repeat(raf_thresholds, 't -> b t', b=raf_omegas.shape[0]).clone()

    net = Model(
        input_dim=raf_omegas.shape[0],
        t1_t2_omegas=raf_omegas,
        t1_t2_bs=raf_bs,
        threshold=raf_thresholds,
        dt=1/24000,
        learn_threshold=True,
        num_classes=len(spike_classes),
        beta=0.5,
        pos_threshold=1.0,
        neg_threshold=-1.0,
        reset_mechanism="subtract"
    )
    net.to(DEVICE)
    
    # net2 = Model(
    #     input_dim=raf_omegas.shape[0],
    #     t1_t2_omegas=raf_omegas,
    #     t1_t2_bs=raf_bs,
    #     threshold=raf_thresholds,
    #     dt=1/24000,
    #     learn_threshold=True,
    #     num_classes=len(spike_classes),
    #     beta=0.5,
    #     pos_threshold=1.0,
    #     neg_threshold=-1.0,
    #     reset_mechanism="subtract"
    # )
    net2 = copy.deepcopy(net)
    net2.load_state_dict(torch.load(MODEL_FILENAME))
    net2.to(DEVICE)

    # print("net1:\n", net.rafs.t1_t2_omegas)
    # print("net2:\n", net2.rafs.t1_t2_omegas)

    # calculate_tsne(net, train_loader, load_model_path=False)
    # calculate_tsne(net, train_loader, load_model_path=True)
    calculate_tsne_with_sign(net, net2, train_loader)
