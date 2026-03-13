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
from snntorch.surrogate import atan

from einops import repeat

from scipy.signal import ellip, lfilter, butter, find_peaks
from scipy.io import loadmat

import h5py 
import math
from typing import Tuple, Union, List, Optional

from tqdm import tqdm

from BRF.neurons import RAF, TwoThresholdLIF, BRF
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
        self.dtlif = TwoThresholdLIF(
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

            hidden_states = raf_spk, u, v, q, use_t1

        return torch.stack(spk_hist), torch.stack(mem_hist)
    
class Model2(nn.Module):
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
        self.dtlif = TwoThresholdLIF(
            beta=beta,
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
            reset_mechanism=reset_mechanism,
        )

        self.fc1 = nn.Linear(input_dim, num_classes, bias=True)
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

        spk_hist, mem_hist = [], []
        for i in range(seq_len):
            curr = torch.clamp(x[:, i].unsqueeze(-1), min=-1.0, max=1.0)    # Shape: (batch_size, 1)
            raf_spk, u, v, q, use_t1 = self.rafs(curr, hidden_states) # Output Shape: (batch_size, # of RAF neurons)
            raf_spk = torch.sign(x[:, i].unsqueeze(-1)) * raf_spk

            out1 = self.fc1(raf_spk)
            spk1, mem1 = self.lif1(out1, mem1)

            # out2 = self.fc2(spk1)
            # spk2, mem2 = self.lif2(out2, mem2)

            spk_hist.append(spk1)
            mem_hist.append(mem1)

            hidden_states = raf_spk, u, v, q, use_t1

        return torch.stack(spk_hist), torch.stack(mem_hist)

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

def dt_lif_neuron(filtered_signal, time_step, threshold1=0.8, threshold2=1.0, lif_tau=5e-3):
    u_hist = []
    spk_hist = []
    u, u_rest = 0, 0
    time_lif = np.linspace(0, filtered_signal.shape[0], filtered_signal.shape[0], dtype=float)

    def leaky_integrate_neuron(U, time_step, I=0, R=5, u_rest=0, tau=5e-3):
        # tau = R*C
        U += (time_step/tau)*(-(U) + I*R) - u_rest
        return U

    for step in range(filtered_signal.shape[0]):
        u_hist.append(u)
        u = leaky_integrate_neuron(u, time_step=time_step, I=filtered_signal[step], u_rest=u_rest, tau=lif_tau)  # solve next step of U
        if u >= threshold1:
            u_rest = threshold1
            spk_hist.append(float(1))
        elif u<= -threshold1:
            u_rest = -threshold1
            spk_hist.append(float(-1))
        else:
            u_rest = 0
            spk_hist.append(float(0))

    return np.array(spk_hist), time_lif, u_hist

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

def visualise_test_results(net, data, label, predictions, raf_spk, raf_u, spk_filt_hist, batch_no):
    for i in tqdm(range(data.shape[0])):
        curr_data = data[i]
        curr_label = label[i]

        fig, ax = plt.subplots(1 + raf_spk.shape[-1], 1, figsize=(12, 6*raf_spk.shape[-1]))
        ax[0].stem(curr_data[:, 0], linefmt ='blue', markerfmt=" ", label="Positive Events")
        ax[0].stem(curr_data[:, 1], linefmt ='red', markerfmt=" ", label="Negative Events")
        ax[0].minorticks_on()
        ax[0].legend()
        
        # pos_loc = np.where(curr_data > 0)[0]
        # neg_loc = np.where(curr_data < 0)[0]
        # ax[0].eventplot(pos_loc, lineoffsets=0, linelengths=0.5, colors="blue")
        # ax[0].eventplot(neg_loc, lineoffsets=0, linelengths=0.5, colors="green")
        for j in range(raf_spk.shape[-1]):
            ax[j+1].stem(raf_spk[:, i, j].detach().cpu().numpy() * 2e-4, linefmt ='blue', markerfmt=" ", label="RAF Spk")
            ax[j+1].stem(spk_filt_hist[:, i, j].detach().cpu().numpy() * 1e-4, linefmt ='black', markerfmt=" ", label="LIF Spk")
            ax[j+1].plot(raf_u[:, i, j].detach().cpu().numpy(), color="red")
            ax[j+1].minorticks_on()
            ax[j+1].set_title(f"RAF Neuron {j}'s Omega: {net.raf.omegas[j].detach().cpu().item():.4f}")
            ax[j+1].legend()
            ax[j+1].grid(visible=True, which="major", axis="both", alpha=0.4, color="gray")
            ax[j+1].grid(visible=True, which="minor", axis="both", alpha=0.4, color="lightgray")

        # handles = []
        # labels = []
        # for a in ax.flat:
        #     h, l = a.get_legend_handles_labels()
        #     handles.extend(h)
        #     labels.extend(l)

        # fig.legend(handles, labels, loc="lower left")
        # plt.grid(visible=True, which="major", axis="both", alpha=0.5, color="gray")
        # plt.grid(visible=True, which="minor", axis="both", alpha=0.4, color="lightgray")
        plt.savefig(f"{PREDICTION_OUTPUT_PATH}batch_no_{batch_no}_signal_{i}_predicted_{predictions[i].detach().item()}_actual_{label[i].item()}.jpg")
        plt.close()

def train(
    net,
    train_loader,
    optimiser,
    loss_fn,
    acc_mode="count",
    scheduler=None,
):
    best_acc = 0.0
    best_loss = float('inf')
    for epoch in tqdm(range(NUM_EPOCHS)):
        # print(f"epoch {epoch}:\n\
        #       LIF filter threshold: {net.lif_filt.threshold}\n\
        #         LIF filter beta: {net.lif_filt.beta}\n\
        #         RAF omegas: {net.raf.omegas}\n\
        #         RAF bs: {net.raf.bs}\n\
        #         RAF thresholds: {net.raf.threshold}")
        net.train()
        curr_loss = 0.0
        for data, label in train_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            spk_out, mem_out = net(data)

            loss = loss_fn(spk_out, label)
            curr_loss += loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        if acc_mode == "temporal":
            acc_fn = SF.acc.accuracy_temporal
            complete_spikes, complete_label = [], []
        correct_samples, total_samples = 0, 0
        with torch.no_grad():
            for data, label in train_loader:
                data = data.to(DEVICE)
                label = label.to(DEVICE)

                spk_out, mem_out = net(data)

                if acc_mode == "count":
                    idx = spk_out.sum(0).argmax(1)
                    correct_samples += (idx == label).sum().item()
                    total_samples += label.shape[0]
                elif acc_mode == "temporal":
                    complete_spikes.append(spk_out)
                    complete_label.append(label)

        if acc_mode == "count":
            train_acc = correct_samples / total_samples
        elif acc_mode == "temporal":
            complete_spikes = torch.cat(complete_spikes, dim=1)
            complete_label = torch.cat(complete_label, dim=0)
            train_acc = acc_fn(complete_spikes, complete_label)

        tqdm.write(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Accuracy: {train_acc:.4f}, Loss: {curr_loss:.4f}")
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
    acc_mode="count",
    visualise: bool=False,
    final_test: bool=False,
):
    net.load_state_dict(torch.load(MODEL_FILENAME, weights_only=True))
    net.to(DEVICE)
    net.eval()

    if acc_mode == "temporal":
        acc_fn = SF.acc.accuracy_temporal
        complete_spikes, complete_label = [], []
    correct_samples, total_samples = 0, 0
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            if visualise:
                spk_out, mem_out = net(data)

                # correct, total, idx = calc_population_code(raf_spk, label, num_classes=2, pop_size=raf_spk.shape[-1], return_predictions=True)
                # correct_samples += correct
                # total_samples += total

                if acc_mode == "count":
                    idx = spk_out.sum(0).argmax(1)
                    correct_samples += (idx == label).sum().item()
                    total_samples += label.shape[0]

                    # visualise_test_results(net, data, label, idx, raf_spk, raf_u, lif_spk, i)
                elif acc_mode == "temporal":
                    complete_spikes.append(spk_out)
                    complete_label.append(label)

                    # visualise_test_results(net, data, label, torch.zeros(data.shape[0], dtype=torch.long), raf_spk, raf_u, lif_spk, i)
            else:
                spk_out, mem_out = net(data)

                # correct, total = calc_population_code(raf_spk, label, num_classes=2, pop_size=raf_spk.shape[-1])
                # correct_samples += correct
                # total_samples += total

                if acc_mode == "count":
                    idx = spk_out.sum(0).argmax(1)
                    correct_samples += (idx == label).sum().item()
                    total_samples += label.shape[0]
                elif acc_mode == "temporal":
                    complete_spikes.append(spk_out)
                    complete_label.append(label)

    if acc_mode == "count":
        test_acc = correct_samples / total_samples
    elif acc_mode == "temporal":
        complete_spikes = torch.cat(complete_spikes, dim=1)
        complete_label = torch.cat(complete_label, dim=0)
        test_acc = acc_fn(complete_spikes, complete_label)
    if final_test:
        # tqdm.write(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"\t\tFinal Test Accuracy: {test_acc:.4f}")

def clean_images(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Cleaned up folder: {folder_path}")
    os.makedirs(folder_path, exist_ok=True)

if __name__ == "__main__":
    """
        Dataset downloaded from: https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595?file=21819066
    """
    BATCH_SIZE = 64
    NUM_EPOCHS= 50 # 30 was the original setting. 40 Gave pretty good result. 50 is the best so far
    MODEL_FILENAME = f"./spike_sorting_best_model.pth"
    TEST_SAMPLES_OUTPUT_PATH = "./test_samples_output/"
    # clean_images(TEST_SAMPLES_OUTPUT_PATH)
    TRAIN_SAMPLES_OUTPUT_PATH = "./train_samples_output/"
    # clean_images(TRAIN_SAMPLES_OUTPUT_PATH)
    PREDICTION_OUTPUT_PATH = "./prediction_plots/"
    # clean_images(PREDICTION_OUTPUT_PATH)
    TRAINING_PRED_OUTPUT_PATH = "./training_prediction_plots/"
    # clean_images(TRAINING_PRED_OUTPUT_PATH)

    SEED = 5673 # 1337, 5673, 87353
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    gt_noise_level = "02" # 005, 01, 015, 02
    difficulty = "Difficult2"   # Difficult1, Difficult2, Easy1, Easy2
    filepath = "./intracortical_dataset/"
    filename = f"C_{difficulty}_noise{gt_noise_level}.mat"

    dm_thresholds = np.array([0.2])
    select_top_x_t1_t2 = 10  # At least 9 to make meaningful accuracy
    train_test_split_ratio = 0.5

    signal, spike_class_label, spike_times, sampling_interval, sampling_rate, spike_pulse_1ms_idx_length, spike_classes, filtered_signal = load_dataset(filepath, filename)
    on_threshold = dm_thresholds
    off_threshold = -dm_thresholds

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

    raf_omegas1 = (torch.pi) / (torch.linspace(4, 24, steps=30, dtype=torch.float32) / 24000) # original shape (30,). was 2*pi
    raf_omegas2 = (torch.pi) / (torch.linspace(4, 32, steps=30, dtype=torch.float32) / 24000) # original shape (30,). was 2*pi
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

    # net = Model2(
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

    net.to(DEVICE)
    test_net = copy.deepcopy(net)
    loss_fn = SF.ce_count_loss()
    # loss_fn = SF.loss.ce_temporal_loss(inverse="negate") # negate or reciprocal

    # optimiser = torch.optim.AdamW(
    #     [
    #         {'params': net.rafs.t1_t2_omegas, 'lr': 1e-3},
    #         {'params': net.rafs.t1_t2_bs, 'lr': 1e-3},
    #         {'params': net.rafs.threshold, 'lr': 1e-5},
    #         {'params': net.dtlif.beta, 'lr': 1e-2},
    #         {'params': net.dtlif.pos_threshold, 'lr': 1e-1},
    #         {'params': net.dtlif.neg_threshold, 'lr': 1e-1},
    #         {'params': net.fc1.parameters()},
    #         {'params': net.lif1.parameters()},
    #     ], lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01,
    # ) # This setting seems to work the best

    # optimiser = torch.optim.AdamW(
    #     [
    #         {'params': net.rafs.t1_t2_omegas, 'lr': 1e-4},
    #         {'params': net.rafs.t1_t2_bs, 'lr': 1e-4},
    #         {'params': net.rafs.threshold, 'lr': 1e-6},
    #         {'params': net.dtlif.beta, 'lr': 1e-2},
    #         {'params': net.dtlif.pos_threshold, 'lr': 1e-1},
    #         {'params': net.dtlif.neg_threshold, 'lr': 1e-1},
    #         {'params': net.fc1.parameters()},
    #         {'params': net.lif1.parameters()},
    #     ], lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01,
    # ) # This setting seems to work the best for Model()

    optimiser = torch.optim.AdamW(
        [
            {'params': net.rafs.t1_t2_omegas, 'lr': 1e-5},
            {'params': net.rafs.t1_t2_bs, 'lr': 1e-5},
            {'params': net.rafs.threshold, 'lr': 1e-7},
            {'params': net.dtlif.beta, 'lr': 1e-2},
            {'params': net.dtlif.pos_threshold, 'lr': 1e-1},
            {'params': net.dtlif.neg_threshold, 'lr': 1e-1},
            {'params': net.fc1.parameters()},
            {'params': net.lif1.parameters()},
        ], lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01,
    ) # This setting seems to work the best for Model2()

    # optimiser = torch.optim.RMSprop(net.parameters(), lr=1e-3, alpha=0.99, eps=1e-8)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS, eta_min=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.1)
    scheduler = None

    train(net, train_loader, optimiser, loss_fn, acc_mode="count", scheduler=scheduler) # acc_mode="temporal" or "count"
    test(test_net, test_loader, acc_mode="count", final_test=True, visualise=True)
    
