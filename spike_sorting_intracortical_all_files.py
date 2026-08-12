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

from datetime import datetime

from utils import (
    IntracorticalDataset,
    get_threshold_reset_counts,
    generate_event_stream_dm,
    generate_event_stream_lif,
    leaky_integrate_neuron,
    lif_neuron,
    load_dataset_intracortical,
    train_test_split
)
from model import (
    DBRFDTLIFModel,
)

raf_interval_to_b_mapping = {
    4: 2,
    5: 2,
    6: 2,
    7: 3,
    8: 3,
    9: 3,
    10: 4,
    11: 4,
    12: 4,
    13: 5,
    14: 5,
    15: 5,
    16: 6,
    17: 6,
    18: 6,
    19: 7,
    20: 7,
    21: 7,
    22: 8,
    23: 8,
    24: 8,
    25: 8,
}

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
        plt.savefig(f"./prediction_plots/batch_no_{batch_no}_signal_{i}_predicted_{predictions[i].detach().item()}_actual_{label[i].item()}.jpg")
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
            data = data.to(DEVICE)  # shape (batch_size, seq_len)
            label = label.to(DEVICE) # shape (batch_size)
            # print(f"data shape: {data.shape}, label shape: {label.shape}")

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

        with open(TRAINING_LOG_NAME, "a") as f:
            f.write(f"\tEpoch {epoch+1}/{NUM_EPOCHS}, training Acc: {train_acc}, Loss: {curr_loss:.4f}\n")
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
        with open(TRAINING_LOG_NAME, "a") as f:
            f.write(f"\t\tFinal Test Accuracy: {test_acc:.4f}\n")

def clean_images(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Cleaned up folder: {folder_path}")
    os.makedirs(folder_path, exist_ok=True)

if __name__ == "__main__":
    """
        Dataset downloaded from: https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595?file=21819066
    """
    BATCH_SIZE = 64 # 128 or 64
    NUM_EPOCHS= 50 # 50 is the best so far
    MODEL_FILENAME = f"./intracortical_weights/spike_sorting_best_model.pth"
    # clean_images(TRAINING_PRED_OUTPUT_PATH)
    TRAINING_LOG_PATH = "./spike_sorting_training_log"
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

    with open(TRAINING_LOG_NAME, "a") as f:
        f.write(f"Seed Number: {SEED}\n\n")

    filepath = "./intracortical_dataset/"
    # for lif, the threshold is: 0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0
    dm_thresholds = np.array([0.2])
    thresholds = np.array([0.8])
    train_test_split_ratio = 0.5

    complete_train_data, complete_train_labels, complete_test_data, complete_test_labels = [], [], [], []
    for difficulty in ["Difficult1", "Difficult2", "Easy1", "Easy2"]:
        for noise_level in ["005", "01", "015", "02"]:
            filename = f"C_{difficulty}_noise{noise_level}.mat"
    
            with open(TRAINING_LOG_NAME, "a") as f:
                f.write(f"Current Setting: thresholds{dm_thresholds}, filename: {filename}\n\n")

            signal, spike_class_label, spike_times, sampling_interval, sampling_rate, spike_pulse_1ms_idx_length, spike_classes, filtered_signal = load_dataset_intracortical(filepath, filename)

            on_threshold = dm_thresholds
            off_threshold = -dm_thresholds

            # Use DM to generate event stream and spike train
            event_stream = generate_event_stream_dm(filtered_signal, on_threshold, off_threshold)
            spike_train = np.zeros_like(signal)
            spike_train[event_stream[:, 0].astype(int)] = event_stream[:, 1] - event_stream[:, 2]

            # Use LIF to generate spike train
            # spike_train = generate_event_stream_lif(filtered_signal, sampling_interval, uth=thresholds, lif_tau=sampling_interval, if_reconstruct=False)
            
            all_spike_signals = {i: [] for i in spike_classes}
            all_spk_trains = {i: [] for i in spike_classes}
            
            for i in range(len(spike_times)):
                # all_spike_signals[spike_class_label[i]].append(filtered_signal[spike_times[i] - 23:spike_times[i] + 23 + 1])
                # all_spk_trains[spike_class_label[i]].append(spike_train[spike_times[i] - 23:spike_times[i] + 23 + 1])

                # if i == 1:
                #     print(spike_train[spike_times[i] - 23:spike_times[i] + 23])
                # with open(f"intracortical_spike_train_examples/{spike_class_label[i]}_spike_train_examples.txt", "a") as f:
                #     f.write(str(spike_train[spike_times[i] - 23:spike_times[i] + 23].tolist()) + "\n")

                all_spike_signals[spike_class_label[i]].append(filtered_signal[spike_times[i] - 23:spike_times[i] + 23])
                all_spk_trains[spike_class_label[i]].append(spike_train[spike_times[i] - 23:spike_times[i] + 23])

            # print("Done collecting all of the spike trains")
            train_spk_train, test_spk_train, train_signal, test_signal, train_label, test_label = train_test_split(spike_classes, all_spk_trains, all_spike_signals, train_test_split_ratio)

            ######## Setup Training & Test Tensors ########
            training_spikes_tensor = torch.tensor(np.array(train_spk_train), dtype=torch.float32) # train_spk_train or filtered_spk_trains
            training_labels_tensor = torch.tensor(train_label, dtype=torch.long) - 1   # Offset by 1 to start from 0

            test_spikes_tensor = torch.tensor(np.array(test_spk_train), dtype=torch.float32)    # test_spk_train or filtered_spk_trains_test
            test_labels_tensor = torch.tensor(test_label, dtype=torch.long) - 1         # Offset by 1 to start from 0

            complete_train_data.append(training_spikes_tensor)
            complete_train_labels.append(training_labels_tensor)
            complete_test_data.append(test_spikes_tensor)
            complete_test_labels.append(test_labels_tensor)

    complete_train_data = torch.cat(complete_train_data, dim=0)
    complete_train_labels = torch.cat(complete_train_labels, dim=0)
    complete_test_data = torch.cat(complete_test_data, dim=0)
    complete_test_labels = torch.cat(complete_test_labels, dim=0)

    train_dataset = IntracorticalDataset(complete_train_data, complete_train_labels)
    test_dataset = IntracorticalDataset(complete_test_data, complete_test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # raf_omegas1 = (torch.pi) / (torch.linspace(4, 24, steps=30, dtype=torch.float32) / 24000) # original shape (30,). was 2*pi
    # raf_omegas2 = (torch.pi) / (torch.linspace(4, 32, steps=30, dtype=torch.float32) / 24000) # original shape (30,). was 2*pi
    # raf_omegas = torch.stack((raf_omegas1, raf_omegas2), dim=-1) # stack to form (24, 2)
    # raf_bs = raf_omegas / 8
    # # raf_thresholds = torch.ones_like(raf_omegas) * 7.5e-5
    # initial_dv = 4.1667e-5
    # k_threshold1 = 2.5
    # k_threshold2 = 3.0  # original is 1.9
    # threshold1 = k_threshold1 * initial_dv # original value: 6e-5
    # threshold2 = k_threshold2 * initial_dv # original value: 7.8e-5
    # raf_thresholds = torch.tensor([threshold1, threshold2], dtype=torch.float32)
    # raf_thresholds = repeat(raf_thresholds, 't -> b t', b=raf_omegas.shape[0]).clone()

    interval1 = torch.arange(start=4, end=20, step=2, dtype=torch.float32)
    interval2 = torch.arange(start=4, end=25, step=2, dtype=torch.float32)
    raf_omega_interval = torch.cartesian_prod(interval1, interval2)
    raf_omegas = torch.pi / (raf_omega_interval / 24000)

    raf_bs = torch.tensor([
        [
            raf_omegas[i, 0] / raf_interval_to_b_mapping[raf_omega_interval[i, 0].item()], 
            raf_omegas[i, 1] / raf_interval_to_b_mapping[raf_omega_interval[i, 1].item()]
        ] for i in range(raf_omegas.shape[0])
    ])

    initial_dv = 4.1667e-5
    
    k_threshold1 = 2
    k_threshold2 = 3.0  # original is 1.9
    threshold1 = k_threshold1 * initial_dv # original value: 6e-5
    threshold2 = k_threshold2 * initial_dv # original value: 7.8e-5
    raf_thresholds = torch.tensor([threshold1, threshold2], dtype=torch.float32)
    raf_thresholds = repeat(raf_thresholds, 't -> b t', b=raf_omegas.shape[0]).clone()

    raf_q_coeff = torch.tensor([1e-1, 1e-3], dtype=torch.float32)
    raf_q_coeff = repeat(raf_q_coeff, 't -> b t', b=raf_omegas.shape[0]).clone()

    net = DBRFDTLIFModel(
        dbrf_input_dim=raf_omegas.shape[0],
        dtlif_input_dim=1,
        dual_omegas=raf_omegas,
        dual_bs=raf_bs,
        dual_threshold=raf_thresholds,
        dual_q_coeff=raf_q_coeff,
        dt=1/24000,
        learn_dual_threshold=True,
        num_classes=len(spike_classes),
        beta=0.5,
        pos_threshold=3.0,
        neg_threshold=-3.0,
        learn_beta=False,
        learn_dtlif_threshold=True,
        reset_mechanism="subtract"
    )

    net.to(DEVICE)
    test_net = copy.deepcopy(net)
    loss_fn = SF.ce_count_loss()
    # loss_fn = SF.loss.ce_temporal_loss(inverse="negate") # negate or reciprocal

    optimiser = torch.optim.AdamW(
        [
            {'params': net.rafs.dual_omegas, 'lr': 0.001},
            {'params': net.rafs.dual_bs, 'lr': 0.001},
            {'params': net.rafs.dual_threshold, 'lr': 1e-6},
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
    print(f"Before training, RAF's threshold: {net.rafs.dual_threshold}")

    train(net, train_loader, optimiser, loss_fn, acc_mode="count", scheduler=scheduler) # acc_mode="temporal" or "count"
    test(test_net, test_loader, acc_mode="count", final_test=True, visualise=True)

    print(f"After training, RAF's threshold: {net.rafs.dual_threshold}")
