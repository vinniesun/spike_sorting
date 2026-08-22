import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import snntorch.functional as SF
import random
import os
from datetime import datetime
from tqdm import tqdm
import copy

from utils import (
    IntracorticalDataset,
    load_dataset_intracortical,
    dv_to_lif_spike_gen,
    train_test_split_spike_sorting,
    generate_event_stream_dm,
)
from model import SpikingLSTMSpikeSorter

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
        net.train()
        curr_loss = 0.0
        for data, label in train_loader:
            data = data.to(DEVICE)  # shape (batch_size, seq_len)
            label = label.to(DEVICE) # shape (batch_size)
            # print(f"data shape: {data.shape}, label shape: {label.shape}")

            spk_out = net(data)

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

                spk_out = net(data)

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
                spk_out = net(data)

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
                spk_out = net(data)

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

if __name__ == "__main__":
    """
        Dataset downloaded from: https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595?file=21819066
    """
    BATCH_SIZE = 256 # 128 or 64
    NUM_EPOCHS= 60 # Need to increase no of epochs?

    TRAINING_LOG_PATH = "./spike_sorting_training_log"
    if not os.path.exists(TRAINING_LOG_PATH):
        os.makedirs(TRAINING_LOG_PATH)
    TRAINING_LOG_NAME = f"{TRAINING_LOG_PATH}/training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    SEED = 1337 # 1337, 5673, 1234
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

    encoder = "dm" # "dm" or "dv"
    reset_mechanism = "subtract" # "none", "subtract", "zero"
    train_test_split_ratio = 0.7 # 70% training, 30% testing
    encoder_threshold = 0.2
    lif_tau = 1 * (1/24000)

    for difficulty in ["Difficult1", "Difficult2", "Easy1", "Easy2"]:
        for noise_level in ["005", "01", "015", "02"]:
            filename = f"C_{difficulty}_noise{noise_level}.mat"

            MODEL_FILENAME = f"./intracortical_weights/{filename[:-4]}_spike_sorting_best_model.pth"
            
            with open(TRAINING_LOG_NAME, "a") as f:
                f.write(f"Current Setting: filename: {filename}\n\n")

            signal, spike_class_label, spike_times, sampling_interval, \
            sampling_rate, spike_pulse_1ms_idx_length, spike_classes, \
            filtered_signal = load_dataset_intracortical(filepath, filename)

            if encoder == "dm":
                on_threshold = encoder_threshold
                off_threshold = -encoder_threshold
                event_stream = generate_event_stream_dm(filtered_signal, on_threshold, off_threshold)
                spike_train = np.zeros_like(signal)
                spike_train[event_stream[:, 0].astype(int)] = event_stream[:, 1] - event_stream[:, 2]
            elif encoder == "dv":
                dv_u_hist, spike_train, dv_time_lif = dv_to_lif_spike_gen(
                    signal=filtered_signal,
                    lif_threshold=encoder_threshold,
                    sampling_interval=sampling_interval,
                    lif_tau=lif_tau,
                    reset_mechanism=reset_mechanism
                )
            else:
                raise ValueError("Invalid encoder type. Choose either 'dm' or 'dv'.")

            all_spike_signals = {i: [] for i in spike_classes}
            all_spk_trains = {i: [] for i in spike_classes}
            for i in range(len(spike_times)):
                all_spike_signals[spike_class_label[i]].append(filtered_signal[spike_times[i] - 23:spike_times[i] + 24])
                all_spk_trains[spike_class_label[i]].append(spike_train[spike_times[i] - 23:spike_times[i] + 24])

                if i == 100:
                    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                    ax[0].plot(filtered_signal[spike_times[i] - 23:spike_times[i] + 24])
                    ax[1].stem(spike_train[spike_times[i] - 23:spike_times[i] + 24])

                    plt.tight_layout()
                    plt.savefig(f"verify_spike_det_labeling/spike_{i}_label_{spike_class_label[i]}_signal_and_spike_train.jpg", dpi=300)
                    plt.close()

            train_spk_train, test_spk_train, \
            train_signal, test_signal, \
            train_label, test_label = train_test_split_spike_sorting(
                spike_classes, 
                all_spk_trains, 
                all_spike_signals, 
                train_test_split_ratio
            )

            ######## Setup Training & Test Tensors ########
            training_spikes_tensor = torch.tensor(np.array(train_spk_train), dtype=torch.float32) # train_spk_train or filtered_spk_trains
            training_labels_tensor = torch.tensor(train_label, dtype=torch.long) - 1   # Offset by 1 to start from 0
            
            test_spikes_tensor = torch.tensor(np.array(test_spk_train), dtype=torch.float32)    # test_spk_train or filtered_spk_trains_test
            test_labels_tensor = torch.tensor(test_label, dtype=torch.long) - 1         # Offset by 1 to start from 0

            train_dataset = IntracorticalDataset(training_spikes_tensor, training_labels_tensor)
            test_dataset = IntracorticalDataset(test_spikes_tensor, test_labels_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

            net = SpikingLSTMSpikeSorter(
                input_dim=1,
                hidden_size=128,
                num_classes=len(spike_classes),
            )
            net.to(DEVICE)
            test_net = copy.deepcopy(net)

            optimiser = torch.optim.AdamW(net.parameters(), lr=2e-3, betas=(0.9, 0.999), weight_decay=0.1)
            loss_fn = SF.ce_count_loss()
            scheduler = None

            train(net, train_loader, optimiser, loss_fn, acc_mode="count", scheduler=scheduler) # acc_mode="temporal" or "count"
            test(test_net, test_loader, acc_mode="count", final_test=True, visualise=True)
