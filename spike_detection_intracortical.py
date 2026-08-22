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
    BATCH_SIZE = 64 # 128 or 64
    NUM_EPOCHS= 20 # 50 is the best so far

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
    reset_mechanism = "none" # "none", "subtract", "zero"
    train_test_split_ratio = 0.8 # 80% training, 20% testing
    label_window_size = 3
    lif_threshold = 0.3
    lif_tau = 8 * (1/24000)

    for difficulty in ["Difficult1", "Difficult2", "Easy1", "Easy2"]:
        for noise_level in ["005", "01", "015", "02"]:
            filename = f"C_{difficulty}_noise{noise_level}.mat"

            rng = np.random.default_rng(seed=SEED)

            signal, spike_class_label, spike_times, sampling_interval, \
            sampling_rate, spike_pulse_1ms_idx_length, spike_classes, \
            filtered_signal = load_dataset_intracortical(filepath, filename)

            dv_u_hist, dv_spk_hist, dv_time_lif = dv_to_lif_spike_gen(
                signal=filtered_signal,
                lif_threshold=lif_threshold,
                sampling_interval=sampling_interval,
                lif_tau=lif_tau,
                reset_mechanism=reset_mechanism
            )

            train_spike_train, train_spike_labels, \
            test_spike_train, test_spike_labels, \
            train_spike_times, test_spike_times = train_test_split_spike_detection(
                spike_train=dv_spk_hist,
                spike_times=spike_times,
                split_ratio=train_test_split_ratio,
                label_window=label_window_size
            )

            ############ Verify spike window and gt spike time and spike signal matches
            # fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

            # ax[0].plot(filtered_signal[train_spike_times[0]-48:train_spike_times[0]+48], color="blue", label=r"V(t)")
            # ax[0].plot(train_spike_labels[train_spike_times[0]-48:train_spike_times[0]+48], color="red", label=r"\hat{y}(t)")
            # ax[0].legend(loc="lower left")

            # ax[1].plot(filtered_signal[train_spike_times[3]-48:train_spike_times[3]+48], color="blue", label=r"V(t)")
            # ax[1].plot(train_spike_labels[train_spike_times[3]-48:train_spike_times[3]+48], color="red", label=r"\hat{y}(t)")
            # ax[1].legend(loc="lower left")

            # ax[2].plot(filtered_signal[train_spike_times[8]-48:train_spike_times[8]+48], color="blue", label=r"V(t)")
            # ax[2].plot(train_spike_labels[train_spike_times[8]-48:train_spike_times[8]+48], color="red", label=r"\hat{y}(t)")
            # ax[2].legend(loc="lower left")

            # ax[3].plot(filtered_signal[train_spike_times[23]-48:train_spike_times[23]+48], color="blue", label=r"V(t)")
            # ax[3].plot(train_spike_labels[train_spike_times[23]-48:train_spike_times[23]+48], color="red", label=r"\hat{y}(t)")
            # ax[3].legend(loc="lower left")

            # plt.tight_layout()
            # plt.savefig(f"./verify_spike_det_labeling/{difficulty}_noise{noise_level}.jpg", dpi=300)
            # plt.close()
            ############ End of verification

            train_samples, train_labels = create_training_dataset_spike_detection(
                spike_train=train_spike_train,
                spike_labels=train_spike_labels,
                spike_times=train_spike_times
            ) # list of torch tensors

            train_samples = torch.stack(train_samples, dim=0) # shape (num_samples, seq_len)
            train_labels = torch.stack(train_labels, dim=0) # shape (num_samples, seq_len)
            train_dataset = IntracorticalDataset(train_samples, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

            net = SpikeDetector(
                input_dim=1,
                hidden_size=32,
                output_dim=2
            )
            net.to(DEVICE)

            optimiser = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
            loss_fn = torch.nn.CrossEntropyLoss()

            best_loss = float("inf")
            loss_hist = []
            for epoch in tqdm(range(NUM_EPOCHS)):
                net.train()
                local_loss = 0.0
                for data, target in train_loader:
                    data = data.unsqueeze(-1).to(DEVICE) # shape (batch_size, seq_len, 1)
                    target = target.to(DEVICE) # shape (batch_size, seq_len)

                    spk_hist, mem_hist = net(data) # (seq_len, 1, 2)

                    spk_hist = spk_hist.permute(1, 2, 0)
                    loss_val = loss_fn(spk_hist, target)

                    optimiser.zero_grad()
                    loss_val.backward()
                    optimiser.step()

                    local_loss += loss_val.item()

                if local_loss < best_loss:
                    best_loss = local_loss
                    torch.save(net.state_dict(), f"./spike_detection_weights/best_model_{difficulty}_noise{noise_level}.pth")
                loss_hist.append(local_loss)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot(loss_hist, color="blue", label="Training Loss")
            plt.tight_layout()
            plt.savefig(f"./loss_{difficulty}_noise{noise_level}.jpg", dpi=300)
            plt.close()

            net.load_state_dict(torch.load(f"./spike_detection_weights/best_model_{difficulty}_noise{noise_level}.pth", weights_only=True))
            net.eval()

            with torch.no_grad():
                test_data = torch.tensor(test_spike_train, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE) # shape (1, seq_len, 1)
                test_target = torch.tensor(test_spike_labels, dtype=torch.long).unsqueeze(0).to(DEVICE) # shape (1, seq_len)

                spk_hist, mem_hist = net(test_data)

            class_index = torch.argmax(spk_hist, dim=2) # shape (seq_len, batch_size)
            print(f"final output shape: {class_index.shape}")

            # calc accuracy

