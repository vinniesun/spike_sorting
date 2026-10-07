import argparse
import copy
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# Hungarian algorithm for optimal assignment
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import KMeansHead, LSTMAutoencoder
from utils import (
    IntracorticalDataset,
    dv_to_lif_spike_gen,
    generate_event_stream_dm,
    load_dataset_intracortical,
    train_test_split_spike_sorting,
)


def train_ae(
    net,
    train_loader,
    optimiser,
    loss_fn,
    scheduler=None,
):
    best_loss = float("inf")
    for epoch in tqdm(range(NUM_EPOCHS)):
        net.train()
        curr_loss = 0.0
        for data, label in train_loader:
            data = data.to(DEVICE)  # shape (batch_size, seq_len)
            label = label.to(DEVICE)  # shape (batch_size)
            # print(f"data shape: {data.shape}, label shape: {label.shape}")

            latent_representation, reconstructed, _ = net(data)

            loss = loss_fn(reconstructed, data)
            curr_loss += loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        with open(TRAINING_LOG_NAME, "a") as f:
            f.write(f"\tEpoch {epoch + 1}/{NUM_EPOCHS}, Loss: {curr_loss:.4f}\n")
        # tqdm.write(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Accuracy: {train_acc:.4f}, Loss: {curr_loss:.4f}")

        if curr_loss < best_loss:
            torch.save(net.state_dict(), MODEL_FILENAME_LOSS)
            best_loss = curr_loss
        # torch.save(net.state_dict(), MODEL_FILENAME)

        # test(test_net, test_loader, acc_fn)

        if scheduler is not None:
            scheduler.step()

def train_kmeans(
    ae,
    net,
    train_loader,
    optimiser,
    loss_fn,
    scheduler=None,
):
    best_recon_loss = float("inf")
    best_cluster_loss = float("inf")
    cluster_scale = 0.1
    for epoch in tqdm(range(NUM_EPOCHS)):
        net.train()
        curr_loss = 0.0
        total_recon_loss = 0.0
        total_cluster_loss = 0.0
        clustering_loss_weight = cluster_scale * min((epoch + 1) / NUM_EPOCHS, 1.0)
        for data, label in train_loader:
            data = data.to(DEVICE)  # shape (batch_size, seq_len)
            label = label.to(DEVICE)  # shape (batch_size)
            # print(f"data shape: {data.shape}, label shape: {label.shape}")

            with torch.no_grad():
                latent_representation, reconstructed, _ = ae(data)
            
            clustering_loss, _ = net(latent_representation)

            reconstruction_loss = loss_fn(reconstructed, data)
            loss = clustering_loss_weight * clustering_loss

            curr_loss += loss.item()
            total_recon_loss += reconstruction_loss.item()
            total_cluster_loss += clustering_loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        with open(TRAINING_LOG_NAME, "a") as f:
            f.write(f"\tEpoch {epoch + 1}/{NUM_EPOCHS}, Total Loss: {curr_loss:.4f}, Reconstruction Loss: {total_recon_loss:.4f}, Clustering Loss: {total_cluster_loss:.4f}\n")
        # tqdm.write(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Accuracy: {train_acc:.4f}, Loss: {curr_loss:.4f}")

        if total_cluster_loss < best_cluster_loss:
            torch.save(net.state_dict(), MODEL_FILENAME_ACC)
            best_cluster_loss = total_cluster_loss
        if total_recon_loss < best_recon_loss:
            torch.save(ae.state_dict(), MODEL_FILENAME_LOSS)
            best_recon_loss = total_recon_loss

        if scheduler is not None:
            scheduler.step()

def test_kmeans(
    test_ae_net,
    net,
    test_loader,
    loss_fn,
    n_clusters,
    n_classes,
    final_test: bool = False,
):
    test_ae_net.load_state_dict(torch.load(MODEL_FILENAME_LOSS, weights_only=True))
    test_ae_net.to(DEVICE)
    test_ae_net.eval()

    net.load_state_dict(torch.load(MODEL_FILENAME_ACC, weights_only=True))
    net.to(DEVICE)
    net.eval()

    predictions, labels = [], []

    total_loss = 0.0
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            latent_representation, reconstructed = test_ae_net(data)
            _, assignments = net(latent_representation)

            predictions.append(assignments.cpu().numpy())
            labels.append(label.cpu().numpy())

            loss = loss_fn(reconstructed, data)
            total_loss += loss.item()

    counts = np.zeros((n_clusters, n_classes), dtype=np.int64)
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    np.add.at(counts, (predictions, labels), 1)

    rows, cols = linear_sum_assignment(-counts)
    final_acc = counts[rows, cols].sum() / np.concatenate(labels).shape[0]

    if final_test:
        # tqdm.write(f"Final Test Accuracy: {test_acc:.4f}")
        with open(TRAINING_LOG_NAME, "a") as f:
            f.write(f"\t\tFinal Test Reconstruction Loss: {total_loss:.4f}\n")
            f.write(f"\t\tFinal Test Accuracy: {final_acc:.4f}\n")

def test_ae(
    net,
    test_loader,
    loss_fn,
    final_test: bool = False,
):
    net.load_state_dict(torch.load(MODEL_FILENAME_LOSS, weights_only=True))
    net.to(DEVICE)
    net.eval()

    total_loss = 0.0
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            latent_representation, reconstructed = net(data)

            loss = loss_fn(reconstructed, data)
            total_loss += loss.item()

    if final_test:
        # tqdm.write(f"Final Test Accuracy: {test_acc:.4f}")
        with open(TRAINING_LOG_NAME, "a") as f:
            f.write(f"\t\tFinal Test Loss: {total_loss:.4f}\n")

if __name__ == "__main__":
    """
        Dataset downloaded from: https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595?file=21819066
    """
    parser = argparse.ArgumentParser(description="Spike Detection on Neuropixel")
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed"
    )  # 1337, 5673, 1234
    parser.add_argument(
        "--model_type",
        type=str,
        default="ae",
        choices=["ae", "kmeans"],
        help="Model used for spike sorting",
    )
    parser.add_argument(
        "--acc_mode",
        type=str,
        default="mse",
        choices=["mse", "mae"],
        help="Accuracy mode for training and testing",
    )
    parser.add_argument(
        "--test_only",
        type=bool,
        default=False,
        help="If True, only test the model without training",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension for the autoencoder",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=16,
        help="Latent dimension for the autoencoder",
    )

    args = parser.parse_args()

    BATCH_SIZE = 256  # 128 or 64
    NUM_EPOCHS = 20  # 60 epochs seems to work for lstm + lif model. slstm + lif seems to need more epochs.

    TRAINING_LOG_PATH = "./spike_sorting_training_log"
    if not os.path.exists(TRAINING_LOG_PATH):
        os.makedirs(TRAINING_LOG_PATH)
    TRAINING_LOG_NAME = f"{TRAINING_LOG_PATH}/training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"  # noqa: DTZ005

    SEED = args.seed  # 1337, 5673, 1234
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

    encoder = "dm"  # "dm" or "dv"
    reset_mechanism = "subtract"  # "none", "subtract", "zero"
    train_test_split_ratio = 0.7  # 70% training, 30% testing
    encoder_threshold = 0.2
    lif_tau = 1 * (1 / 24000)
    detection_window_size = 8
    sorting_window_size = 32  # 2ms

    MODEL_FILENAME_ACC = "./intracortical_weights/spike_sorting_kmeans_best_model_acc.pth"
    MODEL_FILENAME_LOSS = "./intracortical_weights/spike_sorting_ae_best_model_loss.pth"
    DETECTION_IDX = "Intracortical_Spike_Detection_IDX.txt"

    # parse the detection_idx file to get the detection idx and spike time idx for each file
    # The reason for this is that other papers only do spike sorting on the detected spikes,
    # not the entire signal.
    detection_idx_results_when_detected = {}
    detection_idx_results_label_spike_time = {}
    with open(DETECTION_IDX, "r") as f:
        detection_idx_lines = f.readlines()

    line_no = 0
    while line_no < len(detection_idx_lines):
        line = detection_idx_lines[line_no]
        if "Filename: " in line:
            filename = line.split("Filename: ")[1][:-2]
            detection_idx_results_when_detected[filename] = []
            detection_idx_results_label_spike_time[filename] = []
        else:
            temp = line.split(",")
            detected_idx = int(temp[0].split("Detection idx: ")[1])
            spike_time_idx = int(temp[1].split(" Spike time idx: ")[1])
            detection_idx_results_when_detected[filename].append(detected_idx)
            detection_idx_results_label_spike_time[filename].append(spike_time_idx)

        line_no += 1

    (
        complete_train_data,
        complete_train_labels,
        complete_test_data,
        complete_test_labels,
    ) = [], [], {}, {}
    for difficulty in ["Difficult1", "Difficult2", "Easy1", "Easy2"]:
        for noise_level in ["005", "01", "015", "02"]:
            filename = f"C_{difficulty}_noise{noise_level}.mat"

            with open(TRAINING_LOG_NAME, "a") as f:
                f.write(f"Currently Loading: filename: {filename}\n\n")

            (
                signal,
                spike_class_label,
                spike_times,
                sampling_interval,
                sampling_rate,
                spike_pulse_1ms_idx_length,
                spike_classes,
                filtered_signal,
            ) = load_dataset_intracortical(filepath, filename)

            if encoder == "dm":
                on_threshold = encoder_threshold
                off_threshold = -encoder_threshold
                event_stream = generate_event_stream_dm(
                    filtered_signal, on_threshold, off_threshold
                )
                spike_train = np.zeros_like(signal)
                spike_train[event_stream[:, 0].astype(int)] = (
                    event_stream[:, 1] - event_stream[:, 2]
                )
            elif encoder == "dv":
                dv_u_hist, spike_train, dv_time_lif = dv_to_lif_spike_gen(
                    signal=filtered_signal,
                    lif_threshold=encoder_threshold,
                    sampling_interval=sampling_interval,
                    lif_tau=lif_tau,
                    reset_mechanism=reset_mechanism,
                )
            else:
                raise ValueError("Invalid encoder type. Choose either 'dm' or 'dv'.")

            detected_spikes = np.array(detection_idx_results_label_spike_time[filename])
            detected_spike_times = np.array(
                detection_idx_results_when_detected[filename]
            )
            all_spike_signals = {i: [] for i in spike_classes}
            all_spk_trains = {i: [] for i in spike_classes}
            for i in range(len(spike_times)):
                if i in detected_spikes:
                    idx = np.where(detected_spikes == i)[0][0]
                    all_spike_signals[spike_class_label[i]].append(
                        filtered_signal[
                            detected_spike_times[idx]
                            - detection_window_size : detected_spike_times[idx]
                            + sorting_window_size
                            - detection_window_size
                        ]
                    )
                    all_spk_trains[spike_class_label[i]].append(
                        spike_train[
                            detected_spike_times[idx]
                            - detection_window_size : detected_spike_times[idx]
                            + sorting_window_size
                            - detection_window_size
                        ]
                    )

            (
                train_spk_train,
                test_spk_train,
                train_signal,
                test_signal,
                train_label,
                test_label,
            ) = train_test_split_spike_sorting(
                spike_classes, all_spk_trains, all_spike_signals, train_test_split_ratio
            )

            training_spikes_tensor = torch.tensor(
                np.array(train_spk_train), dtype=torch.float32
            )  # train_spk_train or filtered_spk_trains
            training_labels_tensor = (
                torch.tensor(train_label, dtype=torch.long) - 1
            )  # Offset by 1 to start from 0

            test_spikes_tensor = torch.tensor(
                np.array(test_spk_train), dtype=torch.float32
            )  # test_spk_train or filtered_spk_trains_test
            test_labels_tensor = (
                torch.tensor(test_label, dtype=torch.long) - 1
            )  # Offset by 1 to start from 0

            complete_train_data.append(training_spikes_tensor)
            complete_train_labels.append(training_labels_tensor)
            complete_test_data[filename[:-4]] = test_spikes_tensor
            complete_test_labels[filename[:-4]] = test_labels_tensor

    complete_train_data = torch.cat(complete_train_data, dim=0)
    complete_train_labels = torch.cat(complete_train_labels, dim=0)

    train_dataset = IntracorticalDataset(
        complete_train_data,
        complete_train_labels,
        # transform=v2.Compose([
        #     SwapAdjacent(p=0.5)
        # ])
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    if args.model_type == "ae":
        net = LSTMAutoencoder(
            input_dim=1,
            hidden_size=args.hidden_dim,
            latent_dim=args.latent_dim,
        )
        net.to(DEVICE)
        test_net = copy.deepcopy(net)

        optimiser = torch.optim.AdamW(
            net.parameters(), lr=2e-3, betas=(0.9, 0.999), weight_decay=0.1
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=NUM_EPOCHS, eta_min=1e-6
        )
    elif args.model_type == "kmeans":
        ae = LSTMAutoencoder(
            input_dim=1,
            hidden_size=args.hidden_dim,
            latent_dim=args.latent_dim,
        )
        ae.to(DEVICE)

        ae.load_state_dict(torch.load(MODEL_FILENAME_LOSS, weights_only=True))

        for p in ae.parameters():
            p.requires_grad_(False)

        # get the latent representations for the training data
        latent_representations = []
        ae.eval()
        with torch.no_grad():
            for data, label in train_loader:
                data = data.to(DEVICE)
                latent_representation, reconstructed = ae(data)
                latent_representations.append(latent_representation.cpu())
        latent_representations = torch.cat(latent_representations, dim=0).numpy()

        # get initial cluster centroids
        kmeans = KMeans(n_clusters=len(spike_classes), n_init=10, random_state=SEED)
        kmeans.fit(latent_representations)
        initial_centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        net = KMeansHead(
            hidden_dims=args.latent_dim,  # This should match the latent dimension of the autoencoder
            num_clusters=len(spike_classes),
            initial_centroids=initial_centroids,
        )
        net.to(DEVICE)
        test_ae_net = copy.deepcopy(ae)
        test_net = copy.deepcopy(net)

        optimiser = torch.optim.AdamW(
            net.parameters(), lr=2e-3, betas=(0.9, 0.999), weight_decay=0.1
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=NUM_EPOCHS, eta_min=1e-6
        )
    else:
        raise ValueError(
            "Invalid model type. Choose either 'raf', 'window', or 'non_raf'."
        )

    if args.acc_mode == "mse":
        loss_fn = nn.MSELoss()
    elif args.acc_mode == "mae":
        loss_fn = nn.L1Loss()

    if not args.test_only:
        if args.model_type == "ae":
            train_ae(
                net,
                train_loader,
                optimiser,
                loss_fn,
                scheduler=scheduler,
            )  # acc_mode="temporal" or "count"
        elif args.model_type == "kmeans":
            train_kmeans(
                ae,
                net,
                train_loader,
                optimiser,
                loss_fn,
                scheduler=scheduler,
            )

    for difficulty in ["Difficult1", "Difficult2", "Easy1", "Easy2"]:
        for noise_level in ["005", "01", "015", "02"]:
            filename = f"C_{difficulty}_noise{noise_level}.mat"

            test_dataset = IntracorticalDataset(
                complete_test_data[filename[:-4]], complete_test_labels[filename[:-4]]
            )
            test_loader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
            )

            with open(TRAINING_LOG_NAME, "a") as f:
                f.write(f"Currently Testing: {filename}\n\n")

            if args.model_type == "ae":
                test_ae(test_net, test_loader, loss_fn=loss_fn, final_test=True)
            elif args.model_type == "kmeans":
                test_kmeans(test_ae_net, test_net, test_loader, loss_fn=loss_fn, n_clusters=len(spike_classes), n_classes=len(spike_classes), final_test=True)

            with open(TRAINING_LOG_NAME, "a") as f:
                f.write("\n")
