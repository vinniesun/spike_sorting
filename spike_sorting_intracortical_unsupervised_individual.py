import argparse
import copy
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hungarian algorithm for optimal assignment
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *
from utils import (
    IntracorticalDataset,
    IntracorticalDatasetForDEC,
    dv_to_lif_spike_gen,
    generate_event_stream_dm,
    huber_vr,
    load_dataset_intracortical,
    target_distribution,
    train_test_split_spike_sorting,
    van_rossum_distance,
    vr_mae_count,
    weighted_huber_loss,
)


def train_ae(
    net,
    train_loader,
    optimiser,
    loss_fn,
    args,
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

            if args.acc_mode == "huber":
                loss = loss_fn(reconstructed, data, event_weight=10.0, zero_weight=1.0, delta=1.0)
            elif args.acc_mode == "vr":
                loss = loss_fn(reconstructed, data, tau=args.tau, dt=1/24000) # tau of 10us
            elif args.acc_mode == "huber_vr":
                loss = loss_fn(reconstructed, data, event_weight=10.0, zero_weight=1.0, delta=1.0, tau=args.tau, dt=1/24000)
            elif args.acc_mode == "vr_mae_count":
                loss = loss_fn(reconstructed, data, tau=args.tau, dt=1/24000)
            else:
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

        if scheduler is not None:
            scheduler.step()

def train_clustering(
    ae,
    net,
    train_dataset,
    train_loader,
    test_loader, 
    optimiser,
    loss_fn,
    args,
    kmeans_acc,
    n_clusters=3,
    scheduler=None,
):
    best_recon_loss = float("inf")
    best_cluster_loss = float("inf")
    best_acc = kmeans_acc
    dec_loss_fn = nn.KLDivLoss(reduction="batchmean")  # KL divergence loss for clustering
    reconstruction_weight = 0.1

    torch.save(net.state_dict(), MODEL_FILENAME_ACC)
    torch.save(ae.state_dict(), MODEL_FILENAME_DEC_AE)

    # get all data for target_distribution
    target_distribution_data = train_dataset.spikes

    for epoch in tqdm(range(NUM_EPOCHS)):
        if epoch % 5 == 0: # try refreshing every 5-10 epochs
            ae.eval()
            net.eval()
            with torch.no_grad():
                target_distribution_data = target_distribution_data.to(DEVICE)
                latent_representation, reconstructed, _ = ae(target_distribution_data)
                cluster_assignment = net(latent_representation)
            target = target_distribution(cluster_assignment).detach()

        net.train()
        ae.train()
        curr_loss = 0.0
        total_recon_loss = 0.0
        total_cluster_loss = 0.0
        for data, label, idx in train_loader:
            data = data.to(DEVICE)  # shape (batch_size, seq_len)
            label = label.to(DEVICE)  # shape (batch_size)
            # print(f"data shape: {data.shape}, label shape: {label.shape}")

            # with torch.no_grad():
            #     latent_representation, reconstructed, _ = ae(data)
            latent_representation, reconstructed, _ = ae(data)
            cluster_assignment = net(latent_representation)

            # cluster_loss = dec_loss_fn(cluster_assignment.log(), target) / cluster_assignment.shape[0]  # KL divergence loss
            cluster_loss = dec_loss_fn(cluster_assignment.log(), target[idx])  # KL divergence loss

            if args.acc_mode == "huber":
                reconstruction_loss = loss_fn(reconstructed, data, event_weight=10.0, zero_weight=1.0, delta=1.0)
            elif args.acc_mode == "vr":
                reconstruction_loss = loss_fn(reconstructed, data, tau=args.tau, dt=1/24000) # tau of 10us
            elif args.acc_mode == "huber_vr":
                reconstruction_loss = loss_fn(reconstructed, data, event_weight=10.0, zero_weight=1.0, delta=1.0, tau=args.tau, dt=1/24000)
            elif args.acc_mode == "vr_mae_count":
                reconstruction_loss = loss_fn(reconstructed, data, tau=args.tau, dt=1/24000)
            else:
                reconstruction_loss = loss_fn(reconstructed, data)

            normalized_reconstruction_loss = reconstruction_loss / reconstruction_loss.detach().clamp_min(1.0)
            loss = cluster_loss + reconstruction_weight * normalized_reconstruction_loss

            curr_loss += loss.item()
            total_recon_loss += reconstruction_loss.item()
            total_cluster_loss += cluster_loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        test_acc = test_clustering(
            ae, 
            net, 
            test_loader, 
            loss_fn=loss_fn, 
            n_clusters=n_clusters, 
            n_classes=n_clusters, 
            args=args,
            final_test=False,
            load_checkpoint=False,
            load_ae_checkpoint=False,
        )

        if test_acc > best_acc:
            torch.save(net.state_dict(), MODEL_FILENAME_ACC)
            torch.save(ae.state_dict(), MODEL_FILENAME_DEC_AE)
            best_acc = test_acc

        # if total_cluster_loss < best_cluster_loss:
        #     torch.save(net.state_dict(), MODEL_FILENAME_ACC)
        #     best_cluster_loss = total_cluster_loss
        # if total_recon_loss < best_recon_loss:
        #     torch.save(ae.state_dict(), MODEL_FILENAME_LOSS)
        #     best_recon_loss = total_recon_loss

        with open(TRAINING_LOG_NAME, "a") as f:
            f.write(f"\tEpoch {epoch + 1}/{NUM_EPOCHS}, Total Loss: {curr_loss:.4f}, Reconstruction Loss: {total_recon_loss:.4f}, Clustering Loss: {total_cluster_loss:.4f}, Test acc: {test_acc:.4f}\n")
        # tqdm.write(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Accuracy: {train_acc:.4f}, Loss: {curr_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

def test_clustering(
    test_ae_net,
    net,
    test_loader,
    loss_fn,
    n_clusters,
    n_classes,
    args,
    final_test: bool = False,
    load_checkpoint: bool = True,
    load_ae_checkpoint: bool = True,
):
    if load_ae_checkpoint:
        test_ae_net.load_state_dict(torch.load(MODEL_FILENAME_DEC_AE, weights_only=True))
    test_ae_net.to(DEVICE)
    test_ae_net.eval()

    if load_checkpoint:
        net.load_state_dict(torch.load(MODEL_FILENAME_ACC, weights_only=True))
    net.to(DEVICE)
    net.eval()

    predictions, labels = [], []

    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            with torch.no_grad():
                latent_representation, reconstructed, _ = test_ae_net(data)
            # latent_representation = F.normalize(latent_representation, p=2, dim=1)  # Normalize the latent representation
            cluster_assignment = net(latent_representation)

            predictions.append(cluster_assignment.argmax(dim=1).cpu().numpy())
            labels.append(label.cpu().numpy())

    counts = np.zeros((n_clusters, n_classes), dtype=np.int64)
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    np.add.at(counts, (predictions, labels), 1)

    rows, cols = linear_sum_assignment(-counts)
    final_acc = counts[rows, cols].sum() / labels.shape[0]

    if final_test:
        # tqdm.write(f"Final Test Accuracy: {test_acc:.4f}")
        with open(TRAINING_LOG_NAME, "a") as f:
            f.write(f"\t\tFinal Test Accuracy: {final_acc:.4f}\n")
    else:
        return final_acc

def test_ae(
    net,
    test_loader,
    loss_fn,
    args,
    final_test: bool = False,
):
    net.load_state_dict(torch.load(MODEL_FILENAME_LOSS, weights_only=True))
    net.to(DEVICE)
    net.eval()

    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            latent_representation, reconstructed, _ = net(data)

            if args.acc_mode == "huber":
                loss = loss_fn(reconstructed, data, event_weight=10.0, zero_weight=1.0, delta=1.0)
            elif args.acc_mode == "vr":
                loss = loss_fn(reconstructed, data, tau=args.tau, dt=1/24000) # tau of 10us
            elif args.acc_mode == "huber_vr":
                loss = loss_fn(reconstructed, data, event_weight=10.0, zero_weight=1.0, delta=1.0, tau=args.tau, dt=1/24000)
            elif args.acc_mode == "vr_mae_count":
                loss = loss_fn(reconstructed, data, tau=args.tau, dt=1/24000)
            else:
                loss = loss_fn(reconstructed, data)
            total_loss += loss.item()
            total_samples += data.shape[0]

    if final_test:
        # tqdm.write(f"Final Test Accuracy: {test_acc:.4f}")
        with open(TRAINING_LOG_NAME, "a") as f:
            f.write(f"\t\tFinal Average Test Loss: {total_loss/total_samples:.4f}\n")

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
        choices=["mse", "mae", "huber", "vr", "huber_vr", "vr_mae_count"],
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
    parser.add_argument(
        "--encoder_th",
        type=float,
        default=0.1,
        help="Encoder threshold for the DM or DV encoder",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="dm",
        choices=["dm", "dv"],
        help="Encoder type for the DM or DV encoder",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=5e-6,
        help="Time constant for the VR loss function",
    )

    args = parser.parse_args()

    BATCH_SIZE = 256  # 128 or 64
    if args.model_type == "ae":  # noqa: SIM114
        NUM_EPOCHS = 500
    elif args.model_type == "kmeans":
        NUM_EPOCHS = 500

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

    encoder = args.encoder  # "dm" or "dv"
    reset_mechanism = "subtract"  # "none", "subtract", "zero"
    train_test_split_ratio = 0.7  # 70% training, 30% testing
    encoder_threshold = args.encoder_th
    lif_tau = 1 * (1 / 24000)
    detection_window_size = 8
    sorting_window_size = 32  # 2ms

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

            MODEL_FILENAME_ACC = f"./intracortical_weights/spike_sorting_kmeans_{filename[:-4]}_best_model_acc.pth"
            MODEL_FILENAME_LOSS = f"./intracortical_weights/spike_sorting_ae_{filename[:-4]}_best_model_loss.pth"
            MODEL_FILENAME_DEC_AE = f"./intracortical_weights/spike_sorting_dec_{filename[:-4]}_best_model_ae.pth"

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

            # ae = LSTMAutoencoder(
            #     input_dim=1,
            #     seq_len=training_spikes_tensor.shape[1],
            #     hidden_size=args.hidden_dim,
            #     latent_dim=args.latent_dim,
            # )
            ae = Conv1DLSTMAE(
                seq_len=training_spikes_tensor.shape[1],
                input_dim=1,
                out_channels=32,
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim,
                kernel_size=4,
                kernel_stride=2,
                padding=1,
            )
            # ae = Conv1DAE(
            #     input_dim=1,
            #     seq_len=training_spikes_tensor.shape[1],
            #     hidden_size=(32, 64, 96),
            #     latent_dim=args.latent_dim,
            # )
            ae.to(DEVICE)
            test_ae_net = copy.deepcopy(ae)

            optimiser = torch.optim.AdamW(
                ae.parameters(), lr=2e-3, betas=(0.9, 0.999), weight_decay=0.1
            )

            if args.acc_mode == "mse":
                loss_fn = nn.MSELoss()
            elif args.acc_mode == "mae":
                loss_fn = nn.L1Loss()
            elif args.acc_mode == "huber":
                loss_fn = weighted_huber_loss
            elif args.acc_mode == "vr":
                loss_fn = van_rossum_distance
            elif args.acc_mode == "huber_vr":
                loss_fn = huber_vr
            elif args.acc_mode == "vr_mae_count":
                loss_fn = vr_mae_count

            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimiser, T_max=NUM_EPOCHS, eta_min=1e-5
            # )
            scheduler = None

            if args.model_type == "ae":
                train_dataset = IntracorticalDataset(
                    training_spikes_tensor,
                    training_labels_tensor,
                    # transform=v2.Compose([
                    #     SwapAdjacent(p=0.5)
                    # ])
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
                )
                test_dataset = IntracorticalDataset(
                    test_spikes_tensor,
                    test_labels_tensor,
                    # transform=v2.Compose([
                    #     SwapAdjacent(p=0.5)
                    # ])
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
                )

                train_ae(
                    ae,
                    train_loader,
                    optimiser,
                    loss_fn,
                    args,
                    scheduler=scheduler,
                )

                test_ae(test_ae_net, test_loader, loss_fn=loss_fn, args=args, final_test=True)

            elif args.model_type == "kmeans":
                train_dataset = IntracorticalDataset(
                    training_spikes_tensor,
                    training_labels_tensor,
                    # transform=v2.Compose([
                    #     SwapAdjacent(p=0.5)
                    # ])
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
                )
                test_dataset = IntracorticalDataset(
                    test_spikes_tensor,
                    test_labels_tensor,
                    # transform=v2.Compose([
                    #     SwapAdjacent(p=0.5)
                    # ])
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
                )
                ae.load_state_dict(torch.load(MODEL_FILENAME_LOSS, weights_only=True))

                # get the latent representations for the training data
                train_latent_representations = []
                ae.eval()
                with torch.no_grad():
                    for data, label in train_loader:
                        data = data.to(DEVICE)
                        latent_representation, reconstructed, _ = ae(data)
                        train_latent_representations.append(latent_representation.cpu())
                train_latent_representations = torch.cat(train_latent_representations, dim=0).numpy()

                # get initial cluster centroids/Kmeans
                kmeans = KMeans(n_clusters=len(spike_classes), n_init=50, random_state=SEED)
                kmeans.fit(train_latent_representations)
                initial_centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

                # SpectralClustering
                # sc = SpectralClustering(
                #     n_clusters=len(spike_classes),
                #     affinity="nearest_neighbors",
                #     n_neighbors=10,
                #     random_state=SEED,
                #     assign_labels="discretize",
                #     n_init=50
                # )
                # sc.fit(train_latent_representations)

                # GaussianMixture
                gmm = GaussianMixture(
                    n_components=len(spike_classes),
                    covariance_type="full",
                    random_state=SEED,
                    n_init=50
                )
                gmm.fit(train_latent_representations)

                test_latent_representations, labels = [], []
                ae.eval()
                with torch.no_grad():
                    for data, label in test_loader:
                        data = data.to(DEVICE)
                        latent_representation, reconstructed, _ = ae(data)
                        test_latent_representations.append(latent_representation.cpu())
                        labels.append(label.cpu())

                test_latent_representations = torch.cat(test_latent_representations, dim=0).numpy()
                labels = torch.cat(labels, dim=0).numpy()

                kmeans_predictions = kmeans.predict(test_latent_representations)
                # sc_predictions = sc.fit_predict(test_latent_representations)
                gmm_predictions = gmm.predict(test_latent_representations)

                # Calculate kmeans accuracy
                counts = np.zeros((len(spike_classes), len(spike_classes)), dtype=np.int64)
                np.add.at(counts, (kmeans_predictions, labels), 1)
                rows, cols = linear_sum_assignment(-counts)
                kmeans_acc = counts[rows, cols].sum() / labels.shape[0]

                with open(TRAINING_LOG_NAME, "a") as f:
                    f.write(f"\t\tKmeans Accuracy: {kmeans_acc:.4f}\n")

                # # Calculate SpectralClustering accuracy
                # counts = np.zeros((len(spike_classes), len(spike_classes)), dtype=np.int64)
                # np.add.at(counts, (sc_predictions, labels), 1)
                # rows, cols = linear_sum_assignment(-counts)
                # sc_acc = counts[rows, cols].sum() / labels.shape[0]
                
                # with open(TRAINING_LOG_NAME, "a") as f:
                #     f.write(f"\t\tSpectral Clustering Accuracy: {sc_acc:.4f}\n")

                # Calculate GMM accuracy
                counts = np.zeros((len(spike_classes), len(spike_classes)), dtype=np.int64)
                np.add.at(counts, (gmm_predictions, labels), 1)
                rows, cols = linear_sum_assignment(-counts)
                gmm_acc = counts[rows, cols].sum() / labels.shape[0]
            
                with open(TRAINING_LOG_NAME, "a") as f:
                    f.write(f"\t\tGMM Accuracy: {gmm_acc:.4f}\n")

                train_dataset = IntracorticalDatasetForDEC(
                    training_spikes_tensor,
                    training_labels_tensor,
                    # transform=v2.Compose([
                    #     SwapAdjacent(p=0.5)
                    # ])
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
                )

                net = DEC(
                    num_clusters=len(spike_classes),
                    hidden_dim=args.latent_dim,
                    alpha=0.5, # compare to see if its better to use 1.0 or 2.0
                    initial_centroids=initial_centroids,
                )

                net.to(DEVICE)
                test_ae_net = copy.deepcopy(ae)
                test_net = copy.deepcopy(net)

                # optimiser = torch.optim.AdamW(
                #     net.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5
                # )
                optimiser = torch.optim.AdamW(
                    [
                        {"params": ae.parameters(), "lr": 1e-5},
                        {"params": net.parameters(), "lr": 1e-4},
                    ],
                    weight_decay=1e-6,
                )
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                #     optimiser, T_max=NUM_EPOCHS, eta_min=1e-6
                # )
                scheduler = None

                train_clustering(
                    ae,
                    net,
                    train_dataset,
                    train_loader,
                    test_loader, 
                    optimiser,
                    loss_fn,
                    args,
                    kmeans_acc,
                    n_clusters=len(spike_classes),
                    scheduler=scheduler,
                )
                test_clustering(
                    test_ae_net, 
                    test_net, 
                    test_loader, 
                    loss_fn=loss_fn, 
                    n_clusters=len(spike_classes), 
                    n_classes=len(spike_classes), 
                    args=args,
                    final_test=True,
                    load_checkpoint=True,
                    load_ae_checkpoint=True,
                )

