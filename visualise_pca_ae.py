import argparse
import copy
import os
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
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


if __name__ == "__main__":
    """
        Dataset downloaded from: https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595?file=21819066
    """
    parser = argparse.ArgumentParser(description="Spike Detection on Neuropixel")
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed"
    )  # 1337, 5673, 1234
    parser.add_argument(
        "--visualise",
        type=str,
        default="pcm",
        choices=["pcm", "ae"],
        help="Model used for spike sorting",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension for the autoencoder",
    )

    args = parser.parse_args()

    BATCH_SIZE = 256  # 128 or 64
    NUM_EPOCHS = 20  # 60 epochs seems to work for lstm + lif model. slstm + lif seems to need more epochs.

    TRAINING_LOG_PATH = "./spike_sorting_training_log"
    if not os.path.exists(TRAINING_LOG_PATH):
        os.makedirs(TRAINING_LOG_PATH)

    SEED = args.seed  # 1337, 5673, 1234
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    PCA_ANALYSIS_DIR = "./pca_analysis/"

    curr_pca_path = f"{PCA_ANALYSIS_DIR}/{args.visualise}/"
    if not os.path.exists(curr_pca_path):
        os.makedirs(curr_pca_path)

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

            training_dataset = IntracorticalDataset(
                training_spikes_tensor, training_labels_tensor
            )
            test_dataset = IntracorticalDataset(
                test_spikes_tensor, test_labels_tensor
            )

            train_loader = DataLoader(
                training_dataset, batch_size=BATCH_SIZE, shuffle=True
            )
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            if args.visualise == "pcm":
                inputs_for_pca = torch.cat((training_spikes_tensor, test_spikes_tensor), dim=0).numpy()
                labels_for_pca = torch.cat((training_labels_tensor, test_labels_tensor), dim=0).numpy()
            elif args.visualise == "ae":
                inputs_for_pca = []
                labels_for_pca = []
                net = LSTMAutoencoder(
                    input_dim=1,
                    hidden_size=args.hidden_dim,
                )
                net.to(DEVICE)
                net.load_state_dict(torch.load(MODEL_FILENAME_LOSS, map_location=DEVICE, weights_only=True))
                net.eval()

                with torch.no_grad():
                    for data, label in train_loader:
                        data = data.to(DEVICE)
                        label = label.to(DEVICE)

                        latent_representation, reconstructed = net(data)

                        inputs_for_pca.append(latent_representation.cpu().numpy())
                        labels_for_pca.append(label.cpu().numpy())

                    for data, label in test_loader:
                        data = data.to(DEVICE)
                        label = label.to(DEVICE)
                
                        latent_representation, reconstructed = net(data)
                
                        inputs_for_pca.append(latent_representation.cpu().numpy())
                        labels_for_pca.append(label.cpu().numpy())

                inputs_for_pca = np.concatenate(inputs_for_pca, axis=0)
                labels_for_pca = np.concatenate(labels_for_pca, axis=0)

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(inputs_for_pca)

            colours = ['red', 'green', 'blue']
            labels = ['Class 1', 'Class 2', 'Class 3']

            fig, ax = plt.subplots(figsize=(8, 6))
            for i in range(len(spike_classes)):
                mask = labels_for_pca == i
                ax.scatter(pca_result[mask, 0], pca_result[mask, 1], color=colours[i], label=labels[i], alpha=0.5)

            ax.legend(loc="best")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")

            plt.tight_layout()
            plt.savefig(f"{curr_pca_path}/{filename[:-4]}_pca.jpg", dpi=300)
            plt.close()
