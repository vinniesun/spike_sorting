import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import (
    # ellip, 
    lfilter, 
    butter, 
    # find_peaks, 
    # welch
)
from tqdm import tqdm
import csv

def load_neuropixel(
    filename: str,
    sampling_rate,
    window_size,
    num_rows,
    save=False
):
    raw_data =np.fromfile(
        filename,
        dtype=np.int16,
        count=int((window_size) * sampling_rate * num_rows),
        # offset = start_time * sampling_rate * num_rows * 2
    )

    raw_data = raw_data.reshape(-1, 385).T
    raw_data = raw_data[:, :int(window_size * sampling_rate)]
    raw_data = raw_data[:384, :]
    # raw_data_unfiltered = np.copy(raw_data)
    print("data from recording: ", raw_data.shape)

    raw_data = raw_data - np.mean(raw_data, axis=1).reshape(-1, 1)
    raw_data = raw_data - np.median(raw_data, axis=0)
    # t = np.arange(0, raw_data.shape[1]) / sampling_rate

    for i in tqdm(range(raw_data.shape[0])):
        b, a = butter(2, [300 * 2 / sampling_rate, 3000 * 2 / sampling_rate], btype="band")
        raw_data[i, :] = lfilter(b, a, raw_data[i, :])

        if save:
            np.save(f"{filepath}raw_data_filtered_channel_{i}.npy", raw_data[i, :])

    return raw_data

if __name__ == "__main__":
    ############ 
    # Dataset Downloaded from: https://rdr.ucl.ac.uk/articles/dataset/Chronic_recordings_from_Neuropixels_2_0_probes_in_mice/24411841
    #
    # To use the dataset, we need to install the following package through pip:
    # pip install mtscomp
    #
    # The decompress with the following command:
    # mtsdecomp Neuropixel_Dataset/AL036_2020-03-11/AL036_2020-03-11_stripe240_NatIm_g0_t0.imec0.ap.cbin \
    # -o Neuropixel_Dataset/AL036_2020-03-11/AL036_2020-03-11_stripe240_NatIm_g0_t0.imec0.ap.bin
    #
    # Then we can run this code
    ############
    filename = "AL036_2020-03-11_stripe240_NatIm_g0_t0.imec0.ap.bin"
    filepath = "./Neuropixel/AL036_2020-03-11/AL036_2020-03-11/"
    complete_filename = filepath + filename
    ks_filepath = "./Neuropixel/AL036_2020-03-11/AL036_2020-03-11/ks"

    SAVE_NEUROPIXEL = False

    sampling_rate = 30000
    sampling_interval = 1 / sampling_rate
    num_rows = 385
    num_channels = 384
    window_size = 240 # 240 seconds worth of data

    total_samples = int(window_size * sampling_rate)

    if SAVE_NEUROPIXEL:
        raw_data = load_neuropixel(
            complete_filename,
            sampling_rate,
            window_size,
            num_rows,
            save=SAVE_NEUROPIXEL
        )
        # np.save(f"{filepath}raw_data.npy", raw_data)
    else:
        raw_data = []
        for i in range(num_channels):
            raw_data_channel = np.load(f"{filepath}raw_data_filtered_channel_{i}.npy")
            raw_data.append(raw_data_channel)
        raw_data = np.array(raw_data)
        print(f"loaded data is of shape: {raw_data.shape}")

    gt_spike_times = np.load(f"{ks_filepath}/spike_times.npy")
    gt_spike_clusters = np.load(f"{ks_filepath}/spike_clusters.npy")

    with open(f"{ks_filepath}/cluster_KSLabel.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        cluster_KSLabel = list(reader)

    # For KSLabel, the following labels are possible: "good", "mua", "noise".
    # good: this classification is user determined, and is generally meant to be used to label cells that are well-isolated to a high level of confidence.
    # mua (Multi-unit activity): category for cells that are not well isolated.
    # usually because they consist of poorly isolated cells (e.g. they do not have a clean refractory period)
    #
    # In our work, we will only consider clusters with the "good" label.
    good_kslabel = []
    for id, label in cluster_KSLabel:
        if label == "good":
            good_kslabel.append(int(id))

    good_kslabel = np.array(good_kslabel)
    # print(type(cluster_KSLabel))
    # print(f"Good KS labels: {good_kslabel}")

    spike_times = np.load(f"{ks_filepath}/spike_times.npy") # vector giving the spike time of each spike in samples
    spike_clusters = np.load(f"{ks_filepath}/spike_clusters.npy") # vector giving the cluster identity of each spike
    spike_templates = np.load(f"{ks_filepath}/spike_templates.npy") # vector giving the template identity of each spike
    templates = np.load(f"{ks_filepath}/templates.npy") # matrix giving the template waveform for each template. shape (364, 82, 384) [nTemplates, nTimePoints, nTempChannels]
    whitening_inv = np.load(f"{ks_filepath}/whitening_mat_inv.npy") # matrix giving the inverse of the whitening matrix. shape (384, 384)
    pc_features_ind = np.load(f"{ks_filepath}/pc_feature_ind.npy")
    # print(templates.shape) # 

    # Unwhiten templates so amplitudes reflect real voltage
    templates_unwhitened = templates @ whitening_inv

    # plot all of the good cluster's templates
    # for cluster_id in tqdm(good_kslabel):
    #     curr_template = templates_unwhitened[cluster_id]

    #     fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    #     for channel_id in range(curr_template.shape[1]):
    #         ax.plot(curr_template[:, channel_id])

    #     ax.set_xlabel("Time (samples)")
    #     ax.set_ylabel("Voltage")

    #     plt.savefig(f"neuropixel_template_plots/cluster_{cluster_id}_template.png")
    #     plt.close()

    # plot example of single spike waveform of a good cluster
    i = 0
    while not np.any(spike_clusters[i] == good_kslabel):
        i += 1

    i += 1
    while not np.any(spike_clusters[i] == good_kslabel):
        i += 1

    i += 1
    while not np.any(spike_clusters[i] == good_kslabel):
        i += 1

    i += 1
    while not np.any(spike_clusters[i] == good_kslabel):
        i += 1

    i += 1
    while not np.any(spike_clusters[i] == good_kslabel):
        i += 1

    spike_time = spike_times[i]

    affected_channels = pc_features_ind[spike_templates[i]] # (32)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for chan_id in affected_channels:
        spike_waveform = raw_data[chan_id, spike_time-40:spike_time+40]
        ax.plot(spike_waveform)

    plt.savefig(f"neuropixel_waveform_check_cluser_{spike_clusters[i]}.png")
    plt.close()
