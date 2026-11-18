import csv

import numpy as np

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
    # phy documentation: https://phy.readthedocs.io/en/latest/sorting_user_guide/
    #
    # Then we can run this code
    ############
    filename = "AL036_2020-03-11_stripe240_NatIm_g0_t0.imec0.ap.bin"
    filepath = "./Neuropixel/AL036_2020-03-11/AL036_2020-03-11/ks"
    raw_filepath = "./Neuropixel/AL036_2020-03-11/AL036_2020-03-11/"

    spike_times = np.load(f"{filepath}/spike_times.npy")        # shape (3092428)
    spike_clusters = np.load(f"{filepath}/spike_clusters.npy")  # this is the final cluster id, shape (3092428)
    channel_map = np.load(f"{filepath}/channel_map.npy")        # shape (384)
    channel_positions = np.load(f"{filepath}/channel_positions.npy") # shape (384, 2)
    spike_templates = np.load(f"{filepath}/spike_templates.npy")# original assignment before manual curation. shape (3092428)
    pc_features = np.load(f"{filepath}/pc_features.npy")    # principle component features, shape (3092428, 3, 32)
    pc_features_ind = np.load(f"{filepath}/pc_feature_ind.npy") # shape (364, 32)
    template_ind = np.load(f"{filepath}/templates_ind.npy") # shape (384, 386)

    threshold = 4 # Signal must exceed 4 × channel noise

    raw_data = []
    for i in range(384):
        raw_data_channel = np.load(f"{raw_filepath}raw_data_filtered_channel_{i}.npy")
        raw_data.append(raw_data_channel)
    raw_data = np.array(raw_data)

    sample = int(spike_times[0])

    waveform = raw_data[:, sample-20:sample+20]

    noise_segment = raw_data[:, :30*30000] # shape (384)
    baseline = np.median(noise_segment, axis=1) # shape (384)
    mad = np.median(np.abs(noise_segment - baseline[:, np.newaxis]), axis=1) # shape (384)
    noise_sd = 1.4826 * mad # shape (384)

    trough_amplitude = baseline - waveform.min(axis=1) # shape (384)
    visible = (noise_sd > 0) & (trough_amplitude > threshold * noise_sd)
    print(noise_sd.shape, mad.shape, noise_segment.shape, baseline.shape, trough_amplitude.shape, visible.shape)

    # # Gather all of the good cluster id
    # good_cluster_id = []
    # with open(f"{filepath}/cluster_KSLabel.tsv", encoding='utf-8') as f:
    #     reader = csv.reader(f, delimiter='\t')
    #     for i, row in enumerate(reader):
    #         if i == 0:
    #             continue
    #         if row[1] == "good":
    #             good_cluster_id.append(int(row[0]))
    # print(good_cluster_id[:5])

    # # Now we want to find all of the channels responsible for the good clusters.
    # good_cluster_channels = set()
    # for cluster_id in good_cluster_id:
    #     channels = pc_features_ind[cluster_id]
    #     for channel in channels:
    #         good_cluster_channels.add(channel)
    # print(len(good_cluster_channels))
