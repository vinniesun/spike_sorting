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

    spike_times = np.load(f"{filepath}/spike_times.npy")
    spike_clusters = np.load(f"{filepath}/spike_clusters.npy")
    channel_map = np.load(f"{filepath}/channel_map.npy")
    channel_positions = np.load(f"{filepath}/channel_positions.npy")
    spike_templates = np.load(f"{filepath}/spike_templates.npy")
    pc_features = np.load(f"{filepath}/pc_features.npy")    # principle component features
    pc_features_ind = np.load(f"{filepath}/pc_feature_ind.npy")
    print(spike_times.shape) # shape (3092428)
    print(spike_times[0])
    print(spike_clusters.shape) # shape (3092428)
    print(channel_map.shape)    # shape (384)
    print(channel_positions.shape)  # shape (384, 2)
    # print(channel_positions)
    print(spike_templates.shape)    # shape (3092428)
    # print(spike_templates[:10])
    print(pc_features.shape)        # shape (3092428, 3, 32)
    # print(pc_features[0])
    print(pc_features_ind.shape)    # shape (364, 32) i.e. there are 364 clusters and each cluster has 32 channels responsible for it.
    print(pc_features_ind[270])

    ########
    # spike_times provide the index for when the spike occurs.
    # spike_clusters provide the cluster id for each spike
    # pc_features_ind provide the channels responsible for each cluster group.
    #
    # For cluster 270, the responsible channels are :
    # [115 113 117 111 119 114 112 116 110 118 109 121 108 120 107 123 106 122
    # 105 125 104 124 103 127 102 126 101 129 100 128  99 131]
    # 
    ########