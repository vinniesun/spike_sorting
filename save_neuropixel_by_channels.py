import numpy as np
import os
from tqdm import tqdm

if __name__ == "__main__":
    filename = "AL036_2020-03-11_stripe240_NatIm_g0_t0.imec0.ap.bin"
    filepath = "./Neuropixel/AL036_2020-03-11/AL036_2020-03-11/"
    complete_filename = filepath + filename

    save_dir = "./Neuropixel/AL036_2020-03-11/AL036_2020-03-11/by_channels/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    raw_data = np.fromfile(
        complete_filename,
        dtype=np.int16,
    )

    raw_data = raw_data.reshape(-1, 385).T
    assert raw_data.shape[0] == 385, "The number of channels should be 385"

    for channel in tqdm(range(raw_data.shape[0])):
        channel_data = raw_data[channel]
        np.save(f"{save_dir}/channel_{channel}.npy", channel_data)
