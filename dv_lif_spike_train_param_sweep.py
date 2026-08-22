import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import (
    load_dataset_intracortical,
    leaky_integrate_neuron,
    reconstruction_lif,
    calc_rmse,
    dv_to_lif_spike_gen
)

def compression_ratio(filtered_signal, spike_train):
    idx = np.where(spike_train != 0)[0]
    tdr_fs = filtered_signal.shape[0] * 12
    tdr_apm = idx.shape[0] if idx.shape[0] > 0 else 1 # remove (np.ceil(np.log2(10000)) + 1) to be consistent with ASC?

    return tdr_fs / tdr_apm

def plot_heatmap(results: list, noise_level: str):
    # Extract unique sorted x and y values
    x_vals = sorted(set(x for x, y, rmse, cr in results))
    y_vals = sorted(set(y for x, y, rmse, cr in results))

    # Create lookup dictionaries for grid indices
    x_to_idx = {x: i for i, x in enumerate(x_vals)}
    y_to_idx = {y: i for i, y in enumerate(y_vals)}

    # Create empty RMSE grid
    rmse_grid = np.full((len(y_vals), len(x_vals)), np.nan)
    cr_grid = np.full((len(y_vals), len(x_vals)), np.nan)

    # Fill grid
    for x, y, rmse, cr in results:
        i = y_to_idx[y]
        j = x_to_idx[x]
        rmse_grid[i, j] = rmse
        cr_grid[i, j] = cr

    ################### Plot RMSE heatmap
    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(
        rmse_grid,
        cmap="viridis_r",
        origin="lower",
        aspect="auto"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RMSE")

    # Set tick labels to original parameter values
    ax.set_xticks(
        ticks=np.arange(len(x_vals)),
        labels=x_vals
    )

    ax.set_yticks(
        ticks=np.arange(len(y_vals)),
        labels=y_vals
    )

    ax.set_xticks(np.arange(len(x_vals) + 1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(y_vals) + 1)-0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            value = rmse_grid[i, j]
            if not np.isnan(value):
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    rotation=45,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9
                )

    ax.set_xlabel("LIF Threshold")
    ax.set_ylabel("LIF Tau")

    plt.tight_layout()
    plt.savefig(f"compare_spike_trains/rmse_heatmap_{noise_level}.jpg", dpi=300)
    plt.close()

    ################ Plot CR heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(
        cr_grid,
        cmap="viridis_r",
        origin="lower",
        aspect="auto"
    )
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CR")
    
    # Set tick labels to original parameter values
    ax.set_xticks(
        ticks=np.arange(len(x_vals)),
        labels=x_vals
    )
    
    ax.set_yticks(
        ticks=np.arange(len(y_vals)),
        labels=y_vals
    )
    
    ax.set_xticks(np.arange(len(x_vals) + 1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(y_vals) + 1)-0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            value = cr_grid[i, j]
            if not np.isnan(value):
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    rotation=45,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9
                )
    
    ax.set_xlabel("LIF Threshold")
    ax.set_ylabel("LIF Tau")
    
    plt.tight_layout()
    plt.savefig(f"compare_spike_trains/cr_heatmap_{noise_level}.jpg", dpi=300)
    plt.close()

if __name__ == "__main__":
    filepath = "./intracortical_dataset/"

    # parameters for sweeping
    lif_thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]) # old threshoolds: 0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0
    lif_tau = np.arange(start=1, stop=9, step=1) * 1/24000
    reset_mechanism = "zero" # "none", "subtract", "zero"

    results_005, results_01, results_015, results_02 = [], [], [], [] # each entry is a tuple of (lif_threshold, lif_tau, rmse, cr)
    for lif_threshold in tqdm(lif_thresholds, desc="Sweeping LIF Threshold"):
        for lt in lif_tau:
            rmse_temp_005, rmse_temp_01, rmse_temp_015, rmse_temp_02 = [], [], [], []
            cr_temp_005, cr_temp_01, cr_temp_015, cr_temp_02 = [], [], [], []
            for difficulty in ["Difficult1", "Difficult2", "Easy1", "Easy2"]:
                for noise_level in ["005", "01", "015", "02"]:
                    filename = f"C_{difficulty}_noise{noise_level}.mat"

                    signal, spike_class_label, spike_times, sampling_interval, \
                    sampling_rate, spike_pulse_1ms_idx_length, spike_classes, \
                    filtered_signal = load_dataset_intracortical(filepath, filename)

                    dv_u_hist, dv_spk_hist, dv_time_lif = dv_to_lif_spike_gen(
                        signal=filtered_signal,
                        lif_threshold=lif_threshold,
                        sampling_interval=sampling_interval,
                        lif_tau=lt,
                        reset_mechanism=reset_mechanism
                    )

                    reconstructed_signal = reconstruction_lif(dv_spk_hist, time_step=sampling_interval, reconstruct_tau=10*sampling_interval, alpha=0.8, order=2)
                    rmse = calc_rmse(filtered_signal, reconstructed_signal, spike_times)
                    cr = compression_ratio(filtered_signal, dv_spk_hist)

                    if noise_level == "005":
                        rmse_temp_005.append(rmse)
                        cr_temp_005.append(cr)
                    elif noise_level == "01":
                        rmse_temp_01.append(rmse)
                        cr_temp_01.append(cr)
                    elif noise_level == "015":
                        rmse_temp_015.append(rmse)
                        cr_temp_015.append(cr)
                    elif noise_level == "02":
                        rmse_temp_02.append(rmse)
                        cr_temp_02.append(cr)

            results_005.append((lif_threshold, lt, np.mean(rmse_temp_005), np.mean(cr_temp_005)))
            results_01.append((lif_threshold, lt, np.mean(rmse_temp_01), np.mean(cr_temp_01)))
            results_015.append((lif_threshold, lt, np.mean(rmse_temp_015), np.mean(cr_temp_015)))
            results_02.append((lif_threshold, lt, np.mean(rmse_temp_02), np.mean(cr_temp_02)))

    # plot heatmap of rmse for 005
    plot_heatmap(results_005, "005")

    # plot heatmap of rmse for 01
    plot_heatmap(results_01, "01")

    # plot heatmap of rmse for 015
    plot_heatmap(results_015, "015")

    # plot heatmap of rmse for 02
    plot_heatmap(results_02, "02")
