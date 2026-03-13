import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

import os
from scipy.io import loadmat
from scipy.signal import ellip, lfilter,butter,find_peaks

from layers import *
from neurons import *
from models import *
from grad_functions import *

SEED = 1337
START_IDX = 4200
END_IDX = 4500

INPUT_REFRACTORY_LIMIT = 12

class SingleBRF(RFCell):
    def __init__(self, 
                 input_size: int,
                 layer_size: int,
                 b_offset: float = DEFAULT_RF_B_offset,
                 adaptive_b_offset: bool = TRAIN_B_offset,
                 adaptive_b_offset_a: float = DEFAULT_RF_ADAPTIVE_B_offset_a,
                 adaptive_b_offset_b: float = DEFAULT_RF_ADAPTIVE_B_offset_b,
                 omega: float = DEFAULT_RF_OMEGA,
                 adaptive_omega: bool = TRAIN_OMEGA,
                 adaptive_omega_a: float = DEFAULT_RF_ADAPTIVE_OMEGA_a,
                 adaptive_omega_b: float = DEFAULT_RF_ADAPTIVE_OMEGA_b,
                 dt: float = DEFAULT_DT,
                 bias: bool = False,
                ):
        super().__init__(input_size, layer_size, b_offset=b_offset, adaptive_b_offset=adaptive_b_offset,
                         adaptive_b_offset_a=adaptive_b_offset_a, adaptive_b_offset_b=adaptive_b_offset_b,
                         omega=omega, adaptive_omega=adaptive_omega,
                         adaptive_omega_a=adaptive_omega_a, adaptive_omega_b=adaptive_omega_b,
                         dt=dt, bias=bias)
        
        self.linear = torch.nn.Linear(
                in_features=input_size,
                out_features=layer_size,
                bias=bias
            )
        torch.nn.init.constant_(self.linear.weight, 1.0)

        self.event_polarity = None
        self.input_refractory_period = 0

        # self.linear_pos = torch.nn.Linear(
        #         in_features=input_size,
        #         out_features=layer_size,
        #         bias=bias
        #     )
        # torch.nn.init.constant_(self.linear_pos.weight, 1.0)
        # self.linear_neg = torch.nn.Linear(
        #         in_features=input_size,
        #         out_features=layer_size,
        #         bias=bias
        #     )
        # torch.nn.init.constant_(self.linear_neg.weight, -1.0)

    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # in_sum = self.linear(x)

        # Only connect to positive weights.
        # if x > 0:
        #     in_sum = self.linear_pos(x)
        # else:
        #     in_sum = self.linear_neg(x)

        if self.event_polarity is None:
            if x > 0:
                self.event_polarity = 1
                self.input_refractory_period = 0
            elif x < 0:
                self.event_polarity = -1
                self.input_refractory_period = 0
        elif self.event_polarity == 1:
            if x > 0 and self.input_refractory_period <= INPUT_REFRACTORY_LIMIT:
                x = torch.zeros_like(x)
                self.input_refractory_period += 1
            elif x > 0 and self.input_refractory_period > INPUT_REFRACTORY_LIMIT:
                self.input_refractory_period = 0
            elif x == 0:
                self.input_refractory_period += 1
            elif x < 0:
                self.event_polarity = -1
                self.input_refractory_period = 0
        elif self.event_polarity == -1:
            if x < 0 and self.input_refractory_period <= INPUT_REFRACTORY_LIMIT:
                x = torch.zeros_like(x)
                self.input_refractory_period += 1
            elif x < 0 and self.input_refractory_period > INPUT_REFRACTORY_LIMIT:
                self.input_refractory_period = 0
            elif x == 0:
                self.input_refractory_period += 1
            elif x > 0:
                self.event_polarity = 1
                self.input_refractory_period = 0
        in_sum = self.linear(x)

        # if self.event_polarity is None:
        #     self.event_polarity = torch.sign(x)
        # elif self.event_polarity != torch.sign(x):
        #     self.event_polarity = torch.sign(x)
        # else:
        #     x = torch.zeros_like(x)
        # in_sum = self.linear(x)

        z, u, v, q = state

        omega = torch.abs(self.omega)

        p_omega = sustain_osc(omega)

        b_offset = torch.abs(self.b_offset)

        # divergence boundary
        b = p_omega - b_offset - q

        z, u, v, q = brf_update(
            x=in_sum,
            u=u,
            v=v,
            q=q,
            b=b,
            omega=omega,
            dt=self.dt,
        )

        return z, u, v, q, x

class SpikeDataset(Dataset):
    def __init__(self, data, label):
        self.data = data    # Dim: (1, seq_len)
        self.label = label  # Dim: (1, seq_len)

    def __len__(self):
        return self.data.shape[1]
    
    def __getitem__(self, idx):
        return self.data[:, idx], self.label[:, idx]

def plot_spikes(original_signal, pulse_train, spikes, ground_truth, label_single):
    # lim = [1320,1420]
    # lim = [3410,3460]
    # lim = [0,800]
    lim = [START_IDX, END_IDX]

    fig = plt.figure(figsize=(16, 16))
    plt.subplot(511)
    plt.plot(original_signal)
    plt.ylabel("Original Signal")
    plt.xlim(lim)
    plt.subplot(512)
    plt.stem(pulse_train[0], pulse_train[1])
    plt.ylabel("Pulse Train")
    plt.xlim(lim)
    plt.subplot(513)
    plt.stem(spikes[0, :])
    plt.ylabel("Generated Spikes")
    plt.xlim(lim)
    plt.subplot(514)
    plt.plot(ground_truth[0, :])
    plt.ylabel("Label")
    plt.xlim(lim)
    plt.subplot(515)
    plt.plot(label_single[0, :])
    plt.ylabel("Single Label")
    plt.xlim(lim)
    plt.savefig("signal_vs_spikes_vs_pulse_train.jpg")
    plt.close()

def convert_data_to_spiketrain(filepath, filename, multipier):
    MAT = loadmat(os.path.join(filepath, filename + ".mat"))
    data = np.array(MAT['data'])[0]
    sampling_interval =np.array(MAT['samplingInterval'][0][0]) * 1e-3
    sample_rate = 1/sampling_interval
    b, a = butter(4, [300*2/sample_rate, 5000*2/sample_rate], btype='band')
    data = lfilter(b, a, data)

    ABS_THD = 4*np.median(np.abs(data) / 0.6745)
    data_up = np.copy(data)
    data_up[data_up < ABS_THD] = 0
    peaks, _ = find_peaks(data_up)
    data_down = np.copy(data)
    data_down[data_down > -ABS_THD] = 0
    valleys, _ = find_peaks(abs(data_down))
    median_peak = np.median(data_up[peaks])
    median_valleys = np.median(data_down[valleys])
    spike_amplitude = (median_peak - median_valleys) / 2
    modulation_thd = spike_amplitude * multipier
    
    ### Replace the line below with your method of encoding
    pulseTrain = np.load(os.path.join(filepath, filename + ".npy")) 

    spikeTimeGT = np.array(MAT['OVERLAP_DATA'] > 0).astype(np.float32)
    data_len = spikeTimeGT.shape[1]
    spikeTimeGT = np.insert(spikeTimeGT, 0, [0 for _ in range(22)])
    spikeTimeGT = spikeTimeGT[:data_len].reshape(1, -1)
    
    spikeTimeGT_id = np.array(MAT['spike_times'])[0][0][0] + 22

    return data, pulseTrain, spikeTimeGT, sample_rate, spikeTimeGT_id

def load_data(filepath, filename):
    original_data, pulseTrain, spikeTimeGT, sample_rate, spikeTimeGT_single = convert_data_to_spiketrain(filepath, filename, multipier=0.3)
    
    spike_data_ind = (pulseTrain[0]/10).astype(int).reshape(-1,1)
    spike_num = pulseTrain[1].reshape(-1,1)
    spikeTimeGT_single = spikeTimeGT_single.astype(int)

    spike = np.zeros_like(spikeTimeGT)
    for i, id in enumerate(spike_data_ind):
        spike[:, id] = spike_num[i]

    label_single = np.zeros_like(spikeTimeGT)
    for i, id in enumerate(spikeTimeGT_single):
        label_single[:, id] = np.ones(1)

    pulseTrain[0] = pulseTrain[0]/10
    # print(spikeTimeGT.shape)
    plot_spikes(original_data, pulseTrain, spike, spikeTimeGT, label_single)
    
    return original_data, spike, spikeTimeGT, sample_rate, label_single

def single_neuron_tests(spikes, original_data):
    single_rf_neuron = SingleBRF(
                                    input_size=1,
                                    layer_size=1,

                                )
    # print(single_rf_neuron.linear.weight)
    input_spikes = torch.from_numpy(spikes[:, START_IDX:END_IDX]) # Need to split the spikes into train/val/test set.
    # print(input_spikes)

    hidden_z = torch.zeros((single_rf_neuron.layer_size))
    hidden_u = torch.zeros_like(hidden_z)
    hidden_v = torch.zeros_like(hidden_z)
    hidden_q = torch.zeros_like(hidden_z)

    hidden_u_hist, hidden_v_hist = [], []
    x_hist = []
    for t in range(input_spikes.shape[1]):
        hidden = hidden_z, hidden_u, hidden_v, hidden_q
        hidden_z, hidden_u, hidden_v, hidden_q, x = single_rf_neuron(
            input_spikes[:, t],
            hidden
        )

        hidden_u_hist.append(hidden_u)
        hidden_v_hist.append(hidden_v)
        x_hist.append(x)
    
    hidden_u_hist = torch.stack(hidden_u_hist)
    hidden_v_hist = torch.stack(hidden_v_hist)
    x_hist = torch.stack(x_hist)
    # print(x_hist.shape)
    print(f"Maximum membrane potential: {torch.max(hidden_u_hist)}") # 5.3892854339210317e-05
    print(f"Minimum membrane potential: {torch.min(hidden_u_hist)}") # -6.793563807150349e-05

    # print(hidden_u_hist.shape, hidden_v_hist.shape)

    fig = plt.figure(figsize=(16, 16))
    plt.subplot(511)
    plt.plot(original_data[START_IDX:END_IDX])
    plt.ylabel("Original Signal")
    plt.subplot(512)
    plt.stem(input_spikes[0, :])
    plt.ylabel("Input Spike Train")
    plt.subplot(513)
    plt.stem(x_hist[:, 0])
    plt.ylabel("Input Processed")
    plt.subplot(514)
    plt.stem(hidden_u_hist[:, 0].detach().numpy())
    plt.ylabel("Variable U")
    plt.subplot(515)
    plt.plot(hidden_u_hist[:, 0].detach().numpy(), hidden_v_hist[:, 0].detach().numpy(), marker="o")
    plt.xlabel("Variable U (Real part)")
    plt.ylabel("Variable V (Imaginary part)")
    plt.savefig("single_rf_output.jpg")
    plt.close()

def get_threshold_reset_counts(input_signal, last_reset_voltage, off_threshold, on_threshold, pulse, num_threshold_reset):
    if pulse == 1:
        bound = on_threshold
    else:
        bound = off_threshold
    
    num_threshold_reset += 1
    last_reset_voltage += bound

    if input_signal - last_reset_voltage > on_threshold:
        pulse = 1
        num_threshold_reset, last_reset_voltage = get_threshold_reset_counts(input_signal, last_reset_voltage, off_threshold, on_threshold, pulse, num_threshold_reset)
    elif input_signal - last_reset_voltage < off_threshold:
        pulse = -1
        num_threshold_reset, last_reset_voltage = get_threshold_reset_counts(input_signal, last_reset_voltage, off_threshold, on_threshold, pulse, num_threshold_reset)

    return num_threshold_reset, last_reset_voltage

def generate_event_stream(filepath, filename):
    tolerance_window = 1.5
    threshold = 0.2
    bin_width = 1
    on_threshold = threshold
    off_threshold = -threshold

    complete_path = os.path.join(filepath, filename)
    raw_data = loadmat(complete_path)

    signal = raw_data["data"]       # Shape of signal is (1, 1440000)
    spike_time = raw_data["spike_times"][0, 0]     # Shape of spike_times is (1, 3410)
    sampling_interval = raw_data["samplingInterval"][0, 0]
    
    length_of_signal = signal.shape[1]
    
    # Filter the input signal
    sampling_rate = 1 / (sampling_interval*1e-3)
    order = 2
    rp = 0.1
    rs = 40
    wn = [300, 5000]
    normalised_wn = [(2*w) / (sampling_rate) for w in wn]
    b, a = ellip(order, rp, rs, normalised_wn, btype="bandpass")
    filtered_signal = lfilter(b, a, signal) # Shape of signal is (1, 1440000)

    # Generate PCM
    n_ch = 1
    last_reset_voltage = 0
    event_stream = []
    fixed_delay = 10e-6 # in us

    ################################################ 
    # 
    # Update this for loop with your method.
    # The method shown here is PCM
    #
    ################################################ 
    for i in range(0, length_of_signal-bin_width+1, bin_width):
        event_queue = []
        signal_considered = filtered_signal[:, i:i+bin_width]
        num_on_pulses, num_off_pulses = [], []
        for j in range(signal_considered.shape[1]):
            num_threshold_reset = 0
            if signal_considered[0, j] - last_reset_voltage> on_threshold:
                pulse = 1
                num_threshold_reset, last_reset_voltage = get_threshold_reset_counts(signal_considered[0, j], last_reset_voltage, off_threshold, on_threshold, pulse, num_threshold_reset)
                num_on_pulses.append(num_threshold_reset)
                num_off_pulses.append(0)
            elif signal_considered[0, j] - last_reset_voltage < off_threshold:
                pulse = -1
                num_threshold_reset, last_reset_voltage = get_threshold_reset_counts(signal_considered[0, j], last_reset_voltage, off_threshold, on_threshold, pulse, num_threshold_reset)
                num_on_pulses.append(0)
                num_off_pulses.append(num_threshold_reset)
            else:
                num_on_pulses.append(0)
                num_off_pulses.append(0)
        if sum(num_on_pulses) + sum(num_off_pulses) > 0:
            event_queue.append([i, sum(num_on_pulses), sum(num_off_pulses)])
        if len(event_queue) > 0:
            event_stream.append(event_queue)

    return np.array(filtered_signal), np.array(spike_time), np.array(event_stream).squeeze(axis=1)

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)  # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = True  
    noise_level = "005"
    difficulty = "Easy1"
    filename = f"C_{difficulty}_noise{noise_level}"
    filepath = "./zby_Simulator_data/"

    ######### Plot the dynamic of a single Resonator
    # original_data, spikes, labels, sampling_freq, label_single = load_data(filepath, filename)
    # single_neuron_tests(spikes, original_data)

    ######### Load the Event Stream
    filtered_signal, spike_time, event_stream = generate_event_stream(filepath, filename) # Dim of the three variables: (1, 1440000), (1, 3526), (136369, 3)
    print(filtered_signal.shape, spike_time.shape, event_stream.shape)

    spikes = np.zeros_like(filtered_signal)
    on_times = np.where(event_stream[:, 1] > 0)[0]
    off_times = np.where(event_stream[:, 2] > 0)[0]

    print(on_times.shape, off_times.shape)

    spikes[:, event_stream[on_times, 0]] = event_stream[on_times, 1]
    spikes[:, event_stream[off_times, 0]] = -event_stream[off_times, 2]

    ######### Actual Training of the Resonator Network
    # train_ratio = 0.8
    # val_ratio = 0.1
    # test_ratio = 0.1

    # train_len = int(spikes.shape[1]*train_ratio)
    # val_len = int(spikes.shape[1]*val_ratio)
    # test_len = int(spikes.shape[1]*test_ratio)

    # print(spikes.shape, train_len, val_len, test_len)

    # train_dataset = SpikeDataset(spikes[:, :train_len], labels[:, :train_len])
    # val_dataset = SpikeDataset(spikes[:, train_len:train_len + val_len], labels[:, train_len:train_len + val_len])
    # test_dataset = SpikeDataset(spikes[:, train_len + val_len:], labels[:, train_len + val_len:])
