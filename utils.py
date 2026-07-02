import numpy as np
from scipy.io import loadmat
from scipy.signal import ellip, lfilter

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

def generate_event_stream_dm(input_signal, on_threshold, off_threshold, bin_width=1):
    last_reset_voltage = 0
    length_of_signal = input_signal.shape[0]

    event_stream = []
    for i in range(0, length_of_signal-bin_width+1, bin_width):
        event_queue = []
        signal_considered = input_signal[i:i+bin_width]
        num_on_pulses, num_off_pulses = [], []

        for j in range(signal_considered.shape[0]):
            num_threshold_reset = 0
            if signal_considered[j] - last_reset_voltage > on_threshold:
                pulse = 1
                num_threshold_reset, last_reset_voltage = get_threshold_reset_counts(signal_considered[j], last_reset_voltage, off_threshold, on_threshold, pulse, num_threshold_reset)
                num_on_pulses.append(num_threshold_reset)
                num_off_pulses.append(0)
            elif signal_considered[j] - last_reset_voltage < off_threshold:
                pulse = -1
                num_threshold_reset, last_reset_voltage = get_threshold_reset_counts(signal_considered[j], last_reset_voltage, off_threshold, on_threshold, pulse, num_threshold_reset)
                num_on_pulses.append(0)
                num_off_pulses.append(num_threshold_reset)
            else:
                num_on_pulses.append(0)
                num_off_pulses.append(0)
        if sum(num_on_pulses) + sum(num_off_pulses) > 0:
            event_queue.append([i, 1, 1, sum(num_on_pulses), sum(num_off_pulses)])
        if len(event_queue) > 0:
            event_stream.append(event_queue)

    return np.array(event_stream).squeeze(axis=1)

def generate_event_stream_lif(filtered_signal, time_step, uth, lif_tau, if_reconstruct=False):
    spike_lif, time_lif, u_lif = lif_neuron(filtered_signal.T, time_step, uth, lif_tau)

    on_counts = np.zeros(filtered_signal.T.size)
    off_counts = np.zeros(filtered_signal.T.size)

    on_time = np.where(spike_lif > 0)
    off_time = np.where(spike_lif < 0)

    on_counts[np.array(on_time, dtype=int)] = 1
    off_counts[np.array(off_time, dtype=int)] = 1

    spike_train = on_counts - off_counts

    if if_reconstruct:
        return spike_train.T, spike_lif
    else:
        return spike_train.T

def leaky_integrate_neuron(U, time_step=1e-3, I=0, R=5, Urest=0, tau=5e-3):
    # tau = R*C
    U = (U) + (time_step/tau)*(-(U) + I*R) - Urest
    return U

def lif_neuron(filtered_signal, time_step=1e-3, uth=0.8, lif_tau=5e-3):
    U_trace = []
    spike_rec = []
    u = 0
    urest = 0
    time_lif = np.linspace(0, filtered_signal.shape[0], filtered_signal.shape[0], dtype=float)
    for step in range(filtered_signal.shape[0]):
        U_trace.append(u)
        u = leaky_integrate_neuron(u, time_step=time_step, I=filtered_signal[step], Urest=urest, tau=lif_tau)  # solve next step of U
        if u >= uth:
            urest = uth
            spike_rec.append(float(1))
        elif u<= -uth:
            urest = -uth
            spike_rec.append(float(-1))
        else:
            urest = 0
            spike_rec.append(float(0))

    return np.array(spike_rec), time_lif, U_trace

def load_dataset_intracortical(filepath: str, filename: str):
    complete_path = filepath + filename

    raw_data = loadmat(complete_path)
    # print(raw_data.keys())
    # print(raw_data["spike_class"].shape, raw_data["spike_class"]) # spike_class[0, 0] gives the spike class, and spike_times[0, 0] give the location for when that spike class occurs
    # print(raw_data["OVERLAP_DATA"].shape)
    # print(raw_data["data"].shape)
    # print(raw_data["startData"].shape)

    signal = raw_data["data"].squeeze() # shape (seq_len)
    spike_class_label = raw_data["spike_class"].squeeze()[0].squeeze()    # shape (num_of_spikes)
    spike_times = np.array(raw_data["spike_times"][0, 0].squeeze()) # shape (num_of_spikes)
    sampling_interval = raw_data["samplingInterval"][0, 0] * 1e-3
    sampling_rate = 1 / (sampling_interval) # 24kHz
    spike_pulse_1ms_idx_length = int(1e-3 / sampling_interval)

    spike_classes = np.unique(spike_class_label) # label is (1, 2, 3)

    order = 2
    rp = 0.1
    rs = 40
    wn = [300, 5000]
    normalised_wn = [(2*w) / (sampling_rate) for w in wn]
    b, a = ellip(order, rp, rs, normalised_wn, btype="bandpass")
    filtered_signal = lfilter(b, a, signal)

    return signal, spike_class_label, spike_times, sampling_interval, sampling_rate, spike_pulse_1ms_idx_length, spike_classes, filtered_signal
