import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go

import snntorch as snn
import snntorch.functional as SF

import os
from scipy.io import loadmat
from scipy.signal import ellip, lfilter, butter, find_peaks
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from layers import *
from neurons import *
from models import *
from grad_functions import *

SEED = 1337
# START_IDX = 4200
# END_IDX = 4500

INPUT_REFRACTORY_LIMIT = 12

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data        # Each data is of the shape (instances, timestep, channel)
        self.labels = labels    # Each label is of the shape (instances, timestep, spike_or_not)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class BRFCell(RFCell):
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
        torch.nn.init.xavier_uniform_(self.linear.weight)

        self.event_polarity = None
        self.input_refractory_period = None

    def check_refractory_status(self, x, batch_size):
        x_ = torch.clone(x)
        for i in range(batch_size):
            if self.event_polarity[i] == float("inf"):
                if x[i] > 0:
                    self.event_polarity[i] = 1
                    self.input_refractory_period[i] = 0
                elif x[i] < 0:
                    self.event_polarity[i] = -1
                    self.input_refractory_period[i] = 0
                else:
                    self.input_refractory_period[i] = self.input_refractory_period[i] + 1
            elif self.event_polarity[i] == 1:
                if x[i] > 0 and self.input_refractory_period[i] <= INPUT_REFRACTORY_LIMIT:
                    x_[i] = 0
                    self.input_refractory_period[i] = self.input_refractory_period[i] + 1
                elif x[i] > 0 and self.input_refractory_period[i] > INPUT_REFRACTORY_LIMIT:
                    self.input_refractory_period[i] = 0
                elif x[i] == 0:
                    self.input_refractory_period[i] = self.input_refractory_period[i] + 1
                elif x[i] < 0:
                    self.event_polarity[i] = -1
                    self.input_refractory_period[i] = 0
            elif self.event_polarity[i] == -1:
                if x[i] < 0 and self.input_refractory_period[i] <= INPUT_REFRACTORY_LIMIT:
                    x_[i] = 0
                    self.input_refractory_period[i] = self.input_refractory_period[i] + 1
                elif x[i] < 0 and self.input_refractory_period[i] > INPUT_REFRACTORY_LIMIT:
                    self.input_refractory_period[i] = 0
                elif x[i] == 0:
                    self.input_refractory_period[i] = self.input_refractory_period[i] + 1
                elif x[i] > 0:
                    self.event_polarity[i] = 1
                    self.input_refractory_period[i] = 0
        return x_

    def reset_refractory(self, x):
        self.event_polarity = torch.ones(x.shape[0], device=x.device)*float("inf")
        self.input_refractory_period = torch.ones(x.shape[0], device=x.device)*float("inf")     

    def forward(
            self, 
            x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x is of the shape: (batch_size, seq_len, input_dim)

        batch_size, input_dim = x.shape

        # x = self.check_refractory_status(x, batch_size)

        in_sum = self.linear(x)
        z, u, v, q = state

        omega = self.omega * 2 * torch.pi
        omega = torch.abs(omega)

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

        return z, u, v, q

class Model(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int
        ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.brf = BRFCell(
            input_size=input_dim,
            layer_size=hidden_dim
        )
        self.fc1 = nn.Linear(hidden_dim, output_dim, bias=False)
        # torch.nn.init.uniform_(self.fc1.weight, 0.0, 1.0)
        # torch.nn.init.uniform_(self.fc1.bias, 0.0, 1.0)
        self.lif1 = snn.Leaky(beta=0.99, threshold=0.2, spike_grad=step_double_gaussian(), reset_mechanism="subtract")
        self.relu = nn.ReLU()

    def forward(self, x, hidden_states=None):
        mem1 = self.lif1.reset_mem()
        self.brf.reset_refractory(x)

        bs, seq_len, _ = x.shape

        if hidden_states is None:
            hidden_z = torch.zeros((bs, self.hidden_dim), device=self.fc1.weight.device)
            hidden_u = torch.zeros_like(hidden_z, device=self.fc1.weight.device)
            hidden_v = torch.zeros_like(hidden_z, device=self.fc1.weight.device)
            hidden_q = torch.zeros_like(hidden_z, device=self.fc1.weight.device)
        else:
            hidden_z, hidden_u, hidden_v, hidden_q = hidden_states

        spk_hist, mem_hist = [], []
        hidden_z_hist, hidden_u_hist, hidden_v_hist, hidden_q_hist = [], [], [], []
        for i in range(seq_len):
            curr_input = x[:, i, :]

            hidden_z, hidden_u, hidden_v, hidden_q = self.brf(curr_input, (hidden_z, hidden_u, hidden_v, hidden_q))

            curr_1 = self.fc1(hidden_z)
            spk1, mem1 = self.lif1(curr_1, mem1)

            spk_hist.append(spk1)
            mem_hist.append(mem1)
            
            hidden_z_hist.append(hidden_z)
            hidden_u_hist.append(hidden_u)
            hidden_v_hist.append(hidden_v)
            hidden_q_hist.append(hidden_q)

        # return torch.stack(spk_hist, dim=1), torch.stack(mem_hist, dim=1), (hidden_z, hidden_u, hidden_v, hidden_q)
        return torch.stack(spk_hist, dim=1), torch.stack(mem_hist, dim=1), (hidden_z_hist, hidden_u_hist, hidden_v_hist, hidden_q_hist)

def plot_input_labels(spike_train, label, og_spike_time, save_filename):
    x = torch.arange(label.shape[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, 
            y=spike_train+5, 
            name="Spike Train", 
            line=dict(color="red")
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, 
            y=label+2, 
            name="Label", 
            line=dict(color="blue")
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, 
            y=og_spike_time, 
            name="OG Spike Time", 
            line=dict(color="green")
        )
    )
    pio.write_html(fig, file=save_filename+".html", auto_open=False)

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

def calc_accuracy(pred, label):
    # pred's shape is: (batch_size, seq_len, 1)
    # label's shape is: (batch_size, seq_len, 1)
    idx = pred.sum(dim=1)

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 20

    np.random.seed(SEED)
    torch.manual_seed(SEED)  # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = True
    
    noise_level = "005" # 005, 01, 015, 02
    difficulty = "Easy1"    # Easy1, Easy2, Difficult1, Difficult2
    filename = f"C_{difficulty}_noise{noise_level}"
    filepath = "./Simulator_Data/"

    filtered_signal, spike_time, event_stream = generate_event_stream(filepath, filename) # Dim of the three variables: (1, 1440000), (1, 3526), (136369, 3)
    length_of_signal = filtered_signal.shape[1]

    # Generate Label
    label = np.zeros_like(filtered_signal)
    og_spike_times = np.zeros_like(filtered_signal)
    spike_time_window = 3
    spike_time_delay = 24
    shifted_spike_time = spike_time + spike_time_delay
    for time in shifted_spike_time[0]:
        # label[:, time-spike_time_window+spike_time_delay:time+spike_time_window+spike_time_delay+1] = 1
        label[:, time-spike_time_window:time+spike_time_window+1] = 1
    for time in spike_time[0]: 
        og_spike_times[:, time] = 1

    # Generate Spike Train
    spike_train = np.zeros_like(filtered_signal)
    for i in range(event_stream.shape[0]):
        spike_train[:, event_stream[i, 0]] = event_stream[i, 1] - event_stream[i, 2]

    spike_train_plot = np.where(spike_train > 0, 1, spike_train)
    spike_train_plot = np.where(spike_train < 0, -1, spike_train)

    plot_input_labels(spike_train_plot[0], label[0], og_spike_times[0], save_filename="spike_train_label")

    # Split into Train/Val/Test segments
    spike_samples, non_spike_samples = [], []
    spike_samples_label, non_spike_samples_label = [], []
    for i, time in enumerate(spike_time[0]):
        # So what we want to do here is to extract snippets of the signals,
        # such that each portion have the same length, and at each time step, 
        # we will grab a spike portion and non-spike portion.
        start_index = time
        # end_index = time + spike_time_delay + spike_time_window + 1
        end_index = time + 40
        spike_samples.append(spike_train[:, start_index:end_index].transpose((1, 0)))
        spike_samples_label.append(label[:, start_index:end_index].transpose((1, 0)))
        # spike_samples_label.append(np.ones(1))

        non_spike_start_index = end_index
        # non_spike_end_index = end_index + spike_time_delay + spike_time_window + 1
        non_spike_end_index = end_index + 40
        non_spike_samples.append(spike_train[:, non_spike_start_index:non_spike_end_index].transpose((1, 0)))
        non_spike_samples_label.append(label[:, non_spike_start_index:non_spike_end_index].transpose((1, 0)))
        # non_spike_samples_label.append(np.zeros(1))

    spike_samples = np.array(spike_samples)
    non_spike_samples = np.array(non_spike_samples)
    spike_samples_label = np.array(spike_samples_label)
    non_spike_samples_label = np.array(non_spike_samples_label)

    # print(spike_samples.shape, non_spike_samples.shape)
    # print(spike_samples_label.shape, non_spike_samples_label.shape)

    train_ratio = 0.8
    spike_train_samples, spike_test_samples, label_train_samples, label_test_samples = train_test_split(
        spike_samples, spike_samples_label, train_size=train_ratio
    )
    no_spike_train_samples, no_spike_test_samples, no_label_train_samples, no_label_test_samples = train_test_split(
        non_spike_samples, non_spike_samples_label, train_size=train_ratio
    )

    # print(spike_train_samples.shape, label_train_samples.shape, spike_test_samples.shape, label_test_samples.shape)
    # print(no_spike_train_samples.shape, no_label_train_samples.shape, no_spike_test_samples.shape, no_label_test_samples.shape)

    train_samples = np.concatenate((spike_train_samples, no_spike_train_samples), axis=0)
    train_labels = np.concatenate((label_train_samples, no_label_train_samples), axis=0)
    test_samples = np.concatenate((spike_test_samples, no_spike_test_samples), axis=0)
    test_labels = np.concatenate((label_test_samples, no_label_test_samples), axis=0)
    # print(train_samples.shape, train_labels.shape, test_samples.shape, test_labels.shape)

    train_dataset = MyDataset(train_samples[2810:2812], train_labels[:2])
    test_dataset = MyDataset(test_samples[2810:2812], test_labels[:2])
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=False)
    
    net = Model(
        input_dim=1,
        hidden_dim=16,
        output_dim=1
    )
    net.to(DEVICE)

    optimiser = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.05)
    criterion = nn.MSELoss()    # Try first to spike loss
    # criterion = SF.ce_count_loss()
    # criterion = nn.CrossEntropyLoss()

    loss_hist = []
    best_loss = float("inf")
    for epoch in tqdm(range(EPOCHS)):
        net.train()
        training_loss = 0
        for data, label in train_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            data = data.to(torch.float32)
            label = label.to(torch.float32)
            # label = label.to(torch.long)

            # if i == 0:
            #     spk_out, mem_out, hidden_states = net(data)
            # else:
            #     hidden_states = (hidden_z, hidden_u, hidden_v, hidden_q)
            #     spk_out, mem_out, hidden_states = net(data, hidden_states)
            spk_out, mem_out, hidden_states = net(data)
            # print(net.brf.omega)

            # (hidden_z, hidden_u, hidden_v, hidden_q) = hidden_states

            loss = torch.sqrt(criterion(spk_out, label))
            # print(spk_out.shape, label.shape)
            # loss = criterion(spk_out.permute(1, 0, 2), label.squeeze(dim=-1))
            # loss = torch.zeros(1).to(DEVICE)
            # for i in range(mem_out.shape[1]):
            #     loss += criterion(mem_out[:, i, :], label[:, i, :].squeeze(dim=-1))

            training_loss += loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            # hidden_z = hidden_z.detach()
            # hidden_u = hidden_u.detach()
            # hidden_v = hidden_v.detach()
            # hidden_q = hidden_q.detach()
            hidden_z_hist, hidden_u_hist, hidden_v_hist, hidden_q_hist = hidden_states
            hidden_z_hist = torch.stack(hidden_z_hist, dim=1)
            hidden_u_hist = torch.stack(hidden_u_hist, dim=1)
            hidden_v_hist = torch.stack(hidden_v_hist, dim=1)
            hidden_q_hist = torch.stack(hidden_q_hist, dim=1)

            fig, ax = plt.subplots(hidden_u_hist.shape[-1]+3, 1, figsize=(12, 40))
            ax[0].stem(data[0, :, :].cpu().detach().numpy(), label="Input Signal")
            ax[1].stem(label[0, :, :].cpu().detach().numpy(), label="Label")
            ax[2].plot(mem_out[0, :, 0].cpu().detach().numpy(), color="blue", label="LIF0's Membrane Potential")
            ax[2].stem(spk_out[0, :, 0].cpu().detach().numpy()*0.2, label="LIF1's Membrane Potential")
            for i in range(hidden_u_hist.shape[-1]):
                ax[i+3].plot(hidden_u_hist[0, :, i].cpu().detach().numpy(), color="red", label="Hidden U")
                ax[i+3].plot(hidden_v_hist[0, :, i].cpu().detach().numpy(), color="blue", label="Hidden V")
                # ax[i+3].plot(hidden_q_hist[1, :, i].cpu().detach().numpy(), color="green", label="Hidden Q")
                ax[i+3].stem(hidden_z_hist[0, :, i].cpu().detach().numpy()*5e-5, label="Hidden Z")
                plt.tight_layout()
            plt.legend(loc="best")
            plt.savefig(f"hidden_states_{epoch}.jpg")
            plt.close()
        
        loss_hist.append(training_loss)
        if training_loss < best_loss:
            best_loss = training_loss
            torch.save(net.state_dict(), "resonator_model.pth")
            print("Model Saved!")

        # net.eval()
        # output, labels = [], []
        # for data, label in test_loader:
        #     data = data.to(DEVICE)
        #     label = label.to(DEVICE)
        #     data = data.to(torch.float32)
        #     label = label.to(torch.float32)

        #     # if i == 0:
        #     #     spk_out, mem_out, hidden_states = net(data)
        #     # else:
        #     #     spk_out, mem_out, hidden_states = net(data, hidden_states)
        #     spk_out, mem_out, hidden_states = net(data)

        #     output.append(spk_out)
        #     labels.append(label)
        # output = torch.cat(output, dim=1)
        # labels = torch.cat(labels, dim=0)

        print("Current Loss: ", loss_hist[-1])
    
    fig = plt.figure()
    plt.plot(loss_hist)
    plt.savefig("loss_hist.jpg")
    plt.close()

    # net.load_state_dict(torch.load("resonator_model.pth", weights_only=True))
    # net.eval()
    # for i, (data, label) in enumerate(train_loader):
    #     data = data.to(DEVICE)
    #     label = label.to(DEVICE)
    #     data = data.to(torch.float32)
    #     label = label.to(torch.float32)

    #     # if i == 0:
    #     #     spk_out, mem_out, hidden_states = net(data)
    #     # else:
    #     #     spk_out, mem_out, hidden_states = net(data, hidden_states)
    #     spk_out, mem_out, hidden_states = net(data)

    #     spk_out = spk_out.squeeze(dim=-1)
    #     mem_out = mem_out.squeeze(dim=-1)
    #     labels = label.squeeze(dim=-1)

    #     if i == 0:
    #         fig = plt.figure()
    #         plt.plot(mem_out[50, :, 0].cpu().detach().numpy() + 5, label="Predict Spike Train")
    #         plt.plot(label[50, :, 0].cpu().detach().numpy() + 2, label="Label")
    #         plt.plot(data[50, :, 0].cpu().detach().numpy(), label="Input Signal")
    #         plt.legend(loc="best")
    #         plt.savefig("predicted_output.jpg")
    #         plt.close()
    #     else:
    #         break
