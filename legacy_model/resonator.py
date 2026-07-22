import torch
import torch.nn as nn
import numpy as np
import snntorch as snn
from snntorch.surrogate import atan

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

##### I'll update this later after I've finished all the feature and tests I want to do with the original resonator
class BHResonator(nn.Module):
    instances = []
    """Balanced Harmonic Resonator (BHRF in the paper: Balanced Resonate-and-Fire Neurons)"""
    """mode can be either "homogeneous" of "heterogeneous".

        in homogeneous mode, the resonator only have a single b and omega parameter shared between all of the neurons
        in heterogenous mode, the resonator have unique b and omega for all of the neurons.

    :return: _description_
    :rtype: _type_
    """

    reset_dict = {
        "subtract": 0,
        "zero": 1,
        "none": 2,
    }

    def __init__(self, 
                 b=-1.0, 
                 omega=10.0, 
                 threshold=1.0,
                 learn_b=False,
                 learn_omega=False,
                 mode="homogeneous",
                 as_filter=False,
                 filter_dim=0,
                 spike_grad=None,
                 surrogate_disable=False,
                 init_hidden=False,
                 learn_threshold=False,
                 reset_mechanism="subtract",
                 output=False,
                 graded_spikes_factor=1.0,
                 learn_graded_spikes_factor=False,):
        super().__init__()

        Resonator.instances.append(self)

        if surrogate_disable:
            self.spike_grad = self._surrogate_bypass
        elif spike_grad == None:
            self.spike_grad = atan()
        else:
            self.spike_grad = spike_grad

        self.init_hidden = init_hidden
        self.output = output
        self.surrogate_disable = surrogate_disable

        self._snn_register_buffer(
            threshold=threshold,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
        )

        self._resonator_register_buffer(b, learn_b, omega, learn_omega)

        self.ref_period_constant = 0.9

        self.mode = mode
        self.as_filter = as_filter
        self.filter_dim = filter_dim

        self._reset_mechanism = reset_mechanism

        # current like variable
        self.x = torch.zeros((1))
        # voltage like variable
        self.y = torch.zeros((1))
        # print(f"Using the following parameters:\n b: {self.b}, omega: {self.omega}, threshold: {self.threshold}")

    def forward(self, input_, dt):
        if self.as_filter and self.x.shape != self.omega.shape:
            if self.filter_dim == 0:
                raise ValueError("Resonator is being used as a filter bank (as_filter=True), then the \
                                 output dimension must be set as well!")
            else:
                self.x = torch.zeros(self.filter_dim, device=input_.device)
                self.y = torch.zeros(self.filter_dim, device=input_.device)
        else:
            if not self.x.shape == input_.shape:
                self.x = torch.zeros_like(input_, device=input_.device)
            if not self.y.shape == input_.shape:
                self.y = torch.zeros_like(input_, device=input_.device)

        if self.mode == "heterogeneous" and not self.b.shape == input_.shape:
            # In heterogeneous mode, we don't care about the batch size.
            if self.learn_b:
                self.b = nn.Parameter(torch.ones(input_.shape[1:], dtype=torch.float32, device=input_.device)*self.b)
            else:
                self.b = torch.ones(input_.shape[1:], dtype=torch.float32, device=input_.device)*self.b
        if self.mode == "heterogeneous" and not self.omega.shape == input_.shape:
            if self.learn_omega:
                self.omega = nn.Parameter(torch.ones(input_.shape[1:], dtype=torch.float32, device=input_.device)*self.omega)
            else:
                self.omega = torch.ones(input_.shape[1:], dtype=torch.float32, device=input_.device)*self.omega

        # Reset if needed
        self.reset_ = self.reset_variable(self.x, self.y)

        # Update equation
        if self.reset_mechanism == "subtract":
            self.x, self.y = self.update_state(input_, self.x, self.y, dt)
            self.x = self.x - self.reset_ * self.threshold
            self.y = self.y - self.reset_ * self.threshold
        elif self.reset_mechanism == "zero":
            self.x = (1 - self.reset_) * self.x
            self.y = (1 - self.reset_) * self.y
            self.x, self.y = self.update_state(input_, self.x, self.y, dt)
        elif self.reset_mechanism == "none":
            self.x, self.y = self.update_state(input_, self.x, self.y, dt)

        # print(f"Current reset: {self.reset_}, updated x: {self.x}, updated y: {self.y}")

        # Check if spike is to be generated.
        spike = self.fire(self.y)

        return spike, self.x, self.y
    
    def update_state(self, input_, x, y, dt):
        # #Spike Generation
        # spike = (y > self.threshold).float()
        # #Reset if spike
        # x, y = self.reset(x, y)

        x_, y_ = x, y

        # Allows dt to broadcast across all dimension
        if len(dt.shape) != len(x_.shape):
            for _ in range(len(x_.shape) - len(dt.shape)):
                dt = dt.unsqueeze(dim=-1)

        ### Using Euler's Method
        if input_.dtype == torch.cfloat:
            x = (1 + self.b*dt)*x_ - self.omega*dt*y_ + input_.real
            y = (1 + self.b*dt)*y_ + self.omega*dt*x_ + input_.imag
        else:
            # print(1 + self.b*dt, self.omega*dt, input_)
            # print(f"begin: x_: {x_}, y_: {y_}")
            x = (1 + self.b*dt)*x_ - self.omega*dt*y_ + input_
            y = (1 + self.b*dt)*y_ + self.omega*dt*x_
            # print(f"after: x: {x}, y: {y}")

        return x, y
    
    def fire(self, y):
        """Generates spike if mem > threshold.
        Returns spk."""

        y_shift = y - self.threshold
        spk = self.spike_grad(y_shift)

        spk = spk * self.graded_spikes_factor

        return spk

    def reset_mem(self):
        self.x = torch.zeros_like(self.x, device=self.x.device)
        self.y = torch.zeros_like(self.y, device=self.x.device)

        return self.x, self.y

    def reset_variable(self, x, y):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        # print(y.device, y.shape, self.threshold.device)
        y_shift = y - self.threshold
        reset = self.spike_grad(y_shift).clone().detach()

        return reset
    
    def _snn_register_buffer(
        self,
        threshold,
        learn_threshold,
        reset_mechanism,
        graded_spikes_factor,
        learn_graded_spikes_factor,
    ):
        """Set variables as learnable parameters else register them in the
        buffer."""

        self._threshold_buffer(threshold, learn_threshold)
        self._graded_spikes_buffer(
            graded_spikes_factor, learn_graded_spikes_factor
        )

        # reset buffer
        try:
            # if reset_mechanism_val is loaded from .pt, override
            # reset_mechanism
            if torch.is_tensor(self.reset_mechanism_val):
                self.reset_mechanism = list(Resonator.reset_dict)[
                    self.reset_mechanism_val
                ]
        except AttributeError:
            # reset_mechanism_val has not yet been created, create it
            self._reset_mechanism_buffer(reset_mechanism)

    def _resonator_register_buffer(self, b, learn_b, omega, learn_omega):
        self._b_buffer(b, learn_b)
        self._omega_buffer(omega, learn_omega)

    def _b_buffer(self, b, learn_b):
        if not isinstance(b, torch.Tensor):
            b = torch.as_tensor(b, dtype=torch.float32)

        self.learn_b = learn_b
        if learn_b:
            self.b = nn.Parameter(b)
        else:
            self.register_buffer("b", b)

    def _omega_buffer(self, omega, learn_omega):
        if not isinstance(omega, torch.Tensor):
            omega = torch.as_tensor(omega, dtype=torch.float32)
        
        self.learn_omega = learn_omega
        if learn_omega:
            self.omega = nn.Parameter(omega)
        else:
            self.register_buffer("omega", omega)

    def _graded_spikes_buffer(
        self, graded_spikes_factor, learn_graded_spikes_factor
    ):
        if not isinstance(graded_spikes_factor, torch.Tensor):
            graded_spikes_factor = torch.as_tensor(graded_spikes_factor)
        if learn_graded_spikes_factor:
            self.graded_spikes_factor = nn.Parameter(graded_spikes_factor)
        else:
            self.register_buffer("graded_spikes_factor", graded_spikes_factor)

    def _threshold_buffer(self, threshold, learn_threshold):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.as_tensor(threshold)
        if learn_threshold:
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer("threshold", threshold)

    def _reset_mechanism_buffer(self, reset_mechanism):
        """Assign mapping to each reset mechanism state.
        Must be of type tensor to store in register buffer. See reset_dict
        for mapping."""
        reset_mechanism_val = torch.as_tensor(
            Resonator.reset_dict[reset_mechanism]
        )
        self.register_buffer("reset_mechanism_val", reset_mechanism_val)

    def _V_register_buffer(self, V, learn_V):
        if not isinstance(V, torch.Tensor):
            V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer("V", V)

    @property
    def reset_mechanism(self):
        """If reset_mechanism is modified, reset_mechanism_val is triggered
        to update.
        0: subtract, 1: zero, 2: none."""
        return self._reset_mechanism

    @reset_mechanism.setter
    def reset_mechanism(self, new_reset_mechanism):
        self._reset_cases(new_reset_mechanism)
        self.reset_mechanism_val = torch.as_tensor(
            Resonator.reset_dict[new_reset_mechanism]
        )
        self._reset_mechanism = new_reset_mechanism

    @classmethod
    def init(cls):
        """Removes all items from :mod:`snntorch.SpikingNeuron.instances`
        when called."""
        cls.instances = []

    @staticmethod
    def detach(*args):
        """Used to detach input arguments from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are global variables."""
        for state in args:
            state.detach_()

    @staticmethod
    def zeros(*args):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are global variables."""
        for state in args:
            state = torch.zeros_like(state)

    @staticmethod
    def _surrogate_bypass(input_):
        return (input_ > 0).float()

class Resonator(nn.Module):
    instances = []
    """Each :mod:`snntorch.SpikingNeuron` neuron
    (e.g., :mod:`snntorch.Synaptic`) will populate the
    :mod:`snntorch.SpikingNeuron.instances` list with a new entry.
    The list is used to initialize and clear neuron states when the
    argument `init_hidden=True`."""
    """mode can be either "homogeneous" of "heterogeneous".

        in homogeneous mode, the resonator only have a single b and omega parameter shared between all of the neurons
        in heterogenous mode, the resonator have unique b and omega for all of the neurons.

    :return: _description_
    :rtype: _type_
    """

    reset_dict = {
        "subtract": 0,
        "zero": 1,
        "none": 2,
    }

    def __init__(self, 
                 b=-1.0, 
                 omega=10.0, 
                 threshold=1.0,
                 learn_b=False,
                 learn_omega=False,
                 mode="homogeneous",
                 as_filter=False,
                 filter_dim=0,
                 spike_grad=None,
                 surrogate_disable=False,
                 init_hidden=False,
                 learn_threshold=False,
                 reset_mechanism="subtract",
                 output=False,
                 graded_spikes_factor=1.0,
                 learn_graded_spikes_factor=False,):
        super().__init__()

        Resonator.instances.append(self)

        if surrogate_disable:
            self.spike_grad = self._surrogate_bypass
        elif spike_grad == None:
            self.spike_grad = atan()
        else:
            self.spike_grad = spike_grad

        self.init_hidden = init_hidden
        self.output = output
        self.surrogate_disable = surrogate_disable

        self._snn_register_buffer(
            threshold=threshold,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
        )

        self._resonator_register_buffer(b, learn_b, omega, learn_omega)

        self.mode = mode
        self.as_filter = as_filter
        self.filter_dim = filter_dim

        self._reset_mechanism = reset_mechanism

        # current like variable
        self.x = torch.zeros((1))
        # voltage like variable
        self.y = torch.zeros((1))
        # print(f"Using the following parameters:\n b: {self.b}, omega: {self.omega}, threshold: {self.threshold}")

    def forward(self, input_, dt):
        if self.as_filter and self.x.shape != self.omega.shape:
            if self.filter_dim == 0:
                raise ValueError("Resonator is being used as a filter bank (as_filter=True), then the \
                                 output dimension must be set as well!")
            else:
                self.x = torch.zeros(self.filter_dim, device=input_.device)
                self.y = torch.zeros(self.filter_dim, device=input_.device)
        else:
            if not self.x.shape == input_.shape:
                self.x = torch.zeros_like(input_, device=input_.device)
            if not self.y.shape == input_.shape:
                self.y = torch.zeros_like(input_, device=input_.device)

        if self.mode == "heterogeneous" and not self.b.shape == input_.shape:
            # In heterogeneous mode, we don't care about the batch size.
            if self.learn_b:
                self.b = nn.Parameter(torch.ones(input_.shape[1:], dtype=torch.float32, device=input_.device)*self.b)
            else:
                self.b = torch.ones(input_.shape[1:], dtype=torch.float32, device=input_.device)*self.b
        if self.mode == "heterogeneous" and not self.omega.shape == input_.shape:
            if self.learn_omega:
                self.omega = nn.Parameter(torch.ones(input_.shape[1:], dtype=torch.float32, device=input_.device)*self.omega)
            else:
                self.omega = torch.ones(input_.shape[1:], dtype=torch.float32, device=input_.device)*self.omega

        # Reset if needed
        self.reset_ = self.reset_variable(self.x, self.y)

        # Update equation
        if self.reset_mechanism == "subtract":
            self.x, self.y = self.update_state(input_, self.x, self.y, dt)
            self.x = self.x - self.reset_ * self.threshold
            self.y = self.y - self.reset_ * self.threshold
        elif self.reset_mechanism == "zero":
            self.x = (1 - self.reset_) * self.x
            self.y = (1 - self.reset_) * self.y
            self.x, self.y = self.update_state(input_, self.x, self.y, dt)
        elif self.reset_mechanism == "none":
            self.x, self.y = self.update_state(input_, self.x, self.y, dt)

        # print(f"Current reset: {self.reset_}, updated x: {self.x}, updated y: {self.y}")

        # Check if spike is to be generated.
        spike = self.fire(self.y)

        return spike, self.x, self.y
    
    def update_state(self, input_, x, y, dt):
        # #Spike Generation
        # spike = (y > self.threshold).float()
        # #Reset if spike
        # x, y = self.reset(x, y)

        x_, y_ = x, y

        # Allows dt to broadcast across all dimension
        if len(dt.shape) != len(x_.shape):
            for _ in range(len(x_.shape) - len(dt.shape)):
                dt = dt.unsqueeze(dim=-1)

        ### Using Euler's Method
        if input_.dtype == torch.cfloat:
            x = (1 + self.b*dt)*x_ - self.omega*dt*y_ + input_.real
            y = (1 + self.b*dt)*y_ + self.omega*dt*x_ + input_.imag
        else:
            # print(1 + self.b*dt, self.omega*dt, input_)
            # print(f"begin: x_: {x_}, y_: {y_}")
            x = (1 + self.b*dt)*x_ - self.omega*dt*y_ + input_
            y = (1 + self.b*dt)*y_ + self.omega*dt*x_
            # print(f"after: x: {x}, y: {y}")

        return x, y
    
    def fire(self, y):
        """Generates spike if mem > threshold.
        Returns spk."""

        y_shift = y - self.threshold
        spk = self.spike_grad(y_shift)

        spk = spk * self.graded_spikes_factor

        return spk

    def reset_mem(self):
        self.x = torch.zeros_like(self.x, device=self.x.device)
        self.y = torch.zeros_like(self.y, device=self.x.device)

        return self.x, self.y

    def reset_variable(self, x, y):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        # print(y.device, y.shape, self.threshold.device)
        y_shift = y - self.threshold
        reset = self.spike_grad(y_shift).clone().detach()

        return reset
    
    def _snn_register_buffer(
        self,
        threshold,
        learn_threshold,
        reset_mechanism,
        graded_spikes_factor,
        learn_graded_spikes_factor,
    ):
        """Set variables as learnable parameters else register them in the
        buffer."""

        self._threshold_buffer(threshold, learn_threshold)
        self._graded_spikes_buffer(
            graded_spikes_factor, learn_graded_spikes_factor
        )

        # reset buffer
        try:
            # if reset_mechanism_val is loaded from .pt, override
            # reset_mechanism
            if torch.is_tensor(self.reset_mechanism_val):
                self.reset_mechanism = list(Resonator.reset_dict)[
                    self.reset_mechanism_val
                ]
        except AttributeError:
            # reset_mechanism_val has not yet been created, create it
            self._reset_mechanism_buffer(reset_mechanism)

    def _resonator_register_buffer(self, b, learn_b, omega, learn_omega):
        self._b_buffer(b, learn_b)
        self._omega_buffer(omega, learn_omega)

    def _b_buffer(self, b, learn_b):
        if not isinstance(b, torch.Tensor):
            b = torch.as_tensor(b, dtype=torch.float32)

        self.learn_b = learn_b
        if learn_b:
            self.b = nn.Parameter(b)
        else:
            self.register_buffer("b", b)

    def _omega_buffer(self, omega, learn_omega):
        if not isinstance(omega, torch.Tensor):
            omega = torch.as_tensor(omega, dtype=torch.float32)
        
        self.learn_omega = learn_omega
        if learn_omega:
            self.omega = nn.Parameter(omega)
        else:
            self.register_buffer("omega", omega)

    def _graded_spikes_buffer(
        self, graded_spikes_factor, learn_graded_spikes_factor
    ):
        if not isinstance(graded_spikes_factor, torch.Tensor):
            graded_spikes_factor = torch.as_tensor(graded_spikes_factor)
        if learn_graded_spikes_factor:
            self.graded_spikes_factor = nn.Parameter(graded_spikes_factor)
        else:
            self.register_buffer("graded_spikes_factor", graded_spikes_factor)

    def _threshold_buffer(self, threshold, learn_threshold):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.as_tensor(threshold)
        if learn_threshold:
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer("threshold", threshold)

    def _reset_mechanism_buffer(self, reset_mechanism):
        """Assign mapping to each reset mechanism state.
        Must be of type tensor to store in register buffer. See reset_dict
        for mapping."""
        reset_mechanism_val = torch.as_tensor(
            Resonator.reset_dict[reset_mechanism]
        )
        self.register_buffer("reset_mechanism_val", reset_mechanism_val)

    def _V_register_buffer(self, V, learn_V):
        if not isinstance(V, torch.Tensor):
            V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer("V", V)

    @property
    def reset_mechanism(self):
        """If reset_mechanism is modified, reset_mechanism_val is triggered
        to update.
        0: subtract, 1: zero, 2: none."""
        return self._reset_mechanism

    @reset_mechanism.setter
    def reset_mechanism(self, new_reset_mechanism):
        self._reset_cases(new_reset_mechanism)
        self.reset_mechanism_val = torch.as_tensor(
            Resonator.reset_dict[new_reset_mechanism]
        )
        self._reset_mechanism = new_reset_mechanism

    @classmethod
    def init(cls):
        """Removes all items from :mod:`snntorch.SpikingNeuron.instances`
        when called."""
        cls.instances = []

    @staticmethod
    def detach(*args):
        """Used to detach input arguments from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are global variables."""
        for state in args:
            state.detach_()

    @staticmethod
    def zeros(*args):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are global variables."""
        for state in args:
            state = torch.zeros_like(state)

    @staticmethod
    def _surrogate_bypass(input_):
        return (input_ > 0).float()

# if __name__ == "__main__":
#     resonator = Resonator()

#     # dummy_input = torch.randn(128, 10)
#     # dt = torch.randn(128, 10)*1e-3
#     # spk, x, y = resonator(dummy_input, dt)
#     # print(f"{spk.shape}, {x.shape}, {y.shape}")
#     # print(torch.sum(spk))

#     input_current = torch.zeros((3000), dtype=torch.float)
#     # input_current.real[::10] = 1
#     # input_current.imag[5::50] = 1
#     input_current[10] = 1
#     input_current[1010] = 1
#     # input_current.imag[10] = 1
#     # input_current.real[11] = 1
#     # input_current.real[12] = 1
#     # input_current.real[13] = 1
#     # input_current.real[14] = 1
#     # input_current.real[170] = 1

#     # input_current = torch.zeros((100), dtype=torch.float32)
#     # input_current[::15] = 1

#     # Set up formatting for the movie files
#     # Writer = animation.writers['html']
#     Writer = animation.writers['ffmpeg']
#     writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
#     fig, ax = plt.subplots()
#     artists = []
#     I_hist, V_hist, spike_hist = [], [], []
#     ax.set_xlabel("V")
#     ax.set_ylabel("I")
#     for curr in input_current:
#         spike, x, y = resonator(curr, 0.001)
        
#         # I_hist.append(resonator.x.item())
#         # V_hist.append(resonator.y.item())
#         I_hist.append(x.item())
#         V_hist.append(y.item())
#         spike_hist.append(spike.item())

#         container = ax.plot(I_hist, V_hist, color="b")
#         artists.append(container)
#     ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=100)
#     # ani.save('im.html', writer=writer)
#     ani.save('im.mp4', writer=writer)
#     plt.show()

#     # fig, ax = plt.subplots(2, 3)
#     # if input_current.dtype == torch.cfloat:
#     #     ax[0][0].plot(input_current.real, color="black", label="real")
#     #     ax[0][0].plot(input_current.imag, color="yellow", label="imag")
#     # else:
#     #     ax[0][0].plot(input_current, color="black", label="real")
#     # ax[0][0].set_ylabel("Input Current")
#     # ax[0][0].set_xlabel("t")
#     # ax[0][0].legend()
#     # ax[0][0].set_title("Input Current")
#     # ax[0][1].plot(V_hist, color='red')
#     # ax[0][1].set_ylabel("Voltage")
#     # ax[0][1].set_xlabel("t")
#     # ax[0][1].set_title("Voltage like object's Response (Imaginary)")
#     # ax[1][0].plot(I_hist, color="green")
#     # ax[1][0].set_ylabel("Current")
#     # ax[1][0].set_xlabel("t")
#     # ax[1][0].set_title("Current like object's Response (Real)")
#     # ax[1][1].plot(I_hist, V_hist, color="blue")
#     # ax[1][1].set_ylabel("Voltage")
#     # ax[1][1].set_xlabel("Current")
#     # ax[1][1].set_title("Imaginary vs Real")
#     # ax[0][2].plot(spike_hist, color="brown")
#     # ax[0][2].set_ylabel("Spike")
#     # ax[0][2].set_xlabel("t")
#     # ax[0][2].set_title("Output Spike Gen")
#     # plt.show()

