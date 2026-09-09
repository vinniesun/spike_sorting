import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from snntorch.surrogate import atan

from BRF.grad_functions import StepDoubleGaussianGrad

def tanh_deriv(x: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.tanh(x) ** 2

def sigmoid_deriv(x: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(x)
    return sig * (1 - sig)

@torch.jit.script
def gaussian(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    return (1 / (sigma * torch.sqrt(2 * torch.tensor(torch.pi)))) * torch.exp(
        -((x - mu) ** 2) / (2.0 * (sigma ** 2))
    )

def step_double_gaussian_deriv(x: torch.Tensor) -> torch.Tensor:
    p = 0.15
    scale = 6.
    len = 0.5

    sigma1 = len
    sigma2 = scale * len

    dfd = (1. + p) * gaussian(x, mu=0., sigma=sigma1) - 2. * p * gaussian(x, mu=0., sigma=sigma2)

    return dfd

def atan_deriv(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return 1 / (1 + torch.pow((torch.pi * x * alpha / 2), 2))

if __name__ == "__main__":
    x = torch.arange(-10, 10, 0.1)

    y_tanh_deriv = tanh_deriv(x)
    y_sigmoid_deriv = sigmoid_deriv(x)
    y_step_double_gaussian = step_double_gaussian_deriv(x)
    y_atan_deriv = atan_deriv(x, alpha=0.8)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(x, y_tanh_deriv, label="tanh derivative", color="blue")
    ax.plot(x, y_sigmoid_deriv, label="sigmoid derivative", color="red")
    ax.plot(x, y_step_double_gaussian, label="step double gaussian derivative", color="green")
    ax.plot(x, y_atan_deriv, label="atan derivative", color="orange")
    ax.legend(loc="best", fontsize=14)
    ax.set_xlim(x.min(), x.max())

    plt.tight_layout()
    plt.savefig("surrogate_gradients_derivatives.png", dpi=300)
    plt.close()