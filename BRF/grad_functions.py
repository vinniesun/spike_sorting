import torch
import math

################################################################
# Simple base functional
################################################################

@torch.jit.script
def step(x: torch.Tensor) -> torch.Tensor:
    #
    # x.gt(0.0).float()
    # is slightly faster (but less readable) than
    # torch.where(x > 0.0, 1.0, 0.0)
    #
    return x.gt(0.0).float()

@torch.jit.script
def exp_decay(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.abs(x))

@torch.jit.script
def gaussian(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    return (1 / (sigma * torch.sqrt(2 * torch.tensor(math.pi)))) * torch.exp(
        -((x - mu) ** 2) / (2.0 * (sigma ** 2))
    )


def gaussian_non_normalized(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    return torch.exp(-((x - mu) ** 2) / (2.0 * (sigma ** 2)))


def std_gaussian(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(
        -0.5 * (x ** 2)
    )


def linear_peak(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(1.0 - torch.abs(x))


def linear_peak_antiderivative(x: torch.Tensor) -> torch.Tensor:

    xa = torch.relu(1.0 - torch.abs(x))
    xa_sq = xa ** 2

    return 0.5 * torch.where(
        x < 0,
        xa_sq,
        2.0 - xa_sq
    )

@torch.jit.script
def DoubleGaussian(x: torch.Tensor) -> torch.Tensor:
    p = 0.15
    scale = 6.
    len = 0.5

    gamma = 0.5

    sigma1 = len
    sigma2 = scale * len
    return gamma * (1. + p) * gaussian(x, mu=0., sigma=sigma1) \
    - p * gaussian(x, mu=len, sigma=sigma2) - p * gaussian(x, mu=-len, sigma=sigma2)


def quantize_tensor(tensor: torch.Tensor, f: int) -> torch.Tensor:
    # Quantization formula: tensor_q = round(2^f * tensor) * 2^(-f)
    return torch.round(2**f * tensor) * 0.5**f


def spike_deletion(hidden_z: torch.Tensor, spike_del_p: float) -> torch.Tensor:
    return hidden_z.mul(spike_del_p < torch.rand_like(hidden_z))

################################################################
# Autograd function classes
################################################################

class StepGaussianGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        dfdx = gaussian(x)
        return grad_output * dfdx


class StepLinearGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        dfdx = torch.relu(1.0 - torch.abs(x))
        return grad_output * dfdx


class StepExpGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        dfdx = torch.exp(-torch.abs(x))
        return grad_output * dfdx

class ATan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha=6.0):
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            ctx.alpha
            / 2
            / (1 + (torch.pi / 2 * ctx.alpha * input_).pow_(2))
            * grad_input
        )
        return grad

class StepDoubleGaussianGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors

        p = 0.15
        scale = 6.
        len = 0.5

        sigma1 = len
        sigma2 = scale * len

        gamma = 0.5
        dfd = (1. + p) * gaussian(x, mu=0., sigma=sigma1) - 2. * p * gaussian(x, mu=0., sigma=sigma2)

        return grad_output * dfd * gamma

def step_double_gaussian():
    def inner(x):
        return StepDoubleGaussianGrad.apply(x)
    return inner

class StepMultiGaussianGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors

        p = 0.15
        scale = 6.
        len = 0.5

        sigma1 = len
        sigma2 = scale * len

        gamma = 0.5
        dfd = (1. + p) * gaussian(x, mu=0., sigma=sigma1) \
              - p * gaussian(x, mu=len, sigma=sigma2) - p * gaussian(x, mu=-len, sigma=sigma2)

        return grad_output * dfd * gamma


def FGI_DGaussian(x: torch.Tensor) -> torch.Tensor:

    x_detached = step(x).detach()

    p = 0.15
    scale = 6.
    len = 0.5

    sigma1 = len
    sigma2 = scale * len

    gamma = 0.5

    df = (1. + p) * gaussian(x, mu=0., sigma=sigma1) - 2. * p * gaussian(x, mu=0., sigma=sigma2)

    df_detached = df.detach()

    # detach of df prevents the gradients to flow through x of the gaussian function.
    dfd = gamma * df_detached * x

    dfd_detached = dfd.detach()

    return dfd - dfd_detached + x_detached
