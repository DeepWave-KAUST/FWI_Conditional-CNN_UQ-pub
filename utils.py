import numpy as np
import torch

from pylops import LinearOperator
from torchaudio.functional import biquad
from scipy.signal import butter, sosfiltfilt


def snr(x, x_est):
    """
    Compute the signal-to-noise ratio (SNR) in dB.

    Args:
        x: original signal
        x_est: estimated signal
    Returns:
        SNR in dB
    """
    return 10.0 * np.log10(np.linalg.norm(x) / np.linalg.norm(x - x_est))


def mask(velocity, water_velocity=1.5, device="cpu"):
    """
    Create a mask for the velocity model.

    Args:
        velocity: velocity model
        water_velocity: water velocity
    Returns:
        Mask
    """
    msk = torch.zeros_like(velocity)
    msk[velocity >= water_velocity] = 1
    return msk.to(device)


def constraint_model(velocity, water_velocity=1.5, low=-0.5, high=0.5, device="cpu"):
    """
    Create a mask for the velocity model.

    Args:
        velocity: velocity model
        water_velocity: water velocity
        low: lower bound
        high: upper bound
        device: device to use
    Returns:
        lower and upper bounds
    """
    msk = torch.ones_like(velocity)
    msk[velocity <= water_velocity] = 0.0

    vmin = (low * msk) + velocity
    vmin[velocity <= water_velocity] = water_velocity
    vmin[vmin <= water_velocity] = water_velocity

    vmax = (high * msk) + velocity
    vmax[velocity <= water_velocity] = water_velocity

    return vmin.to(device), vmax.to(device)


def highpass_filter(cutoff_freq, wavelet, cfg, device="cpu"):
    """
    Highpass filter the wavelet

    Args:
        cutoff_freq: cutoff frequency
        wavelet: wavelet
        cfg: configuration
        device: device to use
    Returns:
        Filtered wavelet
    """

    sos = butter(6, cutoff_freq, fs=1 / cfg.params.dt, output="sos")
    sos = [torch.tensor(sosi).to(wavelet.dtype).to(device) for sosi in sos]

    return biquad(biquad(biquad(wavelet, *sos[0]), *sos[1]), *sos[2]).to(device)


def highpass_filter_scipy(cutoff_freq, wavelet, cfg, device="cpu"):
    """
    Highpass filter the wavelet using scipy function

    Args:
        cutoff_freq: cutoff frequency
        wavelet: wavelet
        cfg: configuration
        device: device to use
    Returns:
        Filtered wavelet
    """
    wavelet = wavelet.detach().cpu().numpy()
    sos = butter(
        4,
        cutoff_freq,
        "hp",
        fs=1 / cfg.params.dt,
        output="sos",
    )
    return torch.tensor(
        sosfiltfilt(sos, wavelet, axis=0).copy(), dtype=torch.float32
    ).to(device)


def add_white_noise(data, noise_level=0.05, filter=None):
    """
    Adding white Gaussian noise to the dataset

    Args:
        data: dataset
        noise_level: noise level
        filter: filter to apply to the noise
    Returns:
        Noisy dataset
    """
    noise = noise_level * torch.randn_like(data)
    if filter is not None:
        noise = filter(noise)
    noisy_data = data + noise
    return noisy_data


def load_velocity_model(cfg, device="cpu"):
    """
    Load a velocity model from a file.

    Args:
        cfg: configuration
        device: device to use
    Returns:
        Velocity model
    """
    model_file = f"{cfg.paths.path}/{cfg.files.velocity_model}"
    return torch.from_numpy(
        np.fromfile(model_file, np.float32).reshape(cfg.params.nz, cfg.params.nx)
    ).to(device)


def constrained_operator(val, min_value, max_value):
    """
    Constraining model within min and max value.

    Args:
        val: model
        min_value: lower bound
        max_value: upper bound
    Returns:
        Constrained model
    """
    for i in range(len(val)):
        while (val[i] > max_value[i]) or (val[i] < min_value[i]):
            if val[i] > max_value[i]:
                val[i] = max_value[i] - (val[i] - max_value[i])

            if val[i] < min_value[i]:
                val[i] = min_value[i] + (min_value[i] - val[i])
    return val


class _TorchOperator(torch.autograd.Function):
    """Wrapper class for PyLops operators into Torch functions"""

    @staticmethod
    def forward(ctx, x, forw, adj):
        ctx.forw = forw
        ctx.adj = adj

        # prepare input
        # bring x to cpu and numpy
        x = x.cpu().detach().numpy()

        # apply forward operator
        y = ctx.forw(x)

        # prepare output
        # move y to torch and device
        y = torch.from_numpy(y).cuda()

        return y

    @staticmethod
    def backward(ctx, y):
        # prepare input
        y = y.cpu().detach().numpy()

        # apply adjoint operator
        x = ctx.adj(y)

        # prepare output
        x = torch.from_numpy(x).cuda()
        return x, None, None, None, None


class TorchOperator(LinearOperator):
    def __init__(self, Op, batch=False):
        if not batch:
            self.matvec = Op.matvec
            self.rmatvec = Op.rmatvec
        else:
            self.matvec = lambda x: Op.matmat(x.T).T
            self.rmatvec = lambda x: Op.rmatmat(x.T).T
        self.Top = _TorchOperator.apply
        self.shape = Op.shape
        self.dtype = Op.dtype

    def apply(self, x):
        return self.Top(x, self.matvec, self.rmatvec)
