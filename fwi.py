import gc
import torch
import numpy as np
import deepwave as dw


def generate_ricker_source_wavelet(cfg):
    """
    Generate a Ricker source wavelet

    Args:
        cfg: configuration
    Returns:
        source wavelet
    """
    return (
        dw.wavelets.ricker(
            cfg.params.peak_freq, cfg.params.nt, cfg.params.dt, 1 / cfg.params.peak_freq
        )
        .reshape(-1, 1, 1)
        .repeat(1, cfg.params.ns, cfg.params.num_sources_per_shot)
    )


def generate_data(source, cfg, x_s, x_r, model_true, device="cpu"):
    """
    Generate data from a source and a velocity model

    Args:
        source: source wavelet
        cfg: configuration
        x_s: source coordinates
        x_r: receiver coordinates
        model_true: velocity model
        device: device to use
    Returns:
        generated data
    """
    # create a propegator
    prop = dw.scalar.Propagator({"vp": model_true.to(device)}, cfg.params.dx)

    return prop(
        source.to(device),
        x_s.to(device),
        x_r.to(device),
        cfg.params.dt,
    )


def compute_gradient(
    model,
    x_s,
    x_r,
    data_true,
    source_wavelet,
    cfg,
    log_likelihood,
    log_prior=None,
    log_prior_scale=None,
    data_normalization=None,
    device="cpu",
):
    """
    Compute the gradient of the loss function

    Args:
        model: velocity model
        x_s: source coordinates
        x_r: receiver coordinates
        data_true: true data
        source_wavelet: source wavelet
        cfg: configuration
        log_likelihood: log likelihood function
        log_prior: log prior function
        log_prior_scale: log prior scale
        data_normalization: data normalization
        device: device to use
    Returns:
        gradient of the loss function
    """
    model = (
        model.reshape(cfg.params.nz, cfg.params.nx) if len(model.shape) == 1 else model
    )
    # model.requires_grad = True

    running_loss = 0.0
    grad_loss = torch.zeros_like(model).to(device)
    for it in range(cfg.params.batch_size):
        prop = dw.scalar.Propagator({"vp": model.to(device)}, cfg.params.dx)
        batch_src_wvl = source_wavelet[
            :,
            it :: cfg.params.batch_size,
        ].to(device)
        batch_data_true = data_true[:, it :: cfg.params.batch_size].detach().to(device)
        batch_x_s = x_s[it :: cfg.params.batch_size].to(device)
        batch_x_r = x_r[it :: cfg.params.batch_size].to(device)
  
        data_pred = prop(batch_src_wvl, batch_x_s, batch_x_r, cfg.params.dt).to(device)

        if data_normalization is not None:
            batch_data_true = data_normalization(batch_data_true)
            data_pred = data_normalization(data_pred)

        loss = log_likelihood(data_pred, batch_data_true)
        running_loss += loss.item()
        grad_loss += torch.autograd.grad(loss, model)[-1]

    if log_prior is not None and log_prior_scale is not None:
        loss_reg = log_prior_scale * log_prior(model)
        grad_loss_reg = torch.autograd.grad(loss_reg, model)[-1]
        running_loss += loss_reg.item()
        grad_loss += grad_loss_reg

    gc.collect()
    torch.cuda.empty_cache()

    return running_loss, grad_loss.detach().cpu().numpy()


def compute_gradient_per_batch(model, grad_func):
    """
    Compute the gradient of the loss function per batch

    Args:
        model: velocity model
        grad_func: gradient function
    Returns:
        gradient of the loss function
    """
    log_p = 0.0
    fwi_grad = np.zeros_like(model.detach().cpu().numpy())
    for i, m in enumerate(model):
        m.requires_grad_(True)
        loss, grad_m = grad_func(m)
        fwi_grad[i] = grad_m.ravel()
        log_p += loss
    log_p /= len(model)
    return log_p, fwi_grad


def compute_max_gradient(grad):
    """
    Compute the maximum gradient

    Args:
        grad: gradient
    Returns:
        maximum gradient
    """
    return np.max(np.abs(grad))


def compute_max_gradient_per_batch(grad):
    """
    Compute the maximum gradient per batch

    Args:
        grad: gradient
    Returns:
        maximum gradient
    """
    assert len(grad.shape) == 2
    return np.max(np.abs(grad), axis=1, keepdims=True)
