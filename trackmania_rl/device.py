"""
Helpers to select and configure the torch device used by the learner.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch


class NoOpGradScaler:
    """
    Minimal GradScaler-compatible object for non-AMP device paths.
    """

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None


def resolve_torch_device(device_preference: str = "auto") -> torch.device:
    preference = device_preference.lower()
    if preference in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if preference == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")

    if preference == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported torch device preference: {device_preference}")


def configure_torch_runtime(device: torch.device):
    if device.type == "mps":
        # Official PyTorch fallback knob for operators that do not have an MPS kernel yet.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def make_autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(device: torch.device):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return NoOpGradScaler()


def state_dict_to_device(state_dict, device: torch.device):
    return {
        key: value.detach().to(device) if torch.is_tensor(value) else value
        for key, value in state_dict.items()
    }


def state_dict_to_cpu(state_dict):
    return state_dict_to_device(state_dict, torch.device("cpu"))
