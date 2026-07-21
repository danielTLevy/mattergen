# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for the wrapped (torus) importance-sampling log-prob fix.

The importance log-ratio for a wrapped field must be INVARIANT to shifting the
realized sample (or the mean) by an integer number of periods, because those are
the same physical point on the torus. The pre-fix unwrapped code violated this
by exactly n*boundary*delta_sum/std**2; the minimum-image fix restores it.
"""
import math

import torch

from mattergen.diffusion.sampling.classifier_free_guidance import _gaussian_logp_per_row


def _log_ratio(sample, mean_u, mean_g, std, boundary):
    return (
        _gaussian_logp_per_row(sample, mean_u, std, boundary=boundary)
        - _gaussian_logp_per_row(sample, mean_g, std, boundary=boundary)
    )


def test_wrapped_logratio_invariant_to_integer_period_shifts():
    torch.manual_seed(0)
    B, D, L = 8, 3, 1.0
    mean_u = torch.randn(B, D)
    delta = 0.3 * torch.randn(B, D)          # guided != uncond, so the bias term is nonzero
    mean_g = mean_u + delta
    std = torch.rand(B, 1) * 0.2 + 0.05
    sample = torch.remainder(mean_g + std * torch.randn(B, D), L)  # wrapped as the sampler stores it

    base = _log_ratio(sample, mean_u, mean_g, std, boundary=L)
    for n in (-3, -1, 2, 5):
        shifted_sample = _log_ratio(sample + n * L, mean_u, mean_g, std, boundary=L)
        assert torch.allclose(shifted_sample, base, atol=1e-4), f"sample shift n={n}"
        shifted_mean = _log_ratio(sample, mean_u + n * L, mean_g + n * L, std, boundary=L)
        assert torch.allclose(shifted_mean, base, atol=1e-4), f"mean shift n={n}"


def test_unwrapped_code_exhibits_the_bug():
    # Regression witness: WITHOUT the wrap (boundary=None), a period shift changes the
    # ratio -> demonstrates the original bug this fix removes.
    torch.manual_seed(0)
    B, D, L = 8, 3, 1.0
    mean_u = torch.randn(B, D)
    mean_g = mean_u + 0.3 * torch.randn(B, D)
    std = torch.rand(B, 1) * 0.2 + 0.05
    sample = torch.remainder(mean_g + std * torch.randn(B, D), L)
    base = _log_ratio(sample, mean_u, mean_g, std, boundary=None)
    shifted = _log_ratio(sample + L, mean_u, mean_g, std, boundary=None)
    assert not torch.allclose(shifted, base, atol=1e-3)


def test_boundary_none_matches_plain_gaussian():
    # Regression guard for non-wrapped fields (e.g. cell): boundary=None is the plain formula.
    torch.manual_seed(1)
    B, D = 5, 4
    sample, mean = torch.randn(B, D), torch.randn(B, D)
    std = torch.rand(B, 1) + 0.1
    got = _gaussian_logp_per_row(sample, mean, std, boundary=None)
    diff = (sample - mean) / std
    expected = (
        -0.5 * (diff * diff).sum(dim=1)
        - torch.log(std).sum(dim=1)
        - 0.5 * math.log(2.0 * math.pi) * D
    )
    assert torch.allclose(got, expected, atol=1e-6)
