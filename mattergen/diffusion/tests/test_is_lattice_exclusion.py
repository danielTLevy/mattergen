# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Structural tests for the lattice (`cell`) IS Gaussian ratio.

`cell` uses `LatticeAncestralSamplingPredictor`, which subclasses
`AncestralSamplingPredictor` (so it PASSES the isinstance dispatch in
`_denoise_one_step_with_logp_pair`), but it does not have a plain isotropic
Gaussian kernel:

  * its update draws symmetrized noise (`make_noise_symmetric_preserve_variance`),
    whose off-diagonal entries are duplicated -> a degenerate, rank-6 covariance
    in the 9-dim cell space; a naive per-component Gaussian sum would double-count
    every off-diagonal pair, and
  * its corrector applies a nonlinear `compute_lattice_polar_decomposition`.

The production ratio therefore uses six independent symmetric coordinates and
scores the corrector's pre-polar latent. These tests pin the dispatch and noise
structure that implementation relies on.
"""
import torch

from mattergen.common.diffusion.corruption import LatticeVPSDE, make_noise_symmetric_preserve_variance
from mattergen.diffusion.wrapped.wrapped_sde import WrappedVESDE, WrappedSDEMixin


def test_lattice_predicate_selects_only_lattice():
    lattice = LatticeVPSDE()
    pos = WrappedVESDE()  # the fractional-coordinate corruption

    assert isinstance(lattice, LatticeVPSDE)
    assert not isinstance(pos, LatticeVPSDE)

    # And the lattice field is NOT wrapped, so it must not receive the torus min-image
    # treatment either -- it uses the symmetric six-coordinate density.
    assert not isinstance(lattice, WrappedSDEMixin)
    assert isinstance(pos, WrappedSDEMixin)


def test_lattice_noise_is_symmetric_and_degenerate():
    torch.manual_seed(0)
    raw = torch.randn(4, 3, 3)
    z = make_noise_symmetric_preserve_variance(raw)
    # Symmetric: z_ij == z_ji => the 9 components are NOT independent (rank-6),
    # which is exactly why a per-component isotropic Gaussian ratio is wrong for cell.
    assert torch.allclose(z, z.transpose(1, 2), atol=1e-6)
    # Off-diagonal entries are genuinely duplicated (not trivially zero).
    assert z[:, 0, 1].abs().sum() > 0
