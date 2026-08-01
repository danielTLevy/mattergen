# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for MatterGen lattice predictor/corrector importance log probabilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from mattergen.common.diffusion.corruption import (
    LatticeVPSDE,
    make_noise_symmetric_preserve_variance,
)
from mattergen.common.diffusion.predictors_correctors import (
    LatticeAncestralSamplingPredictor,
    LatticeLangevinDiffCorrector,
)
from mattergen.diffusion.sampling.classifier_free_guidance import (
    GuidedPredictorCorrector,
    _symmetric_gaussian_logp_per_sample,
)


def _symmetric(values: torch.Tensor) -> torch.Tensor:
    """Symmetrize `[B, 3, 3]` values with unit-variance coordinates."""
    return make_noise_symmetric_preserve_variance(values)


class _FakeBatch:
    """Minimal Diffusable used to exercise the production paired transition."""

    def __init__(self, **values: torch.Tensor):
        self._values = values

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._values[name]

    def __contains__(self, name: str) -> bool:
        return name in self._values

    def items(self):
        return self._values.items()

    def clone(self) -> "_FakeBatch":
        return _FakeBatch(**{name: value.clone() for name, value in self.items()})

    def replace(self, **values: torch.Tensor) -> "_FakeBatch":
        return _FakeBatch(**{**self._values, **values})

    def get_batch_size(self) -> int:
        return self._values["cell"].shape[0]

    def get_batch_idx(self, field_name: str) -> None:
        del field_name
        return None


class _FakeMultiCorruption:
    """One-field corruption container matching the production sampler API."""

    def __init__(self, corruption: LatticeVPSDE):
        self.corrupted_fields = ["cell"]
        self.corruptions = {"cell": corruption}

    def _get_batch_indices(self, batch: _FakeBatch) -> dict[str, None]:
        del batch
        return {"cell": None}


def _paired_lattice_step(
    guidance_scale: float,
    timestep_i: int = 5,
    include_lattice_logp: bool = True,
) -> tuple[_FakeBatch, dict[str, Any]]:
    """Run the real paired path with real lattice predictor and corrector kernels."""
    batch_size = 4
    corruption = LatticeVPSDE()
    predictor = LatticeAncestralSamplingPredictor(corruption=corruption, score_fn=None)
    corrector = LatticeLangevinDiffCorrector(
        corruption=corruption,
        score_fn=None,
        n_steps=1,
        use_empirical_stepsize=True,
    )
    sampler = object.__new__(GuidedPredictorCorrector)
    sampler._diffusion_module = SimpleNamespace(
        corruption=_FakeMultiCorruption(corruption)
    )
    sampler._predictors = {"cell": predictor}
    sampler._correctors = {"cell": corrector}
    sampler._n_steps_corrector = 1
    sampler._max_t = 1.0
    sampler._eps_t = 1e-3
    sampler.N = 20
    sampler._device = torch.device("cpu")
    sampler._guidance_scale = guidance_scale

    uncond_cell = _symmetric(torch.randn(batch_size, 3, 3)) * 0.05
    cond_cell = uncond_cell + _symmetric(torch.randn(batch_size, 3, 3)) * 0.03
    uncond_score = _FakeBatch(cell=uncond_cell)
    cond_score = _FakeBatch(cell=cond_cell)
    sampler._score_pair = lambda *, x, t: (uncond_score, cond_score)

    batch = _FakeBatch(
        cell=torch.eye(3).expand(batch_size, 3, 3).clone() * 3.0,
        num_atoms=torch.tensor([4, 6, 8, 10]),
    )
    sample, _, _, info = sampler._denoise_one_step_with_logp_pair(
        batch=batch,
        mask={},
        timestep_i=timestep_i,
        include_lattice_logp=include_lattice_logp,
    )
    return sample, info


def test_symmetric_gaussian_logp_matches_analytic_reference() -> None:
    torch.manual_seed(0)
    sample = _symmetric(torch.randn(7, 3, 3, dtype=torch.float64))
    mean = _symmetric(torch.randn(7, 3, 3, dtype=torch.float64))
    std = torch.rand(7, 1, 1, dtype=torch.float64) + 0.2
    got = _symmetric_gaussian_logp_per_sample(sample, mean, std)
    coordinates = torch.triu_indices(3, 3)
    reference = torch.distributions.Normal(
        mean[:, coordinates[0], coordinates[1]],
        std.expand_as(mean)[:, coordinates[0], coordinates[1]],
    ).log_prob(sample[:, coordinates[0], coordinates[1]]).sum(dim=1)
    assert torch.allclose(got, reference, atol=1e-12, rtol=1e-12)


def test_guidance_zero_has_exact_zero_lattice_log_ratio() -> None:
    torch.manual_seed(1)
    _, info = _paired_lattice_step(guidance_scale=0.0)
    guided = info["_lattice_logp_guided"]
    unconditional = info["_lattice_logp_uncond"]
    assert guided.keys() == unconditional.keys() == {"cell", "cell:corrector"}
    for name in guided:
        assert torch.equal(guided[name], unconditional[name]), name


def test_symmetric_lattice_ratio_normalizes_under_guided_kernel() -> None:
    torch.manual_seed(2)
    draws = 100_000
    mean_guided = _symmetric(torch.randn(1, 3, 3, dtype=torch.float64) * 0.08)
    mean_uncond = torch.zeros_like(mean_guided)
    std = torch.full((1, 1, 1), 0.7, dtype=torch.float64)
    noise = _symmetric(torch.randn(draws, 3, 3, dtype=torch.float64))
    sample = mean_guided + std * noise
    log_q = _symmetric_gaussian_logp_per_sample(
        sample, mean_guided.expand_as(sample), std
    )
    log_p = _symmetric_gaussian_logp_per_sample(
        sample, mean_uncond.expand_as(sample), std
    )
    estimate = torch.exp(log_p - log_q).mean()
    assert torch.isclose(estimate, torch.tensor(1.0, dtype=estimate.dtype), atol=5e-3)


def test_production_pair_returns_finite_lattice_components() -> None:
    torch.manual_seed(3)
    sample, info = _paired_lattice_step(guidance_scale=1.5)
    assert sample["cell"].shape == (4, 3, 3)
    assert {"cell", "cell:corrector"}.issubset(info["_logp_field_names"])
    for density_name in ("_lattice_logp_guided", "_lattice_logp_uncond"):
        components = info[density_name]
        assert components.keys() == {"cell", "cell:corrector"}
        for value in components.values():
            assert value.shape == (4,)
            assert torch.isfinite(value).all()
    assert info["logp_guided"].shape == (4,)
    assert info["logp_uncond"].shape == (4,)
    assert torch.isfinite(info["logp_guided"]).all()
    assert torch.isfinite(info["logp_uncond"]).all()


def test_lattice_components_are_off_by_default() -> None:
    """The legacy partial ratio remains active unless V2 is explicitly enabled."""
    torch.manual_seed(6)
    _, info = _paired_lattice_step(
        guidance_scale=1.5,
        include_lattice_logp=False,
    )
    assert info["_lattice_logp_guided"] == {}
    assert info["_lattice_logp_uncond"] == {}
    assert "cell" not in info["_logp_field_names"]
    assert "cell:corrector" not in info["_logp_field_names"]


def test_final_predictor_is_shared_without_erasing_corrector_ratio() -> None:
    """The deterministic final predictor is shared on a non-1000-step schedule."""
    torch.manual_seed(5)
    _, info = _paired_lattice_step(guidance_scale=1.5, timestep_i=19)

    guided = info["_lattice_logp_guided"]
    unconditional = info["_lattice_logp_uncond"]
    assert info["_shared_final_predictor"].item() == 1
    assert torch.equal(guided["cell"], unconditional["cell"])

    corrector_log_ratio = (
        unconditional["cell:corrector"] - guided["cell:corrector"]
    )
    total_log_ratio = info["logp_uncond"] - info["logp_guided"]
    assert not torch.allclose(corrector_log_ratio, torch.zeros_like(corrector_log_ratio))
    assert torch.allclose(total_log_ratio, corrector_log_ratio, atol=1e-6, rtol=1e-6)


def test_public_lattice_corrector_path_is_unchanged() -> None:
    """Retaining the auxiliary latent does not alter ordinary corrector sampling."""
    corruption = LatticeVPSDE()
    corrector = LatticeLangevinDiffCorrector(
        corruption=corruption,
        score_fn=None,
        n_steps=1,
        use_empirical_stepsize=True,
    )
    x = torch.eye(3).expand(3, 3, 3).clone() * 2.5
    score = _symmetric(torch.randn(3, 3, 3)) * 0.1
    t = torch.full((3,), 0.6)
    dt = torch.tensor(-1.0 / 20)
    torch.manual_seed(4)
    public_sample, public_mean = corrector.step_given_score(
        x=x, batch_idx=None, score=score, t=t, dt=dt
    )
    torch.manual_seed(4)
    latent_sample, latent_mean, _ = corrector.step_given_score_with_latent(
        x=x, batch_idx=None, score=score, t=t, dt=dt
    )
    assert torch.equal(public_sample, latent_sample)
    assert torch.equal(public_mean, latent_mean)
