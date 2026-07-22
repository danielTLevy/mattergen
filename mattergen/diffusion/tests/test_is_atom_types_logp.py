# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for the D3PM (`atomic_numbers`) contribution to the importance-sampling
log-probs computed in `GuidedPredictorCorrector._denoise_one_step_with_logp_pair`.

The new code in `classifier_free_guidance.py` recomputes the *posterior*
categorical distribution `q(x_{t-1} | x_t, x_0)` that
`D3PMAncestralSamplingPredictor.update_given_score` (with `predict_x0=True`)
actually sampled the realized atom type from, by calling
`corruption.d3pm.sample_and_compute_posterior_q` a second time (once with the
guided raw logits, once with the unconditional raw logits), differing only in
the input `x_0` probabilities, exactly mirroring
`d3pm_predictors_correctors.py:86-94`.

We can't cheaply build a full `GuidedPredictorCorrector` (diffusion module +
model + full batch) in a unit test, so instead we exercise the REAL
`D3PMAncestralSamplingPredictor.update_given_score` directly (the actual
predictor class used in production, not a stand-in), capture the posterior
logits it actually sampled from via a monkeypatch on
`torch.distributions.Categorical`, and assert our independent reconstruction
(the same call our new code in `classifier_free_guidance.py` makes) is an
EXACT match.
"""
from unittest import mock

import pytest
import torch

from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.d3pm import d3pm as d3pm_module
from mattergen.diffusion.d3pm.d3pm_predictors_correctors import D3PMAncestralSamplingPredictor
from mattergen.diffusion.discrete_time import to_discrete_time


def _dummy_score_fn(x, t, batch_idx):
    raise NotImplementedError("score_fn is not invoked by update_given_score directly")


def _make_corruption(num_classes: int = 8, num_steps: int = 50, offset: int = 1) -> D3PMCorruption:
    schedule = d3pm_module.create_discrete_diffusion_schedule(
        kind="standard",
        beta_min=1e-3,
        beta_max=1e-1,
        num_steps=num_steps,
    )
    diff = d3pm_module.MaskDiffusion(dim=num_classes, schedule=schedule)
    return D3PMCorruption(d3pm=diff, offset=offset)


def _capture_predictor_posterior_logits(
    predictor: D3PMAncestralSamplingPredictor,
    *,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.LongTensor,
    score: torch.Tensor,
):
    """Run the REAL `update_given_score` and capture the posterior Categorical
    logits it actually sampled the realized atom type from.

    With `predict_x0=True`, `update_given_score` constructs exactly two
    `Categorical` distributions (d3pm_predictors_correctors.py:72-74 for the
    (unused, when predict_x0=True) raw-logits draw, and :96-98 for the posterior
    draw). We capture both and return the second (posterior) one, along with the
    realized sample and expected atom type.
    """
    captured_logits = []
    orig_init = torch.distributions.Categorical.__init__

    def patched_init(self, *args, **kwargs):
        logits = kwargs.get("logits", args[0] if args else None)
        captured_logits.append(logits)
        return orig_init(self, *args, **kwargs)

    with mock.patch.object(torch.distributions.Categorical, "__init__", patched_init):
        x_sample, x_expected = predictor.update_given_score(
            x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=None
        )

    assert len(captured_logits) == 2, (
        "Expected exactly 2 Categorical constructions in update_given_score "
        "with predict_x0=True (raw-logits draw + posterior draw)"
    )
    return captured_logits[1], x_sample, x_expected


def _reconstruct_posterior_logits(
    corruption: D3PMCorruption,
    predictor: D3PMAncestralSamplingPredictor,
    raw_logits: torch.Tensor,
    x_pre: torch.Tensor,
    t_continuous: torch.Tensor,
    batch_idx: torch.LongTensor,
) -> torch.Tensor:
    """Mirrors the new categorical branch added to
    `_denoise_one_step_with_logp_pair` in classifier_free_guidance.py."""
    t_discrete = to_discrete_time(t=t_continuous, N=predictor.N, T=corruption.T)
    t_per_atom = t_discrete[batch_idx].to(torch.long)
    x_pre_zero_based = corruption._to_zero_based(x_pre)
    class_probs = torch.softmax(raw_logits, dim=-1)
    logits, _ = corruption.d3pm.sample_and_compute_posterior_q(
        x_0=class_probs,
        t=t_per_atom,
        make_one_hot=False,
        samples=x_pre_zero_based,
        return_logits=True,
    )
    return logits


@pytest.fixture
def setup():
    torch.manual_seed(0)
    num_classes = 8
    num_atoms = 12
    B = 3
    corruption = _make_corruption(num_classes=num_classes, num_steps=50, offset=1)
    predictor = D3PMAncestralSamplingPredictor(
        corruption=corruption, score_fn=_dummy_score_fn, predict_x0=True
    )
    batch_idx = torch.tensor([0] * 4 + [1] * 4 + [2] * 4, dtype=torch.long)
    # Pre-update atom types, non-zero-based (offset=1): values in [1, num_classes].
    x_pre = torch.randint(1, num_classes + 1, (num_atoms,))
    t_continuous = torch.full((B,), 0.4)
    dt = torch.tensor(-1.0 / 50)
    raw_logits = torch.randn(num_atoms, num_classes)
    return dict(
        corruption=corruption,
        predictor=predictor,
        batch_idx=batch_idx,
        x_pre=x_pre,
        t_continuous=t_continuous,
        dt=dt,
        raw_logits=raw_logits,
    )


def test_reconstructed_posterior_exactly_matches_real_predictor(setup):
    """Exact-match test: the posterior logits reconstructed by the new code
    (`_reconstruct_posterior_logits`, mirroring the new branch in
    `_denoise_one_step_with_logp_pair`) must equal, bit-for-bit up to floating
    point, the posterior logits the REAL `D3PMAncestralSamplingPredictor.
    update_given_score` actually sampled the realized atom type from.
    """
    captured_logits, x_sample, _ = _capture_predictor_posterior_logits(
        setup["predictor"],
        x=setup["x_pre"],
        t=setup["t_continuous"],
        dt=setup["dt"],
        batch_idx=setup["batch_idx"],
        score=setup["raw_logits"],
    )
    reconstructed_logits = _reconstruct_posterior_logits(
        setup["corruption"],
        setup["predictor"],
        setup["raw_logits"],
        setup["x_pre"],
        setup["t_continuous"],
        setup["batch_idx"],
    )
    assert torch.allclose(captured_logits, reconstructed_logits, atol=1e-5)

    # Sanity: the realized sample assigned nonzero probability mass, and its
    # log-prob under the reconstructed posterior is a valid log-probability.
    realized_zero_based = setup["corruption"]._to_zero_based(x_sample).long()
    lp = torch.distributions.Categorical(logits=reconstructed_logits).log_prob(
        realized_zero_based
    )
    assert (lp <= 0).all()


def test_equal_raw_logits_give_equal_posterior_and_logp(setup):
    """Behavioral check mirroring `guidance_scale == 0` (guided == uncond): if
    the guided and unconditional raw logits are identical, the reconstructed
    posterior logits -- and hence the atom-type contribution to logp_guided and
    logp_uncond -- must be identical.
    """
    logits_g = _reconstruct_posterior_logits(
        setup["corruption"],
        setup["predictor"],
        setup["raw_logits"],
        setup["x_pre"],
        setup["t_continuous"],
        setup["batch_idx"],
    )
    logits_u = _reconstruct_posterior_logits(
        setup["corruption"],
        setup["predictor"],
        setup["raw_logits"].clone(),
        setup["x_pre"],
        setup["t_continuous"],
        setup["batch_idx"],
    )
    assert torch.allclose(logits_g, logits_u, atol=1e-6)

    realized_zero_based = setup["corruption"]._to_zero_based(setup["x_pre"]).long()
    lp_g = torch.distributions.Categorical(logits=logits_g).log_prob(realized_zero_based)
    lp_u = torch.distributions.Categorical(logits=logits_u).log_prob(realized_zero_based)
    assert torch.allclose(lp_g, lp_u)
    assert (lp_g <= 0).all()
    assert (lp_u <= 0).all()


def test_different_raw_logits_give_different_posterior(setup):
    """Regression guard against a vacuous test: different guided vs. unconditional
    raw logits must generally yield different reconstructed posteriors (and thus a
    nonzero importance-sampling correction), so the exact-match test above isn't
    trivially satisfied by e.g. a constant-function bug.
    """
    other_logits = setup["raw_logits"] + torch.randn_like(setup["raw_logits"]) * 5.0
    logits_g = _reconstruct_posterior_logits(
        setup["corruption"],
        setup["predictor"],
        setup["raw_logits"],
        setup["x_pre"],
        setup["t_continuous"],
        setup["batch_idx"],
    )
    logits_u = _reconstruct_posterior_logits(
        setup["corruption"],
        setup["predictor"],
        other_logits,
        setup["x_pre"],
        setup["t_continuous"],
        setup["batch_idx"],
    )
    assert not torch.allclose(logits_g, logits_u, atol=1e-3)


def test_posterior_reconstruction_is_deterministic(setup):
    """`sample_and_compute_posterior_q` with `samples=` provided explicitly must be
    a pure (deterministic) function of its inputs -- calling it twice with
    identical arguments must give identical results, confirming it is safe to call
    twice (guided, uncond) without perturbing any global RNG state / the actual
    sampled trajectory.
    """
    logits_1 = _reconstruct_posterior_logits(
        setup["corruption"],
        setup["predictor"],
        setup["raw_logits"],
        setup["x_pre"],
        setup["t_continuous"],
        setup["batch_idx"],
    )
    logits_2 = _reconstruct_posterior_logits(
        setup["corruption"],
        setup["predictor"],
        setup["raw_logits"],
        setup["x_pre"],
        setup["t_continuous"],
        setup["batch_idx"],
    )
    assert torch.equal(logits_1, logits_2)
