# Copyright 2020 The Google Research Authors.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Adapted from https://github.com/yang-song/score_sde_pytorch which is released under Apache license.

# Key changes:
# - Introduced batch_idx argument to work with graph-like data (e.g. molecules)
# - Introduced `..._given_score` methods so that multiple fields can be sampled at once using a shared score model. See PredictorCorrector for how this is used.

import abc

import torch
from torch_scatter import scatter_add

from mattergen.diffusion.corruption.corruption import maybe_expand
from mattergen.diffusion.corruption.sde_lib import (
    VESDE,
    VPSDE,
    BaseVPSDE,
    Corruption,
    ScoreFunction,
)
from mattergen.diffusion.exceptions import IncompatibleSampler
from mattergen.diffusion.wrapped.wrapped_sde import WrappedSDEMixin

SampleAndMean = tuple[torch.Tensor, torch.Tensor]

def softplus(z):
    return torch.log1p(torch.exp(-torch.abs(z))) + torch.clamp(z, min=0.0)

def model_log_step_size_2knee(
    x, c, k1, x1, k2, x2,
    bump_amp=0.0, bump_mu=0.0, bump_sigma=1.0
):
    """
    y(x) = c - softplus(k1*(x-x1)) + softplus(k2*(x-x2)) + optional bump
    """
    y = c - softplus(k1 * (x - x1)) + softplus(k2 * (x - x2))

    if bump_amp != 0.0:
        y = y + bump_amp * torch.exp(
            -0.5 * ((x - bump_mu) / bump_sigma) ** 2
        )

    return torch.exp(y)


def empirical_step_size(t):
    "A function that roughly approximates the observed stepsize for base_model inference on MP-20"
    timestep = 1000*(1 - t)  # scale to [0, 1000]
    p_opt = torch.tensor([3.65566571e+00, 6.53975257e-02, 2.46792010e+02, 5.38615792e-02, 3.73406277e+02, 6.83580477e+00, 1.66335738e+02, 1.84996667e+02])
    return model_log_step_size_2knee(timestep, *p_opt)

class Sampler(abc.ABC):
    def __init__(self, corruption: Corruption, score_fn: ScoreFunction | None):
        if not self.is_compatible(corruption):
            raise IncompatibleSampler(
                f"{self.__class__.__name__} is not compatible with {corruption}"
            )
        self.corruption = corruption
        self.score_fn = score_fn

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        return True


class LangevinCorrector(Sampler):
    def __init__(
        self,
        corruption: Corruption,
        score_fn: ScoreFunction | None,
        n_steps: int,
        snr: float = 0.2,
        max_step_size: float = 1.0,
        use_empirical_stepsize = False
    ):
        """The Langevin corrector.

        Args:
            corruption: corruption process
            score_fn: score function
            n_steps: number of Langevin steps at each noise level
            snr: signal-to-noise ratio
            max_step_size: largest coefficient that the score can be multiplied by for each Langevin step.
        """
        super().__init__(corruption=corruption, score_fn=score_fn)
        self.n_steps = n_steps
        self.snr = snr
        self.max_step_size = torch.tensor(max_step_size)
        self.use_empirical_stepsize = use_empirical_stepsize

    @classmethod
    def is_compatible(cls, corruption: Corruption):
        return (
            isinstance(corruption, (VESDE, BaseVPSDE))
            and super().is_compatible(corruption)
            and not isinstance(corruption, WrappedSDEMixin)
        )

    def update_fn(self, *, x, t, batch_idx, dt: torch.Tensor) -> SampleAndMean:
        assert self.score_fn is not None, "Did you mean to use step_given_score?"
        for _ in range(self.n_steps):
            score = self.score_fn(x, t, batch_idx)
            x, x_mean = self.step_given_score(x=x, batch_idx=batch_idx, score=score, t=t, dt=dt)

        return x, x_mean

    def get_alpha(self, t: torch.FloatTensor, dt: torch.FloatTensor) -> torch.Tensor:
        sde = self.corruption

        if isinstance(sde, VPSDE):
            alpha_bar = sde._marginal_mean_coeff(t) ** 2
            alpha_bar_before = sde._marginal_mean_coeff(t + dt) ** 2
            alpha = alpha_bar / alpha_bar_before
        else:
            alpha = torch.ones_like(t)
        return alpha

    def step_given_score(
        self, *, x, batch_idx: torch.LongTensor | None, score, t: torch.Tensor, dt: torch.Tensor
    ) -> SampleAndMean:
        alpha = self.get_alpha(t, dt=dt)
        snr = self.snr
        noise = torch.randn_like(score)
        grad_norm_square = torch.square(score).reshape(score.shape[0], -1).sum(dim=1)
        noise_norm_square = torch.square(noise).reshape(noise.shape[0], -1).sum(dim=1)
        if batch_idx is None:
            grad_norm = grad_norm_square.sqrt().mean()
            noise_norm = noise_norm_square.sqrt().mean()
        else:
            grad_norm = torch.sqrt(scatter_add(grad_norm_square, dim=-1, index=batch_idx)).mean()

            noise_norm = torch.sqrt(scatter_add(noise_norm_square, dim=-1, index=batch_idx)).mean()

        # If gradient is zero (i.e., we are sampling from an improper distribution that's flat over the whole of R^n)
        # the step_size blows up. Clip step_size to avoid this.
        # The EGNN reports zero scores when there are no edges between nodes.
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        step_size = torch.minimum(step_size, self.max_step_size)
        step_size[grad_norm == 0, :] = self.max_step_size

        # Use empirical step size if specified
        if self.use_empirical_stepsize:
            step_size = empirical_step_size(t)

        # Expand step size to batch structure (score and noise have the same shape).
        step_size = maybe_expand(step_size, batch_idx, score)

        # Perform update, using custom update for SO(3) diffusion on frames.
        mean = x + step_size * score
        x = mean + torch.sqrt(step_size * 2) * noise

        return x, mean
