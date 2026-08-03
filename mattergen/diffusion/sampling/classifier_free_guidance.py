# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from typing import Any, Callable

import torch

from mattergen.diffusion.sampling.pc_sampler import Diffusable, PredictorCorrector
from mattergen.common.data.collate import collate
from mattergen.diffusion.wrapped.wrapped_sde import WrappedSDEMixin

BatchTransform = Callable[[Diffusable], Diffusable]


def identity(x: Diffusable) -> Diffusable:
    """
    Default function that transforms data to its conditional state
    """
    return x


def _gaussian_logp_per_row(
    sample: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    boundary: float | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-row Gaussian log-density, summing over all non-batch (row>0) dims.

    Args:
        sample: realized value, shape [num_rows, ...].
        mean: predicted mean, same shape as sample.
        std: standard deviation, broadcastable to sample.
        boundary: if not None, the field lives on a torus of this period
            (wrapped coordinates). The displacement (sample - mean) is reduced
            to its minimum image in [-boundary/2, boundary/2) before forming the
            Gaussian. REQUIRED for wrapped fields whose stored `sample` is
            wrapped into [0, boundary) while `mean` is the raw (unwrapped)
            predictor mean; without it a boundary crossing injects a spurious
            +boundary*delta/std**2 bias into the importance log-ratio. Exact as
            std -> 0 (where the bug is worst); a minimum-image approximation to
            the full periodic image sum at high noise.
        eps: floor on std for numerical stability.

    Returns:
        Tensor of shape [num_rows]: the per-row log-density (normalization
        constant included but it cancels in guided/uncond log-ratios).
    """
    std = torch.clamp(std, min=eps)
    diff = sample - mean
    if boundary is not None:
        diff = diff - boundary * torch.round(diff / boundary)
    diff = diff / std
    reduce_dims = tuple(range(1, sample.ndim))
    n = sample[0].numel() if sample.ndim > 1 else 1
    return (
        -0.5 * (diff * diff).sum(dim=reduce_dims)
        - torch.log(std).sum(dim=reduce_dims)
        - 0.5 * math.log(2.0 * math.pi) * n
    )


def _symmetric_gaussian_logp_per_sample(
    sample: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-12,
    symmetry_atol: float = 1e-5,
) -> torch.Tensor:
    """Gaussian log density on the six independent coordinates of a lattice.

    MatterGen's symmetric lattice noise has independent standard-normal
    coordinates ``[00, 01, 02, 11, 12, 22]``. The mirrored lower triangle is
    deterministic and must not be counted a second time.
    """
    if sample.ndim != 3 or sample.shape[1:] != (3, 3):
        raise ValueError(f"Expected lattice sample shape [B, 3, 3], got {tuple(sample.shape)}")
    if mean.shape != sample.shape:
        raise ValueError(
            f"Expected lattice mean shape {tuple(sample.shape)}, got {tuple(mean.shape)}"
        )
    for name, value in (("sample", sample), ("mean", mean)):
        if not torch.allclose(
            value,
            value.transpose(1, 2),
            atol=symmetry_atol,
            rtol=0.0,
        ):
            maximum = torch.max(torch.abs(value - value.transpose(1, 2))).item()
            raise ValueError(
                f"Lattice {name} is not symmetric within atol={symmetry_atol:g}; "
                f"maximum asymmetry={maximum:g}"
            )

    rows = torch.tensor([0, 0, 0, 1, 1, 2], device=sample.device)
    cols = torch.tensor([0, 1, 2, 1, 2, 2], device=sample.device)
    std_matrix = torch.broadcast_to(torch.clamp(std, min=eps), sample.shape)
    sample_coordinates = sample[:, rows, cols]
    mean_coordinates = mean[:, rows, cols]
    std_coordinates = std_matrix[:, rows, cols]
    standardized = (sample_coordinates - mean_coordinates) / std_coordinates
    return (
        -0.5 * torch.square(standardized).sum(dim=1)
        - torch.log(std_coordinates).sum(dim=1)
        - 0.5 * math.log(2.0 * math.pi) * sample_coordinates.shape[1]
    )


class GuidedPredictorCorrector(PredictorCorrector):
    """
    Sampler for classifier-free guidance.
    """

    def __init__(
        self,
        *,
        guidance_scale: float,
        remove_conditioning_fn: BatchTransform,
        keep_conditioning_fn: BatchTransform | None = None,
        **kwargs,
    ):
        """
        guidance_scale: gamma in p_gamma(x|y)=p(x)p(y|x)**gamma for classifier-free guidance
        remove_conditioning_fn: function that removes conditioning from the data
        keep_conditioning_fn: function that will be applied to the data before evaluating the conditional score. For example, this function might drop some fields that you never want to condition on or add fields that indicate which conditions should be respected.
        **kwargs: passed on to parent class constructor.
        """

        super().__init__(**kwargs)
        self._remove_conditioning_fn = remove_conditioning_fn
        self._keep_conditioning_fn = keep_conditioning_fn or identity
        self._guidance_scale = guidance_scale

    def _score_fn(
        self,
        x: Diffusable,
        t: torch.Tensor,
    ) -> Diffusable:
        """For each field, regardless of whether the corruption process is SDE or D3PM, we guide the score in the same way here,
        by taking a linear combination of the conditional and unconditional score model output.

        For discrete fields, the score model outputs are interpreted as logits, so the linear combination here means we compute logits for
        p_\gamma(x|y)=p(x)^(1-\gamma) p(x|y)^\gamma

        """

        def get_unconditional_score():
            return super(GuidedPredictorCorrector, self)._score_fn(
                x=self._remove_conditioning_fn(x), t=t
            )

        def get_conditional_score():
            return super(GuidedPredictorCorrector, self)._score_fn(
                x=self._keep_conditioning_fn(x), t=t
            )

        if abs(self._guidance_scale - 1) < 1e-15:
            return get_conditional_score()
        elif abs(self._guidance_scale) < 1e-15:
            return get_unconditional_score()
        else:
            # guided_score = guidance_factor * conditional_score + (1-guidance_factor) * unconditional_score
            batch_no_condition = self._remove_conditioning_fn(x)
            batch_with_condition = self._keep_conditioning_fn(x)
            joint_batch = collate([batch_no_condition, batch_with_condition])

            for attr,value in batch_no_condition.items():
                if isinstance(value, list):
                    joint_batch[attr] = batch_no_condition[attr]+batch_with_condition[attr]


            combined_score = super(GuidedPredictorCorrector, self)._score_fn(
                x=joint_batch, t=torch.cat([t, t], dim=0),
            )
            # Split the combined score back into unconditional and conditional parts.
            # Any batch.attr: list fields will be wrong here because of the manual concatenation above
            # this should be ok as self._multi_corruption.corrupted_fields are always torch.Tensor
            unconditional_score = combined_score[0]
            conditional_score = combined_score[1]

            return unconditional_score.replace(
                **{
                    k: torch.lerp(
                        unconditional_score[k], conditional_score[k], self._guidance_scale
                    )
                    for k in self._multi_corruption.corrupted_fields
                }
            )

    def _score_pair(
        self,
        *,
        x: Diffusable,
        t: torch.Tensor,
    ) -> tuple[Diffusable, Diffusable]:
        """
        Return (unconditional_score, conditional_score) in one forward pass (for CFG).
        """
        batch_no_condition = self._remove_conditioning_fn(x)
        batch_with_condition = self._keep_conditioning_fn(x)
        joint_batch = collate([batch_no_condition, batch_with_condition])

        # Keep list fields consistent with the original implementation.
        for attr, value in batch_no_condition.items():
            if isinstance(value, list):
                joint_batch[attr] = batch_no_condition[attr] + batch_with_condition[attr]

        combined_score = super(GuidedPredictorCorrector, self)._score_fn(
            x=joint_batch, t=torch.cat([t, t], dim=0)
        )
        unconditional_score = combined_score[0]
        conditional_score = combined_score[1]
        return unconditional_score, conditional_score

    @torch.no_grad()
    def _denoise_one_step_with_logp_pair(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        timestep_i: int,
        *,
        record: bool = False,
        predictor_logp_only: bool = False,
        include_lattice_logp: bool = False,
        record_per_field: bool = False,
        eps: float = 1e-12,
    ) -> tuple[Diffusable, Diffusable, list[Diffusable] | None, dict[str, torch.Tensor]]:
        """
        Like `_denoise_one_step`, but also returns per-sample log-probabilities for the
        realized transition under:
          - proposal: the *guided* (CFG) predictor kernel
          - target: the *unconditional* predictor kernel

        We compute Gaussian log-probs for continuous ancestral predictor updates.
        When ``include_lattice_logp`` is true, this includes the six independent
        symmetric lattice coordinates; it is false by default for compatibility.
        When ``record_per_field`` is true, ``info["_per_field_logp_guided"]`` and
        ``info["_per_field_logp_uncond"]`` break the two scalar totals down by
        component (``pos``, ``pos:corrector``, ``cell``, ``cell:corrector``,
        ``atomic_numbers``), each a per-sample ``[B]`` tensor summing exactly to
        the corresponding total. This is what makes it possible to attribute the
        importance ratio to the Langevin corrector vs the ancestral predictor,
        which cannot be recovered from a saved run (only the two summed scalars
        are persisted per edge) and cannot be obtained by differencing two runs
        (sampling is not reproducible run-to-run).
        If `predictor_logp_only=False`, we also include Gaussian log-probs for
        Langevin corrector steps **only when `use_empirical_stepsize=True`**, because
        the default Langevin corrector chooses step size using the sampled noise norm,
        which makes the induced transition non-Gaussian in closed form. Lattice
        correctors are scored as augmented transitions using their pre-polar latent.

        Returns:
            (batch, mean_batch, recorded_samples, info)
        where info contains:
            - "logp_guided": shape [B]
            - "logp_uncond": shape [B]
            - "logp_num_fields": number of fields included in logp
        """
        import warnings
        from torch_scatter import scatter_add

        from mattergen.diffusion.sampling.predictors import AncestralSamplingPredictor
        from mattergen.diffusion.sampling.predictors_correctors import (
            LangevinCorrector,
            empirical_step_size as base_empirical_step_size,
        )
        from mattergen.diffusion.corruption.multi_corruption import apply as multi_apply
        from mattergen.diffusion.corruption.corruption import maybe_expand
        from mattergen.diffusion.sampling.pc_sampler import _mask_replace
        from mattergen.common.diffusion.predictors_correctors import (
            LatticeLangevinDiffCorrector,
            empirical_step_size as lattice_empirical_step_size,
        )
        from mattergen.diffusion.d3pm.d3pm_predictors_correctors import (
            D3PMAncestralSamplingPredictor,
        )
        from mattergen.diffusion.discrete_time import to_discrete_time
        # Imported here (not at module top) to avoid a circular import: `mattergen.common`
        # depends on `mattergen.diffusion`, so this lower-level module must import lattice
        # symbols lazily.
        from mattergen.common.diffusion.corruption import LatticeVPSDE

        if isinstance(self._diffusion_module, torch.nn.Module):
            self._diffusion_module.eval()

        recorded_samples = None
        if record:
            recorded_samples = []

        # Ensure mask has defaults for all fns like base class.
        for k in self._predictors:
            mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)

        mean_batch = batch.clone()

        # Decreasing timesteps from T to eps_t (matches parent implementation).
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        # Set the timestep.
        t = torch.full((batch.get_batch_size(),), timesteps[timestep_i], device=self._device)

        B = batch.get_batch_size()
        logp_guided = torch.zeros((B,), device=self._device, dtype=torch.float32)
        logp_uncond = torch.zeros((B,), device=self._device, dtype=torch.float32)
        included_fields: list[str] = []
        lattice_logp_guided: dict[str, torch.Tensor] = {}
        lattice_logp_uncond: dict[str, torch.Tensor] = {}

        def _record_lattice_component(
            name: str,
            guided: torch.Tensor,
            unconditional: torch.Tensor,
        ) -> None:
            lattice_logp_guided[name] = lattice_logp_guided.get(
                name, torch.zeros_like(guided)
            ) + guided
            lattice_logp_uncond[name] = lattice_logp_uncond.get(
                name, torch.zeros_like(unconditional)
            ) + unconditional

        # Per-component breakdown of the two scalar totals. Only populated when
        # ``record_per_field``; the components sum exactly to ``logp_guided`` /
        # ``logp_uncond`` because every call site below sits inside the same
        # branch that accumulates into those totals.
        per_field_logp_guided: dict[str, torch.Tensor] = {}
        per_field_logp_uncond: dict[str, torch.Tensor] = {}

        def _record_per_field(
            name: str,
            guided: torch.Tensor,
            unconditional: torch.Tensor,
        ) -> None:
            """Accumulate a per-sample [B] logp contribution under ``name``.

            Accumulates rather than assigns: with ``n_steps_corrector > 1`` a
            corrector component is scored once per corrector step.
            """
            if not record_per_field:
                return
            per_field_logp_guided[name] = per_field_logp_guided.get(
                name, torch.zeros_like(guided)
            ) + guided
            per_field_logp_uncond[name] = per_field_logp_uncond.get(
                name, torch.zeros_like(unconditional)
            ) + unconditional

        # ---- Corrector updates (optional; NOT included in logp if predictor_logp_only) ----
        if self._correctors:
            if predictor_logp_only and self._n_steps_corrector > 0:
                warnings.warn(
                    "Computing predictor-only log-probs: corrector steps are executed but excluded from logp/IS weights "
                    "because the default LangevinCorrector chooses step_size using sampled noise norms.",
                    stacklevel=2,
                )
            for _ in range(self._n_steps_corrector):
                if not predictor_logp_only:
                    for _, corrector in self._correctors.items():
                        if isinstance(corrector, LangevinCorrector):
                            assert corrector.use_empirical_stepsize, (
                                "Corrector log-prob assumes a deterministic step_size. "
                                "Set use_empirical_stepsize=True or use predictor_logp_only=True, "
                                "because the default LangevinCorrector step_size depends on sampled noise norms."
                            )
                uncond_score, cond_score = self._score_pair(x=batch, t=t)
                guided_score = uncond_score.replace(
                    **{
                        k: torch.lerp(
                            uncond_score[k], cond_score[k], self._guidance_scale
                        )
                        for k in self._multi_corruption.corrupted_fields
                    }
                )
                x_pre_corrector: dict[str, torch.Tensor] = {
                    k: batch[k].clone() for k in self._correctors
                }
                lattice_corrector_latents: dict[str, torch.Tensor] = {}

                def _corrector_fn(
                    field_name: str,
                    corrector: Any,
                ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
                    if (
                        not include_lattice_logp
                        or not isinstance(corrector, LatticeLangevinDiffCorrector)
                    ):
                        return corrector.step_given_score

                    def _step_and_retain_latent(
                        **kwargs: Any,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        sample, mean, latent = corrector.step_given_score_with_latent(**kwargs)
                        lattice_corrector_latents[field_name] = latent
                        return sample, mean

                    return _step_and_retain_latent

                fns = {
                    k: _corrector_fn(k, corrector)
                    for k, corrector in self._correctors.items()
                }
                samples_means = multi_apply(
                    fns=fns,
                    broadcast={"t": t, "dt": dt},
                    x=batch,
                    score=guided_score,
                    batch_idx=self._multi_corruption._get_batch_indices(batch),
                )
                if record:
                    recorded_samples.append(batch.clone().to("cpu"))
                batch, mean_batch = _mask_replace(
                    samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                )
                if not predictor_logp_only:
                    batch_indices = self._multi_corruption._get_batch_indices(batch)
                    for field_name, corrector in self._correctors.items():
                        if not isinstance(corrector, LangevinCorrector):
                            continue
                        if isinstance(corrector.corruption, LatticeVPSDE):
                            if not include_lattice_logp:
                                continue
                            if mask.get(field_name) is not None:
                                continue
                            step_size = lattice_empirical_step_size(t)
                            step_size = maybe_expand(
                                step_size,
                                batch_indices[field_name],
                                guided_score[field_name],
                            )
                            std = torch.sqrt(torch.clamp(step_size * 2, min=eps))
                            mean_g = (
                                x_pre_corrector[field_name]
                                + step_size * guided_score[field_name]
                            )
                            mean_u = (
                                x_pre_corrector[field_name]
                                + step_size * uncond_score[field_name]
                            )
                            latent = lattice_corrector_latents.get(field_name)
                            if latent is None:
                                raise RuntimeError(
                                    f"Missing pre-polar latent for lattice corrector {field_name!r}."
                                )
                            lp_g = _symmetric_gaussian_logp_per_sample(
                                latent, mean_g, std, eps=eps
                            )
                            lp_u = _symmetric_gaussian_logp_per_sample(
                                latent, mean_u, std, eps=eps
                            )
                            logp_guided = logp_guided + lp_g
                            logp_uncond = logp_uncond + lp_u
                            component_name = f"{field_name}:corrector"
                            _record_lattice_component(component_name, lp_g, lp_u)
                            _record_per_field(component_name, lp_g, lp_u)
                            included_fields.append(component_name)
                            continue
                        if field_name not in batch_indices:
                            continue
                        if batch_indices[field_name] is None:
                            continue
                        if mask.get(field_name) is not None:
                            continue

                        if isinstance(corrector, LatticeLangevinDiffCorrector):
                            step_size = lattice_empirical_step_size(t)
                        else:
                            step_size = base_empirical_step_size(t)
                        step_size = maybe_expand(step_size, batch_indices[field_name], guided_score[field_name])
                        std = torch.sqrt(torch.clamp(step_size * 2, min=eps))

                        mean_g = x_pre_corrector[field_name] + step_size * guided_score[field_name]
                        mean_u = x_pre_corrector[field_name] + step_size * uncond_score[field_name]
                        sample = batch[field_name]

                        _corruption = corrector.corruption
                        _boundary = _corruption.wrapping_boundary if isinstance(_corruption, WrappedSDEMixin) else None
                        lp_g_rows = _gaussian_logp_per_row(sample, mean_g, std, boundary=_boundary)
                        lp_u_rows = _gaussian_logp_per_row(sample, mean_u, std, boundary=_boundary)
                        bidx = batch_indices[field_name]
                        lp_g_samples = scatter_add(lp_g_rows, index=bidx, dim=0, dim_size=B)
                        lp_u_samples = scatter_add(lp_u_rows, index=bidx, dim=0, dim_size=B)
                        logp_guided = logp_guided + lp_g_samples
                        logp_uncond = logp_uncond + lp_u_samples
                        _record_per_field(f"{field_name}:corrector", lp_g_samples, lp_u_samples)
                        included_fields.append(f"{field_name}:corrector")

        # ---- Predictor update (included in logp) ----
        uncond_score, cond_score = self._score_pair(x=batch, t=t)
        next_t = t + dt
        shared_final_predictor = include_lattice_logp and bool(
            torch.all(next_t <= 0).item()
        )
        if shared_final_predictor:
            # The terminal predictor is deterministic. Use the same unconditional
            # kernel for proposal and target so its ratio is one by construction,
            # while preserving any valid guided-corrector ratio from this step.
            guided_score = uncond_score
        else:
            guided_score = (
                cond_score
                if abs(self._guidance_scale - 1) < 1e-15
                else (
                    uncond_score
                    if abs(self._guidance_scale) < 1e-15
                    else uncond_score.replace(
                        **{
                            k: torch.lerp(
                                uncond_score[k], cond_score[k], self._guidance_scale
                            )
                            for k in self._multi_corruption.corrupted_fields
                        }
                    )
                )
            )

        # Snapshot the pre-update state per field so we can recompute means.
        x_pre: dict[str, torch.Tensor] = {k: batch[k].clone() for k in self._predictors}

        predictor_fns = {k: predictor.update_given_score for k, predictor in self._predictors.items()}
        samples_means = multi_apply(
            fns=predictor_fns,
            x=batch,
            score=guided_score,
            broadcast=dict(t=t, batch=batch, dt=dt),
            batch_idx=self._multi_corruption._get_batch_indices(batch),
        )
        if record:
            recorded_samples.append(batch.clone().to("cpu"))
        batch, mean_batch = _mask_replace(
            samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
        )

        batch_indices = self._multi_corruption._get_batch_indices(batch)
        # A shared terminal predictor contributes exactly zero to log(p/q). Its
        # component diagnostics may still be evaluated, but its common (and for
        # continuous fields singular) log-density is not added to either total.
        for field_name, predictor in self._predictors.items():
            # Only continuous ancestral (Gaussian) and D3PM (categorical) predictors have a
            # well-defined closed-form kernel here.
            if isinstance(predictor, AncestralSamplingPredictor):
                if isinstance(predictor.corruption, LatticeVPSDE):
                    if not include_lattice_logp:
                        continue
                    if mask.get(field_name) is not None:
                        continue
                    batch_idx = batch_indices[field_name]
                    x_coeff, score_coeff, std = predictor._get_coeffs(
                        x=x_pre[field_name],
                        t=t,
                        dt=dt,
                        batch_idx=batch_idx,
                        batch=batch,
                    )
                    mean_coeff = 1 - x_coeff
                    limit_mean = predictor.corruption.get_limit_mean(
                        x=x_pre[field_name], batch=batch
                    )
                    mean_g = (
                        x_coeff * x_pre[field_name]
                        + score_coeff * guided_score[field_name]
                        + mean_coeff * limit_mean
                    )
                    mean_u = (
                        x_coeff * x_pre[field_name]
                        + score_coeff * uncond_score[field_name]
                        + mean_coeff * limit_mean
                    )
                    sample = batch[field_name]
                    lp_g = _symmetric_gaussian_logp_per_sample(
                        sample, mean_g, std, eps=eps
                    )
                    lp_u = _symmetric_gaussian_logp_per_sample(
                        sample, mean_u, std, eps=eps
                    )
                    if lp_g.shape != (B,) or lp_u.shape != (B,):
                        raise RuntimeError(
                            "Lattice predictor log probabilities must have shape "
                            f"[{B}], got {tuple(lp_g.shape)} and {tuple(lp_u.shape)}."
                        )
                    if not shared_final_predictor:
                        logp_guided = logp_guided + lp_g
                        logp_uncond = logp_uncond + lp_u
                        _record_per_field(field_name, lp_g, lp_u)
                    _record_lattice_component(field_name, lp_g, lp_u)
                    included_fields.append(field_name)
                    continue
                if field_name not in batch_indices:
                    continue
                if batch_indices[field_name] is None:
                    # Some fields may not have batch indices (e.g., not present in this batch).
                    continue
                if mask.get(field_name) is not None:
                    # Inpainting masks produce partially deterministic transitions; skip for now.
                    continue

                # Coefficients are deterministic given (x_pre, t, dt, batch_idx, batch).
                x_coeff, score_coeff, std = predictor._get_coeffs(  # pylint: disable=protected-access
                    x=x_pre[field_name],
                    t=t,
                    dt=dt,
                    batch_idx=batch_indices[field_name],
                    batch=batch,
                )

                sample = batch[field_name]
                mean_g = x_coeff * x_pre[field_name] + score_coeff * guided_score[field_name]
                mean_u = x_coeff * x_pre[field_name] + score_coeff * uncond_score[field_name]

                _corruption = predictor.corruption
                _boundary = _corruption.wrapping_boundary if isinstance(_corruption, WrappedSDEMixin) else None
                lp_g_rows = _gaussian_logp_per_row(sample, mean_g, std, boundary=_boundary)
                lp_u_rows = _gaussian_logp_per_row(sample, mean_u, std, boundary=_boundary)

                # Aggregate row-level logp to per-sample logp via batch_idx.
                bidx = batch_indices[field_name]
                if not shared_final_predictor:
                    lp_g_samples = scatter_add(lp_g_rows, index=bidx, dim=0, dim_size=B)
                    lp_u_samples = scatter_add(lp_u_rows, index=bidx, dim=0, dim_size=B)
                    logp_guided = logp_guided + lp_g_samples
                    logp_uncond = logp_uncond + lp_u_samples
                    _record_per_field(field_name, lp_g_samples, lp_u_samples)
                included_fields.append(field_name)
            elif isinstance(predictor, D3PMAncestralSamplingPredictor):
                if field_name not in batch_indices:
                    continue
                if batch_indices[field_name] is None:
                    # Some fields may not have batch indices (e.g., not present in this batch).
                    continue
                if mask.get(field_name) is not None:
                    # Inpainting masks produce partially deterministic transitions; skip for now.
                    continue

                corruption = predictor.corruption  # D3PMCorruption instance
                bidx = batch_indices[field_name]

                # Discrete time index, per-atom via bidx -- mirrors
                # d3pm_predictors_correctors.py:65 (`t = to_discrete_time(...)`) and :88
                # (`t[batch_idx].to(torch.long)`).
                t_discrete = to_discrete_time(t=t, N=predictor.N, T=corruption.T)
                t_per_atom = t_discrete[bidx].to(torch.long)

                # Pre-update x_t, zero-based -- mirrors d3pm_predictors_correctors.py:90-92
                # (`self.corruption._to_zero_based(x)`, where `x` there is the pre-update state).
                x_pre_zero_based = corruption._to_zero_based(x_pre[field_name])

                def _posterior_logits(raw_logits: torch.Tensor) -> torch.Tensor:
                    class_probs = torch.softmax(raw_logits, dim=-1)
                    if predictor.predict_x0:
                        # Mirrors d3pm_predictors_correctors.py:86-94.
                        logits, _ = corruption.d3pm.sample_and_compute_posterior_q(
                            x_0=class_probs,
                            t=t_per_atom,
                            make_one_hot=False,
                            samples=x_pre_zero_based,
                            return_logits=True,
                        )
                        return logits
                    # predict_x0=False: the sampled-from distribution is the raw logits
                    # themselves (d3pm_predictors_correctors.py:67-74). Not exercised by any
                    # current config (sampling_conf/default.yaml always sets predict_x0=True);
                    # kept for API symmetry only, untested.
                    return raw_logits

                # Post-update x_{t-1}, zero-based -- mirrors the `x_sample` realized by
                # `update_given_score` (d3pm_predictors_correctors.py:96-98), converted back to
                # zero-based for `Categorical.log_prob`.
                realized_zero_based = corruption._to_zero_based(batch[field_name]).long()

                logits_g = _posterior_logits(guided_score[field_name])
                logits_u = _posterior_logits(uncond_score[field_name])

                lp_g_rows = torch.distributions.Categorical(logits=logits_g).log_prob(
                    realized_zero_based
                )
                lp_u_rows = torch.distributions.Categorical(logits=logits_u).log_prob(
                    realized_zero_based
                )

                if not shared_final_predictor:
                    lp_g_samples = scatter_add(lp_g_rows, index=bidx, dim=0, dim_size=B)
                    lp_u_samples = scatter_add(lp_u_rows, index=bidx, dim=0, dim_size=B)
                    logp_guided = logp_guided + lp_g_samples
                    logp_uncond = logp_uncond + lp_u_samples
                    _record_per_field(field_name, lp_g_samples, lp_u_samples)
                included_fields.append(field_name)
            else:
                continue
        if not include_lattice_logp and timestep_i >= 999:
            logp_uncond = logp_guided
            # The terminal-step compat shim overrides the totals, so the
            # per-field breakdown no longer sums to them. Drop it rather than
            # return components that disagree with logp_uncond.
            per_field_logp_guided.clear()
            per_field_logp_uncond.clear()
        info: dict[str, torch.Tensor] = {
            "logp_guided": logp_guided,
            "logp_uncond": logp_uncond,
            "logp_num_fields": torch.tensor(
                len(included_fields), device=self._device, dtype=torch.int64
            ),
        }
        # Keep a couple of "meta" items as python-only keys for debugging.
        info["_logp_field_names"] = included_fields  # type: ignore[index]
        info["_per_field_logp_guided"] = per_field_logp_guided  # type: ignore[index]
        info["_per_field_logp_uncond"] = per_field_logp_uncond  # type: ignore[index]
        info["_lattice_logp_guided"] = lattice_logp_guided  # type: ignore[index]
        info["_lattice_logp_uncond"] = lattice_logp_uncond  # type: ignore[index]
        info["_shared_final_predictor"] = torch.tensor(
            1 if shared_final_predictor else 0,
            device=self._device,
            dtype=torch.int64,
        )  # type: ignore[index]
        info["_predictor_logp_only"] = torch.tensor(1 if predictor_logp_only else 0, device=self._device, dtype=torch.int64)  # type: ignore[index]
        return batch, mean_batch, recorded_samples, info
