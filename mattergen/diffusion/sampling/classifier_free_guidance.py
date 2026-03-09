# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable

import torch

from mattergen.diffusion.sampling.pc_sampler import Diffusable, PredictorCorrector
from mattergen.common.data.collate import collate

BatchTransform = Callable[[Diffusable], Diffusable]


def identity(x: Diffusable) -> Diffusable:
    """
    Default function that transforms data to its conditional state
    """
    return x


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
        eps: float = 1e-12,
    ) -> tuple[Diffusable, Diffusable, list[Diffusable] | None, dict[str, torch.Tensor]]:
        """
        Like `_denoise_one_step`, but also returns per-sample log-probabilities for the
        realized transition under:
          - proposal: the *guided* (CFG) predictor kernel
          - target: the *unconditional* predictor kernel

        We compute **Gaussian log-probs for continuous ancestral predictor updates**.
        If `predictor_logp_only=False`, we also include Gaussian log-probs for
        Langevin corrector steps **only when `use_empirical_stepsize=True`**, because
        the default Langevin corrector chooses step size using the sampled noise norm,
        which makes the induced transition non-Gaussian in closed form.

        Returns:
            (batch, mean_batch, recorded_samples, info)
        where info contains:
            - "logp_guided": shape [B]
            - "logp_uncond": shape [B]
            - "logp_num_fields": number of fields included in logp
        """
        import math
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

        def gaussian_logp_per_row(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            # Returns logp per row (first dim), summing across remaining dims.
            std = torch.clamp(std, min=eps)
            diff = (sample - mean) / std
            # Sum over all non-batch dims
            reduce_dims = tuple(range(1, sample.ndim))
            return -0.5 * (diff * diff).sum(dim=reduce_dims) - torch.log(std).sum(dim=reduce_dims) - 0.5 * math.log(
                2.0 * math.pi
            ) * (sample[0].numel() if sample.ndim > 1 else 1)

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
                fns = {k: corrector.step_given_score for k, corrector in self._correctors.items()}
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

                        lp_g_rows = gaussian_logp_per_row(sample, mean_g, std)
                        lp_u_rows = gaussian_logp_per_row(sample, mean_u, std)
                        bidx = batch_indices[field_name]
                        logp_guided = logp_guided + scatter_add(
                            lp_g_rows, index=bidx, dim=0, dim_size=B
                        )
                        logp_uncond = logp_uncond + scatter_add(
                            lp_u_rows, index=bidx, dim=0, dim_size=B
                        )
                        included_fields.append(f"{field_name}:corrector")

        # ---- Predictor update (included in logp) ----
        uncond_score, cond_score = self._score_pair(x=batch, t=t)
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
        for field_name, predictor in self._predictors.items():
            # Only continuous ancestral predictors have a well-defined Gaussian kernel here.
            if not isinstance(predictor, AncestralSamplingPredictor):
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

            lp_g_rows = gaussian_logp_per_row(sample, mean_g, std)
            lp_u_rows = gaussian_logp_per_row(sample, mean_u, std)

            # Aggregate row-level logp to per-sample logp via batch_idx.
            bidx = batch_indices[field_name]
            logp_guided = logp_guided + scatter_add(lp_g_rows, index=bidx, dim=0, dim_size=B)
            logp_uncond = logp_uncond + scatter_add(lp_u_rows, index=bidx, dim=0, dim_size=B)
            included_fields.append(field_name)
        if timestep_i >= 999:
            logp_uncond = logp_guided
        info: dict[str, torch.Tensor] = {
            "logp_guided": logp_guided,
            "logp_uncond": logp_uncond,
            "logp_num_fields": torch.tensor(
                len(included_fields), device=self._device, dtype=torch.int64
            ),
        }
        # Keep a couple of "meta" items as python-only keys for debugging.
        info["_logp_field_names"] = included_fields  # type: ignore[index]
        info["_predictor_logp_only"] = torch.tensor(1 if predictor_logp_only else 0, device=self._device, dtype=torch.int64)  # type: ignore[index]
        return batch, mean_batch, recorded_samples, info
