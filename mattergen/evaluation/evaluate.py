# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pymatgen.core.structure import Structure

from mattergen.common.utils.globals import get_device
from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.relaxation import relax_structures, get_total_energies
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DisorderedStructureMatcher,
    OrderedStructureMatcher,
)


def evaluate(
    structures: list[Structure],
    relax: bool = True,
    energies: list[float] | None = None,
    reference: ReferenceDataset | None = None,
    structure_matcher: (
        OrderedStructureMatcher | DisorderedStructureMatcher
    ) = DefaultDisorderedStructureMatcher(),
    save_as: str | None = None,
    potential_load_path: str | None = None,
    device: str = str(get_device()),
    structures_output_path: str | None = None,
    save_sun_structures: str | None = None,
) -> dict[str, float | int]:
    """Evaluate the structures against a reference dataset.

    Args:
        structures: List of structures to evaluate.
        relax: Whether to relax the structures before evaluation. Note that if this is run, `energies` will be ignored.
        energies: Energies of the structures if already relaxed and computed externally (e.g., from DFT).
        reference: Reference dataset. If this is None, the default reference dataset will be used.
        structure_matcher: Structure matcher to use for matching the structures.
        save_as: Save the metrics as a JSON file.
        potential_load_path: Path to the Machine Learning potential to use for relaxation.
        device: Device to use for relaxation.
        structures_output_path: Path to save the relaxed structures.
        save_sun_structures: Path  a directory to save the structures that are stable, unique, and novel (SUN).

    Returns:
        metrics: a dictionary of metrics and their values.
    """
    if relax and energies is not None:
        raise ValueError("Cannot accept energies if relax is True.")
    if relax:
        relaxed_structures, energies = relax_structures(
            structures, device=device, potential_load_path=potential_load_path, output_path=structures_output_path
        )
    else:
        relaxed_structures = structures
        energies = get_total_energies(
            structures, device=device, potential_load_path=potential_load_path
        )

    evaluator = MetricsEvaluator.from_structures_and_energies(
        structures=relaxed_structures,
        energies=energies,
        original_structures=structures,
        reference=reference,
        structure_matcher=structure_matcher,
    )
    compute_metrics_out =  evaluator.compute_metrics(
        metrics=evaluator.available_metrics,
        save_as=save_as,
        pretty_print=True,
    )
    
    if save_sun_structures:
        print("Saving SUN structures to", save_sun_structures)
        sun_structures = evaluator.filter(structures, evaluator.is_unique & evaluator.is_stable & evaluator.is_novel)
        sun_structures = [s for s in sun_structures if s is not None]
        if len(sun_structures) > 0:
            if not os.path.exists(save_sun_structures):
                os.makedirs(save_sun_structures)
            for i, s in enumerate(sun_structures):
                s.to(filename=os.path.join(save_sun_structures, f"sun_structure_{i}.cif"))
    return compute_metrics_out