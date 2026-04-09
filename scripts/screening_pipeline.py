"""
Formation Energy Prediction & Screening Pipeline

Screens metallosilicon amino acid candidates using:
1. GNoME-predicted formation energies
2. Convex hull distance in low-T, zero-O phase diagram
3. Thermodynamic stability filters for cryogenic reducing environments
4. Solvent compatibility scoring (liquid NH3, CH4, H2S)
"""

import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
from copy import deepcopy

from scripts.gnome_model import (
    GNoMEModel, MolecularGraph, PredictionResult, create_pretrained_model,
    BOND_ENERGIES, ELEMENT_PROPERTIES, ALL_ELEMENTS, METAL_ELEMENTS,
    BACKBONE_ELEMENTS, METAL_COORDINATION,
)

logger = logging.getLogger(__name__)

# ─── Reference Energies for Formation Energy Calculation ─────────────────────
REFERENCE_ENERGIES = {
    "Si": -5.425, "N": -8.336, "H": -3.389, "S": -4.093,
    "P": -5.413, "B": -6.679, "Al": -3.646, "F": -3.398, "C": -9.223,
    "Fe": -8.457, "Ni": -5.780, "Ti": -7.899, "Mo": -10.940,
}

SOLVENT_CORRECTIONS = {
    "liquid_ammonia": {
        "Si": 0.05, "N": -0.15, "H": -0.08, "S": 0.02,
        "P": 0.03, "B": 0.04, "Al": 0.05, "F": -0.10, "C": 0.01,
        "Fe": 0.08, "Ni": 0.07, "Ti": 0.06, "Mo": 0.09,
    },
    "liquid_methane": {
        "Si": 0.12, "N": 0.08, "H": -0.05, "S": 0.10,
        "P": 0.11, "B": 0.09, "Al": 0.10, "F": 0.15, "C": -0.12,
        "Fe": 0.15, "Ni": 0.14, "Ti": 0.13, "Mo": 0.16,
    },
    "liquid_hydrogen_sulfide": {
        "Si": 0.03, "N": 0.05, "H": -0.03, "S": -0.18,
        "P": 0.02, "B": 0.04, "Al": 0.03, "F": 0.08, "C": 0.02,
        "Fe": 0.05, "Ni": 0.04, "Ti": 0.03, "Mo": 0.06,
    },
    "supercritical_ammonia": {
        "Si": 0.08, "N": -0.12, "H": -0.06, "S": 0.04,
        "P": 0.05, "B": 0.06, "Al": 0.07, "F": -0.08, "C": 0.02,
        "Fe": 0.10, "Ni": 0.09, "Ti": 0.08, "Mo": 0.11,
    },
}


@dataclass
class ScreeningResult:
    """Result of screening a single candidate."""
    candidate_id: str
    passed: bool
    formation_energy: float
    hull_distance: float
    stability_score: float
    solvent_scores: Dict[str, float]
    homo_lumo_gap: float
    coordination_geometry: str
    failure_reasons: List[str] = None
    warnings: List[str] = None


class FormationEnergyCalculator:
    """Calculate formation energies for metallosilicon amino acid candidates."""

    def __init__(self, solvent: str = "liquid_ammonia", temperature_K: float = 195.0):
        self.solvent = solvent
        self.temperature_K = temperature_K
        self.solvent_corrections = SOLVENT_CORRECTIONS.get(solvent, {})

    def compute_bond_energy_estimate(self, graph: MolecularGraph) -> float:
        """Estimate total energy from bond energies."""
        adj = graph.adjacency
        atom_types = graph.atom_types
        n_atoms = len(atom_types)

        total_bond_energy = 0.0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if adj[i, j] > 0:
                    bond_order = int(adj[i, j])
                    elem_i = atom_types[i]
                    elem_j = atom_types[j]
                    key1 = (elem_i, elem_j)
                    key2 = (elem_j, elem_i)
                    if key1 in BOND_ENERGIES:
                        be = BOND_ENERGIES[key1]
                    elif key2 in BOND_ENERGIES:
                        be = BOND_ENERGIES[key2]
                    else:
                        be = 250.0
                    total_bond_energy += (be * bond_order) / 96.485

        total_ref_energy = sum(REFERENCE_ENERGIES.get(e, -5.0) for e in atom_types)
        total_solvent_correction = sum(self.solvent_corrections.get(e, 0.0) for e in atom_types)
        # Temperature-dependent entropy correction: ΔG ≈ ΔH - TΔS
        # At higher T, vibrational/rotational entropy favors molecules with
        # more bonds (higher coordination, heavier elements). We apply a
        # entropic stabilization that scales with bond count and T.
        bond_count = int(np.sum(adj) / 2)
        temp_correction = -0.008 * bond_count * (self.temperature_K / 300.0)

        total_energy = -total_bond_energy + total_ref_energy + total_solvent_correction + temp_correction
        return total_energy

    def compute_formation_energy(self, graph: MolecularGraph) -> float:
        """Compute formation energy per atom relative to reference states."""
        n_atoms = len(graph.atom_types)
        total_energy = self.compute_bond_energy_estimate(graph)
        composition = Counter(graph.atom_types)
        ref_sum = sum(count * REFERENCE_ENERGIES.get(elem, -5.0) for elem, count in composition.items())
        formation_energy = (total_energy - ref_sum) / n_atoms
        return formation_energy

    def compute_solvent_stability(self, graph: MolecularGraph) -> Dict[str, float]:
        """Compute stability score in each solvent environment."""
        composition = Counter(graph.atom_types)
        n_atoms = len(graph.atom_types)
        results = {}

        for solvent_name, corrections in SOLVENT_CORRECTIONS.items():
            total_correction = sum(
                count * corrections.get(elem, 0.0) for elem, count in composition.items()
            )
            avg_correction = total_correction / n_atoms
            stability = 1.0 / (1.0 + np.exp(avg_correction * 10))

            if solvent_name in ("liquid_ammonia", "supercritical_ammonia"):
                n_fraction = composition.get("N", 0) / n_atoms
                stability += 0.2 * n_fraction
                if solvent_name == "supercritical_ammonia":
                    stability += 0.10  # higher T solvation stabilization
            elif solvent_name == "liquid_methane":
                si_h_fraction = (composition.get("Si", 0) + composition.get("H", 0)) / n_atoms
                stability += 0.15 * si_h_fraction
            elif solvent_name == "liquid_hydrogen_sulfide":
                s_fraction = composition.get("S", 0) / n_atoms
                stability += 0.25 * s_fraction

            results[solvent_name] = min(1.0, max(0.0, stability))

        return results


class ScreeningPipeline:
    """Multi-stage screening pipeline for metallosilicon amino acid candidates."""

    def __init__(
        self,
        model: Optional[GNoMEModel] = None,
        formation_energy_max: float = 0.2,
        hull_distance_max: float = 0.05,
        solvent_stability_min: float = 0.4,
        solvent: str = "liquid_ammonia",
        temperature_K: float = 195.0,
        use_gnn: bool = True,
        device: str = "cpu",
    ):
        self.formation_energy_max = formation_energy_max
        self.hull_distance_max = hull_distance_max
        self.solvent_stability_min = solvent_stability_min
        self.solvent = solvent
        self.temperature_K = temperature_K
        self.use_gnn = use_gnn
        self.device = device

        if model is not None:
            self.model = model
        elif use_gnn:
            self.model = create_pretrained_model()
            self.model.to(device)
            self.model.eval()
        else:
            self.model = None

        self.energy_calc = FormationEnergyCalculator(solvent=solvent, temperature_K=temperature_K)

        self.stats = {
            "total_screened": 0,
            "passed_formation_energy": 0,
            "passed_hull_distance": 0,
            "passed_solvent": 0,
            "passed_all": 0,
            "failed_formation_energy": 0,
            "failed_hull_distance": 0,
            "failed_solvent": 0,
            "failed_structural": 0,
        }

    def _check_valence_sanity(self, graph: MolecularGraph) -> Tuple[bool, List[str]]:
        """Check that valence constraints are approximately satisfied."""
        warnings = []
        valence_limits = {
            "Si": 4, "N": 3, "H": 1, "S": 2, "P": 3,
            "B": 3, "Al": 3, "F": 1, "C": 4, "Fe": 8, "Ni": 6, "Ti": 6, "Mo": 6,
        }
        for i, elem in enumerate(graph.atom_types):
            valence = int(graph.adjacency[i].sum())
            limit = valence_limits.get(elem, 6)
            if valence > limit + 1:
                warnings.append(f"Atom {i} ({elem}): valence {valence} exceeds limit {limit}")
        return len(warnings) == 0, warnings

    def _check_stability_warnings(self, graph: MolecularGraph) -> List[str]:
        """Flag molecules likely to be unstable in specific environments."""
        warnings = []
        composition = Counter(graph.atom_types)
        n_atoms = len(graph.atom_types)

        si_count = composition.get("Si", 0)
        h_count = composition.get("H", 0)
        if si_count > 0 and h_count > si_count:
            warnings.append(
                "⚠️ STABILITY WARNING: Si-H bonds are pyrophoric in O₂ atmosphere. "
                "Strictly stable only in reducing solvents."
            )

        adj = graph.adjacency
        si_si_bonds = 0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if graph.atom_types[i] == "Si" and graph.atom_types[j] == "Si" and adj[i, j] > 0:
                    si_si_bonds += 1
        if si_si_bonds > 2:
            warnings.append(
                f"⚠️ STABILITY WARNING: {si_si_bonds} Si-Si bonds detected. "
                f"Si-Si bond energy (226 kJ/mol) is significantly weaker than C-C (347 kJ/mol). "
                f"May be unstable above 250K."
            )

        if si_count > 0:
            warnings.append(
                "⚠️ STABILITY WARNING: Silicon-containing molecules will hydrolyze "
                "violently upon contact with H₂O. Si-O bond formation (452 kJ/mol) "
                "drives spontaneous combustion/hydrolysis."
            )

        f_count = composition.get("F", 0)
        if f_count > 2:
            warnings.append(
                f"⚠️ STABILITY WARNING: {f_count} F atoms present. "
                "Si-F bonds are extremely strong (597 kJ/mol) but F is highly reactive."
            )

        metal_atoms = [e for e in graph.atom_types if e in METAL_ELEMENTS]
        if metal_atoms and self.temperature_K > 300:
            warnings.append(
                f"⚠️ STABILITY WARNING: Metal-silicon bonds may be unstable "
                f"at {self.temperature_K}K. Recommend cryogenic conditions (<250K)."
            )

        return warnings

    def _estimate_homo_lumo(self, graph: MolecularGraph) -> float:
        """Estimate HOMO-LUMO gap from composition and bonding."""
        composition = Counter(graph.atom_types)
        n_atoms = len(graph.atom_types)

        electronegativities = []
        for elem in graph.atom_types:
            props = ELEMENT_PROPERTIES.get(elem, [1, 1.0, 50, 1])
            electronegativities.append(props[1])

        en_spread = max(electronegativities) - min(electronegativities)
        base_gap = 0.5 + en_spread * 0.8

        metal_count = sum(composition.get(m, 0) for m in METAL_ELEMENTS)
        if metal_count > 0:
            base_gap *= 0.4

        si_si_bonds = 0
        adj = graph.adjacency
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if graph.atom_types[i] == "Si" and graph.atom_types[j] == "Si" and adj[i, j] > 0:
                    si_si_bonds += 1
        base_gap -= si_si_bonds * 0.05

        f_count = composition.get("F", 0)
        base_gap += f_count * 0.1

        return max(0.1, base_gap)

    def _infer_coordination_geometry(self, graph: MolecularGraph) -> str:
        """Infer coordination geometry from metal center bonding."""
        if graph.metal_index is None:
            return "unknown"
        n_ligands = int(graph.adjacency[graph.metal_index].sum())
        if n_ligands <= 4:
            return "tetrahedral"
        else:
            return "octahedral"

    def screen_single(self, graph: MolecularGraph) -> ScreeningResult:
        """Screen a single candidate through all pipeline stages."""
        self.stats["total_screened"] += 1
        failure_reasons = []

        # ─── Primary: Bond-energy-based formation energy calculation ───────────
        # The GNoME model is untrained (Xavier-initialized), producing random
        # formation energies. Use the physically-motivated bond-energy model
        # as the primary calculator. The GNoME model will be blended in only
        # after proper training on DFT-labeled data.
        formation_energy = self.energy_calc.compute_formation_energy(graph)
        hull_distance = max(0.0, formation_energy)
        stability_score = 1.0 / (1.0 + np.exp(formation_energy * 5))
        homo_lumo = self._estimate_homo_lumo(graph)
        coord_geom = self._infer_coordination_geometry(graph)

        # ─── Optional: GNoME blending (only when model is trained) ────────────
        # If the GNoME model has been trained (detected via a trained flag or
        # checkpoint), blend its predictions with the bond-energy calculator.
        # For now, skip GNoME since it's untrained (random weights).
        if self.use_gnn and self.model is not None and getattr(self.model, 'is_trained', False):
            prediction = self.model.predict(graph)
            # Weighted blend: 60% GNoME, 40% bond-energy (favoring GNoME when trained)
            blend_weight = 0.6
            formation_energy = (blend_weight * prediction.formation_energy +
                              (1 - blend_weight) * formation_energy)
            hull_distance = max(0.0, formation_energy)
            homo_lumo = homo_lumo * 0.5 + prediction.homo_lumo_gap * 0.5

        if formation_energy > self.formation_energy_max:
            failure_reasons.append(
                f"Formation energy {formation_energy:.4f} eV/atom exceeds threshold {self.formation_energy_max:.4f}"
            )
            self.stats["failed_formation_energy"] += 1
        else:
            self.stats["passed_formation_energy"] += 1

        if hull_distance > self.hull_distance_max:
            failure_reasons.append(
                f"Hull distance {hull_distance:.4f} eV exceeds tolerance {self.hull_distance_max:.4f}"
            )
            self.stats["failed_hull_distance"] += 1
        else:
            self.stats["passed_hull_distance"] += 1

        solvent_scores = self.energy_calc.compute_solvent_stability(graph)
        primary_solvent_score = solvent_scores.get(self.solvent, 0.0)

        if primary_solvent_score < self.solvent_stability_min:
            failure_reasons.append(
                f"Solvent stability {primary_solvent_score:.4f} in {self.solvent} below minimum {self.solvent_stability_min:.4f}"
            )
            self.stats["failed_solvent"] += 1
        else:
            self.stats["passed_solvent"] += 1

        valence_ok, valence_warnings = self._check_valence_sanity(graph)
        if not valence_ok:
            failure_reasons.extend(valence_warnings)
            self.stats["failed_structural"] += 1

        stability_warnings = self._check_stability_warnings(graph)

        passed = len(failure_reasons) == 0
        if passed:
            self.stats["passed_all"] += 1

        return ScreeningResult(
            candidate_id=graph.candidate_id,
            passed=passed,
            formation_energy=formation_energy,
            hull_distance=hull_distance,
            stability_score=stability_score,
            solvent_scores=solvent_scores,
            homo_lumo_gap=homo_lumo,
            coordination_geometry=coord_geom,
            failure_reasons=failure_reasons if failure_reasons else None,
            warnings=stability_warnings if stability_warnings else None,
        )

    def screen_batch(
        self,
        candidates: List[MolecularGraph],
        top_n: int = 100,
    ) -> Tuple[List[ScreeningResult], List[ScreeningResult]]:
        """Screen a batch of candidates and return top-N stable ones."""
        all_results = []
        for graph in candidates:
            result = self.screen_single(graph)
            all_results.append(result)

        all_results.sort(key=lambda r: r.formation_energy)
        passed = [r for r in all_results if r.passed]
        top_passed = passed[:top_n]

        logger.info(
            f"Screening complete: {len(passed)}/{len(all_results)} passed. "
            f"Returning top {len(top_passed)}."
        )
        return top_passed, all_results

    def export_results(self, results: List[ScreeningResult], output_path: str):
        """Export screening results to JSON."""
        data = [asdict(r) for r in results]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported {len(data)} screening results to {output_path}")

    def get_screening_report(self) -> Dict:
        """Return screening statistics."""
        return deepcopy(self.stats)
