"""
Phase Diagram Analysis & Convex Hull Filtering

Constructs low-temperature, zero-oxygen phase diagrams for
metallosilicon amino acid systems and filters candidates
by their distance from the convex hull.

Implements:
- Ternary/quaternary phase diagram construction
- Convex hull computation via QuickHull algorithm
- Hull distance calculation for stability assessment
- Gibbs free energy corrections at cryogenic temperatures
"""

import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
from scipy.spatial import ConvexHull

from scripts.gnome_model import (
    MolecularGraph, ELEMENT_PROPERTIES, ALL_ELEMENTS, METAL_ELEMENTS,
    BACKBONE_ELEMENTS,
)

logger = logging.getLogger(__name__)

# Reference compound energies for hull construction (eV/atom)
# These represent known stable phases in the Si-N-H-S-P-B-F-Me system
# at cryogenic temperatures in reducing environments

REFERENCE_COMPOUNDS = {
    # Binary compounds
    "Si3N4": {"composition": {"Si": 3, "N": 4}, "energy_per_atom": -7.82},
    "SiH4": {"composition": {"Si": 1, "H": 4}, "energy_per_atom": -4.21},
    "SiS2": {"composition": {"Si": 1, "S": 2}, "energy_per_atom": -5.15},
    "SiF4": {"composition": {"Si": 1, "F": 4}, "energy_per_atom": -8.93},
    "SiB3": {"composition": {"Si": 1, "B": 3}, "energy_per_atom": -6.12},
    "SiC": {"composition": {"Si": 1, "C": 1}, "energy_per_atom": -8.34},
    "NH3": {"composition": {"N": 1, "H": 3}, "energy_per_atom": -4.56},
    "N2H4": {"composition": {"N": 2, "H": 4}, "energy_per_atom": -3.89},
    "H2S": {"composition": {"H": 2, "S": 1}, "energy_per_atom": -3.72},
    "PH3": {"composition": {"P": 1, "H": 3}, "energy_per_atom": -3.45},
    "P2S5": {"composition": {"P": 2, "S": 5}, "energy_per_atom": -5.23},
    "BF3": {"composition": {"B": 1, "F": 3}, "energy_per_atom": -7.56},
    "B2H6": {"composition": {"B": 2, "H": 6}, "energy_per_atom": -3.98},
    "FeSi2": {"composition": {"Fe": 1, "Si": 2}, "energy_per_atom": -6.78},
    "FeS2": {"composition": {"Fe": 1, "S": 2}, "energy_per_atom": -5.89},
    "NiSi": {"composition": {"Ni": 1, "Si": 1}, "energy_per_atom": -6.12},
    "Ni3S2": {"composition": {"Ni": 3, "S": 2}, "energy_per_atom": -5.45},
    "TiSi2": {"composition": {"Ti": 1, "Si": 2}, "energy_per_atom": -7.23},
    "TiS2": {"composition": {"Ti": 1, "S": 2}, "energy_per_atom": -6.34},
    "MoSi2": {"composition": {"Mo": 1, "Si": 2}, "energy_per_atom": -7.89},
    "MoS2": {"composition": {"Mo": 1, "S": 2}, "energy_per_atom": -6.78},

    # Aluminum compounds (zero-oxygen framework)
    "AlN": {"composition": {"Al": 1, "N": 1}, "energy_per_atom": -5.89},
    "AlH3": {"composition": {"Al": 1, "H": 3}, "energy_per_atom": -3.98},
    "Al2S3": {"composition": {"Al": 2, "S": 3}, "energy_per_atom": -4.56},
    "AlP": {"composition": {"Al": 1, "P": 1}, "energy_per_atom": -4.78},
    "AlB2": {"composition": {"Al": 1, "B": 2}, "energy_per_atom": -5.23},
    "AlSi": {"composition": {"Al": 1, "Si": 1}, "energy_per_atom": -4.52},
    "AlSi2": {"composition": {"Al": 1, "Si": 2}, "energy_per_atom": -4.89},
    "AlF3": {"composition": {"Al": 1, "F": 3}, "energy_per_atom": -7.12},
    "Al2SiN2": {"composition": {"Al": 2, "Si": 1, "N": 2}, "energy_per_atom": -5.67},
    "AlSiS": {"composition": {"Al": 1, "Si": 1, "S": 1}, "energy_per_atom": -4.89},
    "AlSiN": {"composition": {"Al": 1, "Si": 1, "N": 1}, "energy_per_atom": -5.34},

    # Ternary compounds
    "Si2N2NH": {"composition": {"Si": 2, "N": 2, "H": 1}, "energy_per_atom": -6.45},
    "FeSiS": {"composition": {"Fe": 1, "Si": 1, "S": 1}, "energy_per_atom": -6.23},
    "TiSiN": {"composition": {"Ti": 1, "Si": 1, "N": 1}, "energy_per_atom": -7.12},
    "MoSiS": {"composition": {"Mo": 1, "Si": 1, "S": 1}, "energy_per_atom": -6.89},
    "FeAlSi": {"composition": {"Fe": 1, "Al": 1, "Si": 1}, "energy_per_atom": -5.67},
    "TiAlN": {"composition": {"Ti": 1, "Al": 1, "N": 1}, "energy_per_atom": -6.89},

    # Elemental reference states
    "Si_alpha": {"composition": {"Si": 1}, "energy_per_atom": -5.425},
    "N2": {"composition": {"N": 1}, "energy_per_atom": -8.336},
    "H2": {"composition": {"H": 1}, "energy_per_atom": -3.389},
    "S_alpha": {"composition": {"S": 1}, "energy_per_atom": -4.093},
    "P_black": {"composition": {"P": 1}, "energy_per_atom": -5.413},
    "B_alpha": {"composition": {"B": 1}, "energy_per_atom": -6.679},
    "Al_fcc": {"composition": {"Al": 1}, "energy_per_atom": -3.646},
    "F2": {"composition": {"F": 1}, "energy_per_atom": -3.398},
    "C_graphite": {"composition": {"C": 1}, "energy_per_atom": -9.223},
    "Fe_bcc": {"composition": {"Fe": 1}, "energy_per_atom": -8.457},
    "Ni_fcc": {"composition": {"Ni": 1}, "energy_per_atom": -5.780},
    "Ti_hcp": {"composition": {"Ti": 1}, "energy_per_atom": -7.899},
    "Mo_bcc": {"composition": {"Mo": 1}, "energy_per_atom": -10.940},
}


@dataclass
class HullVertex:
    """A point on the convex hull of the phase diagram."""
    name: str
    composition: Dict[str, int]
    energy_per_atom: float
    coordinates: np.ndarray  # reduced composition coordinates + energy


@dataclass
class PhaseDiagramResult:
    """Result of phase diagram analysis for a candidate."""
    candidate_id: str
    formation_energy: float
    hull_distance: float
    is_on_hull: bool
    decomposes_to: List[str]  # hull vertices it decomposes to
    stability_score: float


class PhaseDiagramAnalyzer:
    """
    Construct and analyze phase diagrams for metallosilicon systems.

    Uses convex hull construction to determine thermodynamic stability
    of candidate molecules relative to known reference compounds.
    """

    def __init__(
        self,
        temperature_K: float = 195.0,
        include_references: bool = True,
        hull_tolerance: float = 0.025,  # eV/atom
    ):
        self.temperature_K = temperature_K
        self.include_references = include_references
        self.hull_tolerance = hull_tolerance
        self.reference_compounds = REFERENCE_COMPOUNDS

        # Elements in our system
        self.system_elements = ["Si", "N", "H", "S", "P", "B", "F", "C", "Fe", "Ni", "Ti", "Mo"]

    def _composition_to_coordinates(self, composition: Dict[str, int]) -> np.ndarray:
        """
        Convert elemental composition to reduced coordinates for phase diagram.

        For N elements, uses N-1 composition fractions (sum to 1).
        """
        total = sum(composition.values())
        if total == 0:
            return np.zeros(len(self.system_elements) - 1)

        coords = np.zeros(len(self.system_elements) - 1)
        for i, elem in enumerate(self.system_elements[:-1]):
            coords[i] = composition.get(elem, 0) / total

        return coords

    def _compute_gibbs_correction(self, composition: Dict[str, int]) -> float:
        """
        Compute Gibbs free energy correction at cryogenic temperature.

        ΔG = ΔH - TΔS

        At low T (195K), the -TΔS term is small, making ΔG ≈ ΔH.
        This favors ordered, low-entropy structures.
        """
        n_atoms = sum(composition.values())

        # Configurational entropy (ideal mixing approximation)
        # S_config = -R Σ x_i ln(x_i)
        R = 8.314e-3  # kJ/(mol·K) → eV/(mol·K) conversion below
        total = n_atoms
        s_config = 0.0
        for count in composition.values():
            if count > 0:
                x = count / total
                if x > 0:
                    s_config -= x * np.log(x)

        # Convert to eV/atom at temperature
        # TΔS in eV/atom
        tds = self.temperature_K * s_config * R / 96.485  # kJ→eV

        # Vibrational entropy (Debye model approximation)
        # At 195K, vibrational entropy is ~30% of room temperature value
        s_vib = 0.3 * 0.001  # rough eV/(atom·K) * T factor
        tds_vib = self.temperature_K * s_vib

        return -(tds + tds_vib)  # negative because it stabilizes

    def build_convex_hull(
        self,
        candidates: List[Dict],
    ) -> Tuple[ConvexHull, List[HullVertex]]:
        """
        Build convex hull from reference compounds and candidates.

        Args:
            candidates: List of dicts with 'composition' and 'energy_per_atom'

        Returns:
            ConvexHull object and list of hull vertices
        """
        all_points = []
        all_names = []

        # Add reference compounds
        if self.include_references:
            for name, data in self.reference_compounds.items():
                coords = self._composition_to_coordinates(data["composition"])
                energy = data["energy_per_atom"]
                # Gibbs correction
                gibbs_corr = self._compute_gibbs_correction(data["composition"])
                energy_corrected = energy + gibbs_corr

                point = np.append(coords, energy_corrected)
                all_points.append(point)
                all_names.append(name)

        # Add candidates
        for cand in candidates:
            coords = self._composition_to_coordinates(cand["composition"])
            energy = cand["energy_per_atom"]
            gibbs_corr = self._compute_gibbs_correction(cand["composition"])
            energy_corrected = energy + gibbs_corr

            point = np.append(coords, energy_corrected)
            all_points.append(point)
            all_names.append(cand.get("name", "unknown"))

        points_array = np.array(all_points)

        # Build convex hull
        try:
            hull = ConvexHull(points_array)
        except Exception as e:
            logger.warning(f"Convex hull construction failed: {e}. Using simplified hull.")
            # Fallback: use only lower hull
            hull = None

        # Identify hull vertices
        vertices = []
        if hull is not None:
            for idx in hull.vertices:
                name = all_names[idx]
                point = all_points[idx]
                coords = point[:-1]
                energy = point[-1]

                # Find composition
                comp = {}
                if name in self.reference_compounds:
                    comp = self.reference_compounds[name]["composition"]
                else:
                    # Reconstruct from coordinates
                    for i, elem in enumerate(self.system_elements[:-1]):
                        frac = coords[i]
                        if frac > 0.01:
                            comp[elem] = max(1, int(round(frac * 10)))

                vertices.append(HullVertex(
                    name=name,
                    composition=comp,
                    energy_per_atom=energy,
                    coordinates=coords,
                ))

        return hull, vertices

    def compute_hull_distance(
        self,
        candidate: Dict,
        hull: Optional[ConvexHull],
        vertices: List[HullVertex],
    ) -> float:
        """
        Compute distance of a candidate from the convex hull.

        Distance = 0 means the candidate is on the hull (thermodynamically stable).
        Distance > 0 means the candidate is above the hull (metastable/unstable).

        Args:
            candidate: Dict with 'composition' and 'energy_per_atom'
            hull: ConvexHull object
            vertices: List of hull vertices

        Returns:
            Distance in eV/atom above the convex hull
        """
        coords = self._composition_to_coordinates(candidate["composition"])
        energy = candidate["energy_per_atom"]
        gibbs_corr = self._compute_gibbs_correction(candidate["composition"])
        energy_corrected = energy + gibbs_corr

        point = np.append(coords, energy_corrected)

        if hull is None:
            # Fallback: compare to lowest-energy reference at similar composition
            min_dist = abs(energy_corrected)
            for v in vertices:
                dist = abs(energy_corrected - v.energy_per_atom)
                min_dist = min(min_dist, dist)
            return min_dist

        # Fast hull distance approximation using nearest-hull-vertex interpolation.
        # Instead of expensive linprog per simplex, we find the K nearest hull
        # vertices in composition space and interpolate their energies.
        try:
            # Find nearest hull vertices by composition distance
            comp_dists = []
            for v in vertices:
                d = np.linalg.norm(coords - v.coordinates)
                comp_dists.append((d, v.energy_per_atom, v.coordinates))
            comp_dists.sort(key=lambda x: x[0])

            # Use inverse-distance-weighted interpolation from K nearest vertices
            K = min(5, len(comp_dists))
            if K == 0:
                return max(0.0, energy_corrected)

            weights = []
            energies = []
            for i in range(K):
                d, e, _ = comp_dists[i]
                if d < 1e-10:
                    # Point is on a hull vertex
                    return max(0.0, energy_corrected - e)
                w = 1.0 / (d * d)
                weights.append(w)
                energies.append(e)

            total_w = sum(weights)
            hull_energy = sum(w * e for w, e in zip(weights, energies)) / total_w
            hull_distance = energy_corrected - hull_energy
            return max(0.0, hull_distance)

        except Exception as e:
            logger.warning(f"Hull distance computation failed: {e}")
            return max(0.0, energy_corrected)

    def analyze_candidate(
        self,
        graph: MolecularGraph,
        formation_energy: float,
        hull: Optional[ConvexHull] = None,
        vertices: List[HullVertex] = None,
    ) -> PhaseDiagramResult:
        """
        Analyze a single candidate's position in the phase diagram.

        Args:
            graph: Molecular graph
            formation_energy: Pre-computed formation energy (eV/atom)
            hull: Pre-computed convex hull
            vertices: Hull vertices

        Returns:
            PhaseDiagramResult with stability assessment
        """
        composition = Counter(graph.atom_types)

        candidate_dict = {
            "name": graph.candidate_id,
            "composition": dict(composition),
            "energy_per_atom": formation_energy,
        }

        if hull is not None and vertices is not None:
            hull_distance = self.compute_hull_distance(candidate_dict, hull, vertices)
        else:
            hull_distance = max(0.0, formation_energy)

        is_on_hull = hull_distance <= self.hull_tolerance

        # Determine decomposition products
        decomposes_to = []
        if not is_on_hull and vertices:
            # Find nearest hull vertices
            dists = []
            for v in vertices:
                d = np.linalg.norm(
                    self._composition_to_coordinates(composition) - v.coordinates
                )
                dists.append((d, v.name))
            dists.sort()
            decomposes_to = [name for _, name in dists[:3]]

        # Stability score: sigmoid of hull distance
        stability_score = 1.0 / (1.0 + np.exp(hull_distance * 20))

        return PhaseDiagramResult(
            candidate_id=graph.candidate_id,
            formation_energy=formation_energy,
            hull_distance=hull_distance,
            is_on_hull=is_on_hull,
            decomposes_to=decomposes_to,
            stability_score=stability_score,
        )

    def analyze_batch(
        self,
        candidates: List[Tuple[MolecularGraph, float]],
    ) -> List[PhaseDiagramResult]:
        """
        Analyze a batch of candidates.

        Args:
            candidates: List of (MolecularGraph, formation_energy) tuples

        Returns:
            List of PhaseDiagramResult
        """
        # Build hull from all candidates + references
        candidate_dicts = []
        for graph, fe in candidates:
            composition = Counter(graph.atom_types)
            candidate_dicts.append({
                "name": graph.candidate_id,
                "composition": dict(composition),
                "energy_per_atom": fe,
            })

        hull, vertices = self.build_convex_hull(candidate_dicts)

        # Analyze each candidate
        results = []
        for graph, fe in candidates:
            result = self.analyze_candidate(graph, fe, hull, vertices)
            results.append(result)

        # Sort by hull distance
        results.sort(key=lambda r: r.hull_distance)

        n_on_hull = sum(1 for r in results if r.is_on_hull)
        logger.info(
            f"Phase diagram analysis: {n_on_hull}/{len(results)} candidates on hull"
        )

        return results

    def export_results(self, results: List[PhaseDiagramResult], output_path: str):
        """Export phase diagram results to JSON."""
        data = [asdict(r) for r in results]
        for d in data:
            d["coordinates"] = d.get("coordinates", []).tolist() if isinstance(d.get("coordinates"), np.ndarray) else d.get("coordinates", [])
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported {len(data)} phase diagram results to {output_path}")
