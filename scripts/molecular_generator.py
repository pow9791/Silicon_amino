"""
Metallosilicon Molecular Graph Generator

Generates candidate molecular topologies for metallosilicon amino acid analogs
under strict elemental constraints:
- Primary backbone: Si, N, H, S, P, B, Al, F
- Transition metals: Fe, Ni, Ti, Mo
- Minimal/zero carbon, strict low-oxygen (O < 1% atomic fraction)
- Functional group analogs: phosphine (-PH2), silazane (-Si-N-Si-),
  thiol/dithiocarboxyl (-CSSH, -SiSH), silanoic acid analogs, aluminate
"""

import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from itertools import product, combinations
from copy import deepcopy
from collections import Counter

from scripts.gnome_model import (
    MolecularGraph, ALL_ELEMENTS, BACKBONE_ELEMENTS, METAL_ELEMENTS,
    TRACE_ELEMENTS, ELEMENT_PROPERTIES, BOND_ENERGIES, METAL_COORDINATION,
)

logger = logging.getLogger(__name__)

FORMULA_ORDER = ["Fe", "Ni", "Ti", "Mo", "Si", "Al", "N", "P", "S", "B", "F", "C", "H"]

METAL_DISPLAY = {
    "Fe": "Iron",
    "Ni": "Nickel",
    "Ti": "Titanium",
    "Mo": "Molybdenum",
    None: "",
}

BACKBONE_DISPLAY = {
    "aluminosilazane_chain": "aluminosilazane",
    "silazane_chain": "silazane",
    "aluminosilazane_short": "short aluminosilazane",
    "sila_phosphorus_chain": "phosphasilazane",
    "sila_boron_chain": "borasilazane",
    "sila_sulfur_chain": "thiosilazane",
    "aluminosilazane_sulfur": "thio-aluminosilazane",
}

FUNCTIONAL_GROUP_DISPLAY = {
    "phosphine": "phosphine",
    "silazane_amine": "silazane amine",
    "secondary_silazane": "secondary silazane",
    "thiol": "thiol",
    "silanethiol": "silanethiol",
    "dithiocarboxyl": "dithiocarboxyl",
    "silanoic_sulfur": "thio-silanoic",
    "borane_acid": "borane",
    "alumino_silyl": "aluminosilyl",
    "alumino_amine": "alumino amine",
    "alumino_thiol": "alumino thiol",
    "fluoro_silyl": "fluorosilyl",
    "phosphino_silyl": "phosphinosilyl",
    "boranyl_silyl": "boranylsilyl",
}

# ─── Functional Group Templates ──────────────────────────────────────────────

FUNCTIONAL_GROUPS = {
    # Amine equivalents (electron donors)
    "phosphine": {
        "atoms": ["P", "H", "H"],
        "bonds": [(0, 1, 1), (0, 2, 1)],  # P-H, P-H
        "connect_atom": 0,  # P connects to backbone
        "description": "Phosphine group (-PH2), amine analog",
    },
    "silazane_amine": {
        "atoms": ["N", "Si", "H"],
        "bonds": [(0, 1, 1), (0, 2, 1)],  # N-Si, N-H
        "connect_atom": 0,  # N connects to backbone
        "description": "Silazane-based amine (-N(H)-Si<)",
    },
    "secondary_silazane": {
        "atoms": ["N", "Si", "Si"],
        "bonds": [(0, 1, 1), (0, 2, 1)],
        "connect_atom": 0,
        "description": "Secondary silazane nitrogen (-N(Si)(Si))",
    },

    # Acid equivalents (electron acceptors / condensation sites)
    "thiol": {
        "atoms": ["S", "H"],
        "bonds": [(0, 1, 1)],  # S-H
        "connect_atom": 0,  # S connects to backbone
        "description": "Thiol group (-SH), carboxyl analog",
    },
    "silanethiol": {
        "atoms": ["Si", "S", "H"],
        "bonds": [(0, 1, 1), (1, 2, 1)],  # Si-S, S-H
        "connect_atom": 0,  # Si connects to backbone
        "description": "Silanethiol (-Si-SH), silanoic acid analog",
    },
    "dithiocarboxyl": {
        "atoms": ["C", "S", "S", "H"],
        "bonds": [(0, 1, 2), (0, 2, 1), (2, 3, 1)],  # C=S, C-S, S-H
        "connect_atom": 0,  # C connects to backbone
        "description": "Dithiocarboxyl (-CSSH), carboxyl analog",
    },
    "silanoic_sulfur": {
        "atoms": ["Si", "S", "H"],
        "bonds": [(0, 1, 2), (0, 2, 1)],  # Si=S, Si-H
        "connect_atom": 0,
        "description": "Thio-silanoic acid (=Si(S)-SH analog)",
    },
    "borane_acid": {
        "atoms": ["B", "H", "H"],
        "bonds": [(0, 1, 1), (0, 2, 1)],  # B-H, B-H
        "connect_atom": 0,
        "description": "Borane acid analog (-BH2), Lewis acid site",
    },

    # Aluminum-based functional groups
    "alumino_silyl": {
        "atoms": ["Al", "Si", "H"],
        "bonds": [(0, 1, 1), (0, 2, 1)],  # Al-Si, Al-H
        "connect_atom": 0,
        "description": "Aluminosilyl group (-Al(H)-Si<), Lewis acidic site",
    },
    "alumino_amine": {
        "atoms": ["Al", "N", "H"],
        "bonds": [(0, 1, 1), (1, 2, 1)],  # Al-N, N-H
        "connect_atom": 0,
        "description": "Alumino-amine (-Al-N(H)-), electron-pair donor",
    },
    "alumino_thiol": {
        "atoms": ["Al", "S", "H"],
        "bonds": [(0, 1, 1), (1, 2, 1)],  # Al-S, S-H
        "connect_atom": 0,
        "description": "Alumino-thiol (-Al-SH-), Lewis acid-thiolate",
    },

    # Side chain / R-group fragments
    "fluoro_silyl": {
        "atoms": ["Si", "F"],
        "bonds": [(0, 1, 1)],  # Si-F
        "connect_atom": 0,
        "description": "Fluorosilyl group (-SiF), electron-withdrawing",
    },
    "phosphino_silyl": {
        "atoms": ["Si", "P", "H", "H"],
        "bonds": [(0, 1, 1), (1, 2, 1), (1, 3, 1)],  # Si-P, P-H, P-H
        "connect_atom": 0,
        "description": "Phosphino-silyl side chain",
    },
    "boranyl_silyl": {
        "atoms": ["Si", "B", "H"],
        "bonds": [(0, 1, 1), (1, 2, 1)],  # Si-B, B-H
        "connect_atom": 0,
        "description": "Boranyl-silyl side chain",
    },
}

# ─── Backbone Templates ──────────────────────────────────────────────────────
# Aluminosilazane backbones dominate (analogous to terrestrial pyrosilicates)
# Si and Al are the primary framework atoms

BACKBONE_TEMPLATES = {
    "aluminosilazane_chain": {
        "pattern": ["Si", "Al", "N", "Si", "Al", "N"],
        "bonds": [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)],
        "description": "Aluminosilazane (Si-Al-N-Si-Al-N) primary backbone",
    },
    "silazane_chain": {
        "pattern": ["Si", "N", "Si", "N"],
        "bonds": [(0, 1, 1), (1, 2, 1), (2, 3, 1)],
        "description": "Silazane (-Si-N-Si-N-) backbone",
    },
    "aluminosilazane_short": {
        "pattern": ["Al", "N", "Si", "N"],
        "bonds": [(0, 1, 1), (1, 2, 1), (2, 3, 1)],
        "description": "Short aluminosilazane (Al-N-Si-N) backbone",
    },
    "sila_phosphorus_chain": {
        "pattern": ["Si", "P", "Si", "N"],
        "bonds": [(0, 1, 1), (1, 2, 1), (2, 3, 1)],
        "description": "Si-P-Si-N mixed backbone with phosphine",
    },
    "sila_boron_chain": {
        "pattern": ["Si", "B", "Si", "N"],
        "bonds": [(0, 1, 1), (1, 2, 1), (2, 3, 1)],
        "description": "Si-B-Si-N boron-integrated backbone",
    },
    "sila_sulfur_chain": {
        "pattern": ["Si", "S", "Si", "N"],
        "bonds": [(0, 1, 1), (1, 2, 1), (2, 3, 1)],
        "description": "Si-S-Si-N thio-backbone",
    },
    "aluminosilazane_sulfur": {
        "pattern": ["Al", "S", "Si", "N"],
        "bonds": [(0, 1, 1), (1, 2, 1), (2, 3, 1)],
        "description": "Al-S-Si-N thio-aluminosilazane backbone",
    },
}

# ─── Metal Coordination Templates ────────────────────────────────────────────

METAL_COORDINATION_TEMPLATES = {
    "tetrahedral_Si4": {
        "metal": None,  # filled dynamically
        "ligand_atoms": ["Si", "Si", "S", "N"],
        "geometry": "tetrahedral",
        "bond_lengths": {"Si": 2.30, "S": 2.20, "N": 2.00},
    },
    "tetrahedral_Si2S2": {
        "metal": None,
        "ligand_atoms": ["Si", "Si", "S", "S"],
        "geometry": "tetrahedral",
        "bond_lengths": {"Si": 2.30, "S": 2.20},
    },
    "tetrahedral_Si2NP": {
        "metal": None,
        "ligand_atoms": ["Si", "Si", "N", "P"],
        "geometry": "tetrahedral",
        "bond_lengths": {"Si": 2.30, "N": 2.00, "P": 2.25},
    },
    "octahedral_Si4N2": {
        "metal": None,
        "ligand_atoms": ["Si", "Si", "Si", "Si", "N", "N"],
        "geometry": "octahedral",
        "bond_lengths": {"Si": 2.45, "N": 2.10},
    },
    "octahedral_Si3S2N": {
        "metal": None,
        "ligand_atoms": ["Si", "Si", "Si", "S", "S", "N"],
        "geometry": "octahedral",
        "bond_lengths": {"Si": 2.45, "S": 2.30, "N": 2.10},
    },
    "octahedral_Si2S2N2": {
        "metal": None,
        "ligand_atoms": ["Si", "Si", "S", "S", "N", "N"],
        "geometry": "octahedral",
        "bond_lengths": {"Si": 2.45, "S": 2.30, "N": 2.10},
    },
    "octahedral_Al_Si_S": {
        "metal": None,
        "ligand_atoms": ["Al", "Si", "Si", "S", "S", "N"],
        "geometry": "octahedral",
        "bond_lengths": {"Al": 2.20, "Si": 2.45, "S": 2.30, "N": 2.10},
    },
    "square_planar_Si2N2": {
        "metal": None,
        "ligand_atoms": ["Si", "Si", "N", "N"],
        "geometry": "square_planar",
        "bond_lengths": {"Si": 2.20, "N": 1.95},
    },
    "square_planar_Si2SN": {
        "metal": None,
        "ligand_atoms": ["Si", "Si", "S", "N"],
        "geometry": "square_planar",
        "bond_lengths": {"Si": 2.20, "S": 2.10, "N": 1.95},
    },
}


# ─── Geometry Builder ────────────────────────────────────────────────────────

def build_tetrahedral_coords(center: np.ndarray, bond_length: float = 2.3) -> np.ndarray:
    """Generate 4 ligand positions in tetrahedral geometry around center."""
    # Tetrahedral vertices
    verts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * bond_length
    return verts + center


def build_octahedral_coords(center: np.ndarray, bond_length: float = 2.4) -> np.ndarray:
    """Generate 6 ligand positions in octahedral geometry around center."""
    verts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=float)
    verts = verts * bond_length
    return verts + center


def build_square_planar_coords(center: np.ndarray, bond_length: float = 2.2) -> np.ndarray:
    """Generate 4 ligand positions in square planar geometry around center."""
    verts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
    ], dtype=float)
    verts = verts * bond_length
    return verts + center


GEOMETRY_BUILDERS = {
    "tetrahedral": build_tetrahedral_coords,
    "octahedral": build_octahedral_coords,
    "square_planar": build_square_planar_coords,
    "trigonal_prismatic": build_octahedral_coords,  # approximate
}


# ─── Molecular Graph Generator ───────────────────────────────────────────────

class MetallosiliconGenerator:
    """
    Generates candidate metallosilicon amino acid molecular graphs
    under strict elemental and structural constraints for a prosilicon
    world where life is based on aluminosilicate chemistry.

    Prosilicon World Constraints:
    - O < 1% atomic fraction (effectively zero)
    - No -COOH groups
    - Must include Al and Si (primary framework, ~28% and ~8% crustal abundance)
    - Secondary framework: N, H, S, P, B (significant abundance)
    - Trace transition metals: Fe, Ni, Ti, Mo (scarce, optional)
    - Minimal carbon, minimal fluorine

    Dominant Backbone: Al-N-Si-N aluminosilazane chains
    (analogous to terrestrial silicate/pyrosilicate frameworks)

    Functional Group Analogs:
    - Amine equivalents: phosphine (-PH2), silazane (-Si-N<), alumino-amine (-Al-N<)
    - Acid equivalents: thiol (-SH), thio-silanoic (-SiSH), alumino-thiol (-AlSH)
    - Lewis acid sites: Al coordination complexes, borane (-BH2)
    """

    def __init__(
        self,
        n_candidates: int = 10000,
        max_atoms: int = 30,
        min_atoms: int = 8,
        oxygen_fraction_max: float = 0.01,
        carbon_fraction_max: float = 0.05,
        seed: int = 42,
        solvent: str = "liquid_ammonia",
        temperature_K: float = 195.0,  # boiling point of NH3 at 1atm
    ):
        self.n_candidates = n_candidates
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms
        self.oxygen_fraction_max = oxygen_fraction_max
        self.carbon_fraction_max = carbon_fraction_max
        self.rng = np.random.RandomState(seed)
        self.solvent = solvent
        self.temperature_K = temperature_K
        self.formula_counters: Dict[str, int] = {}

        # Statistics tracking
        self.generation_stats = {
            "total_generated": 0,
            "passed_constraints": 0,
            "failed_oxygen": 0,
            "failed_carbon": 0,
            "failed_framework": 0,  # must have Al or Si framework
            "failed_backbone": 0,
        }

    def _format_formula(self, atom_types: List[str]) -> str:
        """Create a condensed formula label in a fixed metallosilicon order."""
        counts = Counter(atom_types)
        parts = []
        for elem in FORMULA_ORDER:
            n = counts.get(elem, 0)
            if n <= 0:
                continue
            parts.append(elem if n == 1 else f"{elem}{n}")

        for elem in sorted(counts):
            if elem in FORMULA_ORDER:
                continue
            n = counts.get(elem, 0)
            if n > 0:
                parts.append(elem if n == 1 else f"{elem}{n}")

        return "".join(parts)

    def _format_display_name(
        self,
        metal: Optional[str],
        backbone_key: str,
        amine_fg: str,
        acid_fg: str,
        side_chain: str,
    ) -> str:
        """Create a readable chemistry-style name for reports and plots."""
        metal_name = METAL_DISPLAY.get(metal, str(metal))
        backbone_name = BACKBONE_DISPLAY.get(backbone_key, backbone_key.replace("_", " "))
        amine_name = FUNCTIONAL_GROUP_DISPLAY.get(amine_fg, amine_fg.replace("_", " "))
        acid_name = FUNCTIONAL_GROUP_DISPLAY.get(acid_fg, acid_fg.replace("_", " "))
        side_name = FUNCTIONAL_GROUP_DISPLAY.get(side_chain, side_chain.replace("_", " "))
        parts = [metal_name, backbone_name, amine_name, acid_name, side_name]
        return " ".join(part for part in parts if part).strip()

    def _next_candidate_label(self, formula: str) -> str:
        """Build a deterministic candidate label from the formula plus isomer index."""
        index = self.formula_counters.get(formula, 0) + 1
        self.formula_counters[formula] = index
        return f"MSA-{formula}-i{index:02d}"

    def _random_element(self, weight_backbone: float = 0.85) -> str:
        """
        Sample a random element with prosilicon-world abundance weights.

        In a prosilicon world:
        - Si (~28%) and Al (~8%) are the primary framework - dominant
        - N, H, S, P, B are secondary (~2-8% each)
        - Fe, Ni, Ti, Mo are trace transition metals (<0.1% each)
        - C and F are minimal contaminants
        """
        if self.rng.random() < weight_backbone:
            # Primary + secondary framework elements
            # Si and Al dominate like in Earth's crust
            weights = [0.32, 0.18,  # Si, Al (primary framework)
                       0.12, 0.10, 0.08, 0.06, 0.04,  # N, H, S, P, B, F (secondary)
                       0.02]  # C (trace)
            elements = ["Si", "Al", "N", "H", "S", "P", "B", "F", "C"]
        else:
            # Trace transition metals (scarce but functional)
            weights = [0.35, 0.30, 0.20, 0.15]  # Fe, Ni, Ti, Mo
            elements = METAL_ELEMENTS

        return self.rng.choice(elements, p=np.array(weights) / sum(weights))

    def _select_backbone(self) -> str:
        """
        Randomly select a backbone template.
        Aluminosilazane backbones dominate (like terrestrial pyrosilicates).
        """
        templates = list(BACKBONE_TEMPLATES.keys())
        # Heavy weight on aluminosilazane backbone
        weights = [0.35, 0.25, 0.15, 0.10, 0.05, 0.05, 0.05]
        return self.rng.choice(templates, p=np.array(weights) / sum(weights))

    def _select_metal(self) -> str:
        """Randomly select a transition metal (trace, optional)."""
        metals = METAL_ELEMENTS
        weights = [0.35, 0.30, 0.20, 0.15]  # Fe most common, Mo least
        return self.rng.choice(metals, p=np.array(weights) / sum(weights))

    def _select_coordination(self, metal: str) -> Tuple[str, dict]:
        """Select coordination template appropriate for the metal."""
        metal_info = METAL_COORDINATION[metal]
        preferred_cn = metal_info["preferred_cn"]

        if preferred_cn == 4:
            candidates = ["tetrahedral_Si4", "tetrahedral_Si2S2",
                         "tetrahedral_Si2NP", "square_planar_Si2N2",
                         "square_planar_Si2SN"]
        else:  # cn = 6
            candidates = ["octahedral_Si4N2", "octahedral_Si3S2N",
                          "octahedral_Si2S2N2"]

        choice = self.rng.choice(candidates)
        template = deepcopy(METAL_COORDINATION_TEMPLATES[choice])
        template["metal"] = metal
        return choice, template

    def _select_amine_analog(self) -> str:
        """Select amine-equivalent functional group (electron pair donors)."""
        options = ["phosphine", "silazane_amine", "secondary_silazane",
                   "alumino_amine", "alumino_silyl"]
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        return self.rng.choice(options, p=np.array(weights) / sum(weights))

    def _select_acid_analog(self) -> str:
        """Select acid-equivalent functional group (Lewis acid sites)."""
        options = ["thiol", "silanethiol", "dithiocarboxyl", "silanoic_sulfur",
                   "borane_acid", "alumino_thiol"]
        weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
        return self.rng.choice(options, p=np.array(weights) / sum(weights))

    def _select_side_chain(self) -> str:
        """Select R-group / side chain (electron-withdrawing or donating)."""
        options = ["fluoro_silyl", "phosphino_silyl", "boranyl_silyl",
                   "alumino_silyl"]
        weights = [0.30, 0.30, 0.25, 0.15]
        return self.rng.choice(options, p=np.array(weights) / sum(weights))

    def _add_hydrogen_saturation(self, atom_types: List[str], adjacency: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """
        Add hydrogen atoms to satisfy valence constraints.
        Al: 3 (tetrahedral with framework), Si: 4, N: 3, P: 3, B: 3, S: 2, F: 1
        """
        valence_targets = {
            "Si": 4, "Al": 3, "N": 3, "P": 3, "B": 3, "S": 2, "F": 1, "C": 4,
            "Fe": 6, "Ni": 4, "Ti": 6, "Mo": 6, "H": 1,
        }

        n_atoms = len(atom_types)
        new_atom_types = list(atom_types)
        new_adj = adjacency.copy()

        for i in range(n_atoms):
            elem = atom_types[i]
            if elem == "H" or elem in METAL_ELEMENTS:
                continue

            target_valence = valence_targets.get(elem, 4)
            current_valence = int(adjacency[i].sum())

            # Add H atoms to fill valence
            h_needed = max(0, target_valence - current_valence)
            for _ in range(h_needed):
                h_idx = len(new_atom_types)
                new_atom_types.append("H")
                new_adj = np.pad(new_adj, ((0, 1), (0, 1)), constant_values=0)
                new_adj[i, h_idx] = 1
                new_adj[h_idx, i] = 1

        return new_atom_types, new_adj

    def _compute_positions(
        self,
        atom_types: List[str],
        adjacency: np.ndarray,
        metal_index: Optional[int],
        coord_template: Optional[dict],
    ) -> np.ndarray:
        """
        Generate approximate 3D coordinates for the molecular graph.
        Uses template-based geometry for metal center and distance-based
        placement for backbone and functional groups.
        """
        n_atoms = len(atom_types)
        positions = np.zeros((n_atoms, 3))

        # If no metal, place first Si/Al atom at origin and build from there
        if metal_index is None:
            # Find first Si or Al atom as reference
            ref_idx = 0
            for i, elem in enumerate(atom_types):
                if elem in ("Si", "Al"):
                    ref_idx = i
                    break
            positions[ref_idx] = np.array([0.0, 0.0, 0.0])
            # Place remaining atoms along a chain
            z = 1.74  # Si-N bond length
            for i in range(n_atoms):
                if i != ref_idx:
                    positions[i] = np.array([0.0, 0.0, z])
                    z += 1.74
            return positions

        # Place metal at origin
        positions[metal_index] = np.array([0.0, 0.0, 0.0])

        # Build coordination geometry around metal
        geom = coord_template["geometry"]
        bond_lengths = coord_template["bond_lengths"]

        # Find metal-coordinated atoms
        metal_neighbors = np.where(adjacency[metal_index] > 0)[0]
        n_neighbors = len(metal_neighbors)

        if geom == "tetrahedral" and n_neighbors >= 4:
            coords = build_tetrahedral_coords(
                positions[metal_index],
                bond_length=METAL_COORDINATION[atom_types[metal_index]]["typical_bond_len"]
            )
            for i, nb in enumerate(metal_neighbors[:4]):
                elem = atom_types[nb]
                bl = bond_lengths.get(elem, 2.3)
                direction = coords[i] - positions[metal_index]
                direction = direction / (np.linalg.norm(direction) + 1e-8) * bl
                positions[nb] = positions[metal_index] + direction

        elif geom == "octahedral" and n_neighbors >= 6:
            coords = build_octahedral_coords(
                positions[metal_index],
                bond_length=METAL_COORDINATION[atom_types[metal_index]]["typical_bond_len"]
            )
            for i, nb in enumerate(metal_neighbors[:6]):
                elem = atom_types[nb]
                bl = bond_lengths.get(elem, 2.4)
                direction = coords[i] - positions[metal_index]
                direction = direction / (np.linalg.norm(direction) + 1e-8) * bl
                positions[nb] = positions[metal_index] + direction

        elif geom == "square_planar" and n_neighbors >= 4:
            coords = build_square_planar_coords(
                positions[metal_index],
                bond_length=METAL_COORDINATION[atom_types[metal_index]]["typical_bond_len"]
            )
            for i, nb in enumerate(metal_neighbors[:4]):
                elem = atom_types[nb]
                bl = bond_lengths.get(elem, 2.2)
                direction = coords[i] - positions[metal_index]
                direction = direction / (np.linalg.norm(direction) + 1e-8) * bl
                positions[nb] = positions[metal_index] + direction

        # Place remaining atoms using distance-based propagation
        placed = set([metal_index])
        placed.update(metal_neighbors.tolist())

        # BFS from metal center
        from collections import deque
        queue = deque(metal_neighbors.tolist())

        typical_bond = {
            "Si": 2.35, "Al": 2.30, "N": 1.90, "H": 1.50, "S": 2.15,
            "P": 2.25, "B": 2.00, "F": 1.70, "C": 1.85,
        }

        while queue:
            current = queue.popleft()
            neighbors = np.where(adjacency[current] > 0)[0]

            for nb in neighbors:
                if nb in placed:
                    continue

                # Place atom at typical bond distance from current
                elem = atom_types[nb]
                parent_elem = atom_types[current]
                bl = typical_bond.get(elem, 2.0)

                # Random direction avoiding clashes
                direction = self.rng.randn(3)
                direction = direction / (np.linalg.norm(direction) + 1e-8) * bl
                positions[nb] = positions[current] + direction

                placed.add(nb)
                queue.append(nb)

        # Add small random perturbation to break symmetry
        positions += self.rng.randn(*positions.shape) * 0.05

        return positions

    def generate_single(self, candidate_id: Optional[str] = None) -> Optional[MolecularGraph]:
        """
        Generate a single metallosilicon amino acid candidate.

        Returns:
            MolecularGraph or None if constraints violated
        """
        self.generation_stats["total_generated"] += 1

        # 1. Select backbone template
        backbone_key = self._select_backbone()
        backbone = BACKBONE_TEMPLATES[backbone_key]

        # 2. Select metal (optional - prosilicon life can exist without transition metals)
        metal = self._select_metal()
        coord_key, coord_template = None, None
        if metal is not None:
            coord_key, coord_template = self._select_coordination(metal)

        # 3. Select functional groups
        amine_fg = self._select_amine_analog()
        acid_fg = self._select_acid_analog()
        side_chain = self._select_side_chain()

        # 4. Build atom list starting from backbone
        atom_types = list(backbone["pattern"])
        n_backbone = len(atom_types)

        # 5. Add metal center (if metal is not None)
        metal_index = None
        if metal is not None:
            metal_index = len(atom_types)
            atom_types.append(metal)

        # 6. Build adjacency matrix
        n_initial = len(atom_types)
        adjacency = np.zeros((n_initial, n_initial), dtype=int)

        # Backbone bonds
        for i, j, bt in backbone["bonds"]:
            adjacency[i, j] = bt
            adjacency[j, i] = bt

        # Connect backbone to metal if metal is present
        if metal is not None and metal_index is not None and coord_template is not None:
            # Connect backbone Si/Al atoms to metal
            backbone_si_indices = [i for i in range(n_backbone) if atom_types[i] in ("Si", "Al")]
            for si_idx in backbone_si_indices[:2]:  # connect up to 2 Si/Al to metal
                adjacency[si_idx, metal_index] = 1
                adjacency[metal_index, si_idx] = 1

            # 7. Add coordination ligands
            ligand_atoms = coord_template["ligand_atoms"]
            n_ligands_needed = len(ligand_atoms)

            # Count existing metal bonds
            existing_metal_bonds = int(adjacency[metal_index].sum())
            ligands_to_add = max(0, n_ligands_needed - existing_metal_bonds)

            ligand_start_idx = len(atom_types)
            for i in range(ligands_to_add):
                lig_elem = ligand_atoms[min(i, len(ligand_atoms) - 1)]
                lig_idx = len(atom_types)
                atom_types.append(lig_elem)
                # Expand adjacency
                new_size = len(atom_types)
                new_adj = np.zeros((new_size, new_size), dtype=int)
                new_adj[:adjacency.shape[0], :adjacency.shape[1]] = adjacency
                adjacency = new_adj
                adjacency[metal_index, lig_idx] = 1
                adjacency[lig_idx, metal_index] = 1

        # 8. Add amine functional group
        amine = FUNCTIONAL_GROUPS[amine_fg]
        amine_connect = len(atom_types)
        amine_start = len(atom_types)
        for atom in amine["atoms"]:
            atom_types.append(atom)
        # Expand adjacency
        new_size = len(atom_types)
        new_adj = np.zeros((new_size, new_size), dtype=int)
        new_adj[:adjacency.shape[0], :adjacency.shape[1]] = adjacency
        adjacency = new_adj

        # Amine internal bonds
        for i, j, bt in amine["bonds"]:
            adj_i = amine_start + i
            adj_j = amine_start + j
            adjacency[adj_i, adj_j] = bt
            adjacency[adj_j, adj_i] = bt

        # Connect amine to backbone (connect to first Si or N)
        connect_to = 0  # first backbone atom
        adjacency[connect_to, amine_start + amine["connect_atom"]] = 1
        adjacency[amine_start + amine["connect_atom"], connect_to] = 1

        # 9. Add acid functional group
        acid = FUNCTIONAL_GROUPS[acid_fg]
        acid_start = len(atom_types)
        for atom in acid["atoms"]:
            atom_types.append(atom)
        # Expand adjacency
        new_size = len(atom_types)
        new_adj = np.zeros((new_size, new_size), dtype=int)
        new_adj[:adjacency.shape[0], :adjacency.shape[1]] = adjacency
        adjacency = new_adj

        # Acid internal bonds
        for i, j, bt in acid["bonds"]:
            adj_i = acid_start + i
            adj_j = acid_start + j
            adjacency[adj_i, adj_j] = bt
            adjacency[adj_j, adj_i] = bt

        # Connect acid to backbone (connect to last backbone atom)
        connect_to = n_backbone - 1
        adjacency[connect_to, acid_start + acid["connect_atom"]] = 1
        adjacency[acid_start + acid["connect_atom"], connect_to] = 1

        # 10. Add side chain
        side = FUNCTIONAL_GROUPS[side_chain]
        side_start = len(atom_types)
        for atom in side["atoms"]:
            atom_types.append(atom)
        # Expand adjacency
        new_size = len(atom_types)
        new_adj = np.zeros((new_size, new_size), dtype=int)
        new_adj[:adjacency.shape[0], :adjacency.shape[1]] = adjacency
        adjacency = new_adj

        # Side chain internal bonds
        for i, j, bt in side["bonds"]:
            adj_i = side_start + i
            adj_j = side_start + j
            adjacency[adj_i, adj_j] = bt
            adjacency[adj_j, adj_i] = bt

        # Connect side chain to a backbone Si or Al
        si_al_targets = [i for i in range(n_backbone) if atom_types[i] in ("Si", "Al")]
        if si_al_targets:
            connect_to = si_al_targets[len(si_al_targets) // 2]
        else:
            connect_to = n_backbone // 2
        adjacency[connect_to, side_start + side["connect_atom"]] = 1
        adjacency[side_start + side["connect_atom"], connect_to] = 1

        # 11. Hydrogen saturation
        atom_types, adjacency = self._add_hydrogen_saturation(atom_types, adjacency)

        # 12. Constraint validation
        n_atoms = len(atom_types)
        if n_atoms > self.max_atoms or n_atoms < self.min_atoms:
            return None

        # Check oxygen fraction
        o_count = atom_types.count("O")
        if o_count / n_atoms > self.oxygen_fraction_max:
            self.generation_stats["failed_oxygen"] += 1
            return None

        # Check carbon fraction
        c_count = atom_types.count("C")
        if c_count / n_atoms > self.carbon_fraction_max:
            self.generation_stats["failed_carbon"] += 1
            return None

        # Check primary framework: must have Al or Si (or both)
        # In prosilicon world, Al is as important as Si (crustal abundance)
        has_al = "Al" in atom_types
        has_si = "Si" in atom_types
        if not (has_al or has_si):
            self.generation_stats["failed_framework"] += 1
            return None

        # 13. Compute 3D positions
        positions = self._compute_positions(
            atom_types, adjacency, metal_index, coord_template
        )
        formula = self._format_formula(atom_types)
        display_name = self._format_display_name(
            metal, backbone_key, amine_fg, acid_fg, side_chain
        )
        if candidate_id is None:
            candidate_id = self._next_candidate_label(formula)

        # 14. Build node features
        n_features = len(ALL_ELEMENTS) + 4
        node_features = np.zeros((n_atoms, n_features))
        for i, elem in enumerate(atom_types):
            idx = ALL_ELEMENTS.index(elem) if elem in ALL_ELEMENTS else 0
            node_features[i, idx] = 1.0
            props = ELEMENT_PROPERTIES.get(elem, [1, 1.0, 50, 1])
            node_features[i, len(ALL_ELEMENTS):] = [
                props[0] / 42.0, props[1] / 4.0,
                props[2] / 160.0, props[3] / 10.0,
            ]

        # 15. Build edge features
        edge_indices = np.array(np.where(adjacency > 0))
        n_bonds = edge_indices.shape[1]
        edge_features = np.zeros((n_bonds, 8 + 3))
        for b in range(n_bonds):
            i, j = edge_indices[:, b]
            bt = int(adjacency[i, j]) - 1
            bt = max(0, min(bt, 7))
            edge_features[b, bt] = 1.0
            dist = np.linalg.norm(positions[i] - positions[j])
            edge_features[b, 8] = dist / 3.0
            edge_features[b, 9] = 0.0  # angle placeholder
            edge_features[b, 10] = 1.0  # cos placeholder

        self.generation_stats["passed_constraints"] += 1

        # Detect functional groups
        has_phosphine = amine_fg == "phosphine" or side_chain == "phosphino_silyl"
        has_silazane = amine_fg in ("silazane_amine", "secondary_silazane")
        has_thiol = acid_fg in ("thiol", "silanethiol", "dithiocarboxyl")
        has_silanoic = acid_fg in ("silanoic_sulfur", "silanethiol")

        # Generate SMILES-like notation
        smiles_parts = []
        if metal:
            smiles_parts.append(f"[{metal}]")
        smiles_parts.append(backbone["pattern"][0] + backbone["pattern"][1])
        smiles_parts.append(f"({amine_fg})")
        if len(backbone["pattern"]) > 2:
            smiles_parts.append(backbone["pattern"][2] + backbone["pattern"][3])
        smiles_parts.append(f"({acid_fg})")
        smiles_parts.append(f"({side_chain})")
        smiles = "".join(smiles_parts)

        return MolecularGraph(
            node_features=node_features,
            edge_features=edge_features,
            adjacency=adjacency,
            atom_types=atom_types,
            positions=positions,
            metal_center=metal,  # None if no transition metal
            metal_index=metal_index,  # None if no transition metal
            smiles=f"{display_name} [{formula}] :: {smiles}",
            candidate_id=candidate_id,
            formula=formula,
            display_name=display_name,
            oxygen_fraction=o_count / n_atoms,
            backbone_template=backbone_key,
            amine_analog=amine_fg,
            acid_analog=acid_fg,
            side_chain=side_chain,
            has_phosphine=has_phosphine,
            has_silazane=has_silazane,
            has_thiol=has_thiol,
            has_silanoic=has_silanoic,
        )

    def generate_batch(self, n: Optional[int] = None) -> List[MolecularGraph]:
        """
        Generate a batch of candidate molecular graphs.

        Args:
            n: Number of candidates to generate (default: self.n_candidates)

        Returns:
            List of valid MolecularGraph objects
        """
        if n is None:
            n = self.n_candidates

        candidates = []
        attempts = 0
        max_attempts = n * 5  # allow for constraint failures

        while len(candidates) < n and attempts < max_attempts:
            graph = self.generate_single()
            if graph is not None:
                candidates.append(graph)
            attempts += 1

        logger.info(
            f"Generated {len(candidates)} valid candidates from {attempts} attempts. "
            f"Stats: {self.generation_stats}"
        )

        return candidates

    def generate_diverse_set(
        self,
        n: int = 10000,
        diversity_threshold: float = 0.3,
    ) -> List[MolecularGraph]:
        """
        Generate a diverse set of candidates by ensuring structural variation.

        Uses composition-based fingerprinting to avoid duplicates and
        ensure diversity across metal types, backbone templates, and
        functional group combinations.
        """
        # Systematic enumeration of key combinations
        systematic_candidates = []

        # Systematic enumeration: include metals (optional) + all backbone types
        # Metal choices: None (framework only) + each metal
        metal_choices = [None] + METAL_ELEMENTS
        for metal in metal_choices:
            for backbone_key in BACKBONE_TEMPLATES:
                for amine_key in ["phosphine", "silazane_amine", "secondary_silazane",
                                  "alumino_amine", "alumino_silyl"]:
                    for acid_key in ["thiol", "silanethiol", "dithiocarboxyl",
                                     "silanoic_sulfur", "borane_acid", "alumino_thiol"]:
                        for side_key in ["fluoro_silyl", "phosphino_silyl",
                                        "boranyl_silyl", "alumino_silyl"]:
                            # Override random selections for systematic generation
                            old_select = self._select_metal
                            self._select_metal = lambda m=metal: m
                            old_backbone = self._select_backbone
                            self._select_backbone = lambda b=backbone_key: b
                            old_amine = self._select_amine_analog
                            self._select_amine_analog = lambda a=amine_key: a
                            old_acid = self._select_acid_analog
                            self._select_acid_analog = lambda a=acid_key: a
                            old_side = self._select_side_chain
                            self._select_side_chain = lambda s=side_key: s

                            graph = self.generate_single()
                            if graph is not None:
                                systematic_candidates.append(graph)

                            # Restore random selections
                            self._select_metal = old_select
                            self._select_backbone = old_backbone
                            self._select_amine_analog = old_amine
                            self._select_acid_analog = old_acid
                            self._select_side_chain = old_side

        n_systematic = len(systematic_candidates)
        logger.info(f"Generated {n_systematic} systematic candidates")

        # Fill remaining with random generation
        remaining = max(0, n - n_systematic)
        random_candidates = self.generate_batch(remaining)

        all_candidates = systematic_candidates + random_candidates

        # Deduplicate by composition fingerprint
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            # Composition fingerprint: sorted element counts
            from collections import Counter
            comp = tuple(sorted(Counter(c.atom_types).items()))
            if comp not in seen:
                seen.add(comp)
                unique_candidates.append(c)

        logger.info(
            f"Final diverse set: {len(unique_candidates)} unique candidates "
            f"from {len(all_candidates)} total"
        )

        return unique_candidates[:n]

    def export_candidates(
        self,
        candidates: List[MolecularGraph],
        output_path: str,
    ):
        """Export generated candidates to JSON."""
        data = []
        for c in candidates:
            data.append({
                "candidate_id": c.candidate_id,
                "formula": c.formula,
                "display_name": c.display_name,
                "atom_types": c.atom_types,
                "positions": c.positions.tolist(),
                "adjacency": c.adjacency.tolist(),
                "metal_center": c.metal_center,
                "metal_index": c.metal_index,
                "backbone_template": c.backbone_template,
                "amine_analog": c.amine_analog,
                "acid_analog": c.acid_analog,
                "side_chain": c.side_chain,
                "smiles": c.smiles,
                "oxygen_fraction": c.oxygen_fraction,
                "has_phosphine": c.has_phosphine,
                "has_silazane": c.has_silazane,
                "has_thiol": c.has_thiol,
                "has_silanoic": c.has_silanoic,
                "n_atoms": len(c.atom_types),
            })

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data)} candidates to {output_path}")

    def get_generation_report(self) -> Dict:
        """Return statistics about the generation process."""
        return deepcopy(self.generation_stats)
