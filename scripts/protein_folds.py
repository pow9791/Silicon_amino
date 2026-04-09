"""
Metallosilicon Protein Fold Simulator

Simulates common protein fold topologies adapted for silico-protein polymers
built from metallosilicon amino acid monomers.

Key differences from terrestrial protein folding:
- Backbone: Si-N-Si-N (silazane) instead of C-N-C-N (peptide)
- Cross-linking: Metal coordination bonds instead of disulfide bridges
- Hydrogen bonding: Replaced by Si...N dative interactions
- Solvent: Liquid NH3/CH4 instead of H2O
- Byproduct: H2S/PH3 instead of H2O during condensation

Fold classes implemented:
1. α-Helix analog (Si-N helix with metal core)
2. β-Sheet analog (Si-N sheet with S-bridge crosslinks)
3. β-Barrel analog (metal-coordinated barrel)
4. TIM Barrel analog (alternating helix-sheet with metal pocket)
5. Coiled-coil analog (metal-wire helix bundle)
"""

import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import Counter

from scripts.gnome_model import (
    MolecularGraph, METAL_ELEMENTS, METAL_COORDINATION, ELEMENT_PROPERTIES,
)

logger = logging.getLogger(__name__)

# ─── Silico-Protein Polymer Parameters ───────────────────────────────────────

# Bond lengths for silazane backbone (Å)
SILO_BACKBONE_BONDS = {
    "Si-N": 1.74,    # silazane bond
    "N-Si": 1.74,
    "Si-Si": 2.35,   # weaker than C-C
    "Si-P": 2.25,    # phosphine bridge
    "Si-S": 2.15,    # thiol bridge
    "Si-B": 2.00,    # borane bridge
    "Si-Al": 2.50,   # aluminosilane bond
    "Al-N": 1.95,    # alumino-amine bond
    "Al-S": 2.20,    # alumino-thiol bond
    "Al-H": 1.60,    # Al-H bond
}

# Dihedral angles for silico-protein secondary structure
SILO_DIHEDRALS = {
    "alpha_helix": {
        "phi": -57.8,    # N-Si-N-Si dihedral (analog of protein phi)
        "psi": -47.0,    # Si-N-Si-N dihedral (analog of protein psi)
        "omega": 180.0,  # planar Si-N partial double bond character
        "rise_per_residue": 1.50,  # Å (vs 1.5 Å for α-helix)
        "residues_per_turn": 3.6,
        "radius": 2.30,   # Å (slightly larger than carbon helix)
    },
    "beta_sheet": {
        "phi": -139.0,
        "psi": 135.0,
        "omega": 180.0,
        "rise_per_residue": 3.20,  # extended
        "strand_spacing": 4.70,    # Å between strands
    },
    "beta_barrel": {
        "phi": -120.0,
        "psi": 130.0,
        "omega": 175.0,
        "n_strands": 8,
        "barrel_radius": 8.0,
        "barrel_height": 20.0,
    },
    "coiled_coil": {
        "phi": -80.0,
        "psi": -40.0,
        "omega": 180.0,
        "supercoil_pitch": 140.0,  # Å
        "n_helices": 3,
        "helix_radius": 2.50,
    },
    "tim_barrel": {
        "phi_alpha": -57.0,
        "psi_alpha": -47.0,
        "phi_beta": -130.0,
        "psi_beta": 140.0,
        "n_repeats": 8,
    },
}

# Metal coordination cross-link parameters
METAL_CROSSLINK = {
    "Fe": {"bond_len_Si": 2.30, "bond_len_S": 2.20, "bond_len_N": 2.00,
           "geometry": "tetrahedral", "magnetic_moment": 4.0},
    "Ni": {"bond_len_Si": 2.20, "bond_len_S": 2.15, "bond_len_N": 1.95,
           "geometry": "square_planar", "magnetic_moment": 2.0},
    "Ti": {"bond_len_Si": 2.45, "bond_len_S": 2.35, "bond_len_N": 2.10,
           "geometry": "octahedral", "magnetic_moment": 0.0},
    "Mo": {"bond_len_Si": 2.50, "bond_len_S": 2.40, "bond_len_N": 2.15,
           "geometry": "octahedral", "magnetic_moment": 0.0},
    "Al": {"bond_len_Si": 2.45, "bond_len_S": 2.25, "bond_len_N": 2.00,
           "geometry": "tetrahedral", "magnetic_moment": 0.0},
}


@dataclass
class SilicoProteinFold:
    """A folded silico-protein structure."""
    fold_id: str
    fold_type: str                    # alpha_helix, beta_sheet, etc.
    n_residues: int
    metal_center: str
    backbone_atoms: List[str]          # element symbols along backbone
    positions: np.ndarray             # (n_atoms, 3)
    dihedrals: Dict[str, float]        # phi, psi, omega
    metal_positions: np.ndarray        # positions of metal cross-links
    coordination_geometry: str
    estimated_stability: float         # 0-1
    homo_lumo_gap: float              # eV (narrow = conductive)
    conductivity_estimate: float      # S/cm
    solvent_compatibility: Dict[str, float]
    warnings: List[str] = field(default_factory=list)


class SilicoProteinBuilder:
    """
    Build silico-protein polymer chains from metallosilicon amino acid monomers
    and fold them into common protein topology analogs.
    """

    def __init__(
        self,
        solvent: str = "liquid_ammonia",
        temperature_K: float = 195.0,
        metal: str = "Fe",
    ):
        self.solvent = solvent
        self.temperature_K = temperature_K
        self.metal = metal

    def _build_backbone_chain(
        self,
        n_residues: int,
        backbone_type: str = "silazane",
    ) -> Tuple[List[str], np.ndarray]:
        """
        Build an extended silazane backbone chain.

        Backbone pattern: Si-N-Si-N-Si-N-...
        With H atoms saturating valences.
        """
        atoms = []
        positions = []

        # Place backbone atoms along z-axis
        z = 0.0
        for i in range(n_residues):
            # Each residue: Si-N pair
            si_pos = np.array([0.0, 0.0, z])
            n_pos = np.array([0.0, 0.0, z + SILO_BACKBONE_BONDS["Si-N"]])

            atoms.append("Si")
            positions.append(si_pos)
            atoms.append("N")
            positions.append(n_pos)

            # Add H to Si (2 H per Si in backbone)
            atoms.extend(["H", "H"])
            h1_pos = si_pos + np.array([1.0, 0.0, 0.3])
            h2_pos = si_pos + np.array([-0.5, 0.866, 0.3])
            positions.extend([h1_pos, h2_pos])

            # Add H to N (1 H per N in backbone)
            atoms.append("H")
            nh_pos = n_pos + np.array([0.0, 0.0, -0.3])
            positions.append(nh_pos)

            z += SILO_BACKBONE_BONDS["Si-N"] * 2  # one full residue

        return atoms, np.array(positions)

    def _apply_alpha_helix(
        self,
        atoms: List[str],
        positions: np.ndarray,
        params: Dict,
    ) -> np.ndarray:
        """
        Apply α-helix transformation to extended backbone.

        Wraps the linear chain into a helical coil with
        Si-N backbone dihedrals matching silazane helix parameters.
        """
        n_atoms = len(atoms)
        helix_positions = positions.copy()

        phi = np.radians(params["phi"])
        psi = np.radians(params["psi"])
        rise = params["rise_per_residue"]
        radius = params["radius"]
        residues_per_turn = params["residues_per_turn"]

        # Find backbone Si and N atoms
        backbone_indices = [i for i, a in enumerate(atoms) if a in ("Si", "N")]
        n_backbone = len(backbone_indices)

        # Apply helical transformation
        angle_per_atom = 2 * np.pi / (residues_per_turn * 2)  # 2 atoms per residue

        for idx, atom_idx in enumerate(backbone_indices):
            angle = idx * angle_per_atom
            z = idx * rise / 2  # half rise per atom (Si-N pair)

            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            helix_positions[atom_idx] = np.array([x, y, z])

        # Reposition H atoms relative to their parent backbone atoms
        for i, atom in enumerate(atoms):
            if atom == "H":
                # Find nearest backbone atom
                min_dist = float('inf')
                nearest_backbone = 0
                for bi in backbone_indices:
                    d = np.linalg.norm(positions[i] - positions[bi])
                    if d < min_dist:
                        min_dist = d
                        nearest_backbone = bi

                # Place H relative to backbone atom
                offset = positions[i] - positions[nearest_backbone]
                offset_norm = np.linalg.norm(offset)
                if offset_norm > 0:
                    offset = offset / offset_norm * 1.0  # normalize to 1Å
                helix_positions[i] = helix_positions[nearest_backbone] + offset

        return helix_positions

    def _apply_beta_sheet(
        self,
        atoms: List[str],
        positions: np.ndarray,
        params: Dict,
        n_strands: int = 4,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Apply β-sheet transformation: create multiple extended strands
        with S-bridge crosslinks between them.
        """
        strand_spacing = params["strand_spacing"]
        rise = params["rise_per_residue"]

        all_atoms = []
        all_positions = []

        for strand_idx in range(n_strands):
            strand_atoms = list(atoms)
            strand_positions = positions.copy()

            # Offset each strand in x-direction
            x_offset = strand_idx * strand_spacing

            # Apply zigzag pattern for extended conformation
            for i, atom in enumerate(strand_atoms):
                if atom in ("Si", "N"):
                    # Alternate up-down for zigzag
                    z_sign = 1 if strand_atoms.index(atom) % 2 == 0 else -1
                    strand_positions[i][0] += x_offset
                    strand_positions[i][1] += z_sign * 0.5

            all_atoms.extend(strand_atoms)

            # Add S-bridge atoms between strands
            if strand_idx < n_strands - 1:
                n_si = sum(1 for a in strand_atoms if a == "Si")
                for si_i in range(0, n_si, 2):  # every other Si
                    si_idx = [j for j, a in enumerate(strand_atoms) if a == "Si"][si_i]
                    s_pos = strand_positions[si_idx].copy()
                    s_pos[0] += strand_spacing / 2
                    s_pos[1] += 0.3
                    all_atoms.append("S")
                    all_positions.append(s_pos)

                    # Add H to S
                    h_pos = s_pos.copy()
                    h_pos[1] += 1.0
                    all_atoms.append("H")
                    all_positions.append(h_pos)

            all_positions.extend(strand_positions.tolist())

        return all_atoms, np.array(all_positions)

    def _apply_beta_barrel(
        self,
        atoms: List[str],
        positions: np.ndarray,
        params: Dict,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Apply β-barrel transformation: arrange strands in a cylinder.
        Metal centers sit inside the barrel for electron transport.
        """
        n_strands = params["n_strands"]
        barrel_radius = params["barrel_radius"]
        barrel_height = params["barrel_height"]

        all_atoms = []
        all_positions = []

        strand_height = barrel_height / n_strands

        for strand_idx in range(n_strands):
            angle = 2 * np.pi * strand_idx / n_strands
            x_center = barrel_radius * np.cos(angle)
            y_center = barrel_radius * np.sin(angle)

            # Create strand along z-axis, then rotate to barrel position
            strand_atoms = list(atoms)
            strand_positions = positions.copy()

            for i in range(len(strand_positions)):
                z = strand_positions[i][2]
                # Scale z to fit barrel height
                z_scaled = z / (positions[:, 2].max() + 1e-8) * barrel_height

                # Position on barrel surface
                strand_positions[i] = np.array([
                    x_center + 0.5 * np.cos(angle + np.pi/2),
                    y_center + 0.5 * np.sin(angle + np.pi/2),
                    z_scaled,
                ])

            all_atoms.extend(strand_atoms)
            all_positions.extend(strand_positions.tolist())

        # Add metal centers inside barrel
        n_metals = max(1, n_strands // 4)
        for m_idx in range(n_metals):
            z_metal = barrel_height * (m_idx + 1) / (n_metals + 1)
            all_atoms.append(self.metal)
            all_positions.append(np.array([0.0, 0.0, z_metal]))

        return all_atoms, np.array(all_positions)

    def _apply_coiled_coil(
        self,
        atoms: List[str],
        positions: np.ndarray,
        params: Dict,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Apply coiled-coil transformation: bundle of helices
        with metal-wire core for electron transport.
        """
        n_helices = params["n_helices"]
        supercoil_pitch = params["supercoil_pitch"]
        helix_radius = params["helix_radius"]

        all_atoms = []
        all_positions = []

        # Bundle radius
        bundle_radius = helix_radius * 2.5

        for helix_idx in range(n_helices):
            # Apply alpha helix first
            helix_atoms = list(atoms)
            helix_positions = self._apply_alpha_helix(
                helix_atoms, positions.copy(), SILO_DIHEDRALS["alpha_helix"]
            )

            # Offset helix center
            angle = 2 * np.pi * helix_idx / n_helices
            cx = bundle_radius * np.cos(angle)
            cy = bundle_radius * np.sin(angle)

            # Apply supercoil (slow rotation of helix axis)
            for i in range(len(helix_positions)):
                z = helix_positions[i][2]
                supercoil_angle = 2 * np.pi * z / supercoil_pitch

                # Rotate around z-axis
                x = helix_positions[i][0]
                y = helix_positions[i][1]
                cos_a = np.cos(supercoil_angle)
                sin_a = np.sin(supercoil_angle)
                new_x = x * cos_a - y * sin_a + cx
                new_y = x * sin_a + y * cos_a + cy

                helix_positions[i] = np.array([new_x, new_y, z])

            all_atoms.extend(helix_atoms)
            all_positions.extend(helix_positions.tolist())

        # Add central metal wire
        max_z = max(p[2] for p in all_positions) if all_positions else 10.0
        n_metal_atoms = int(max_z / 2.5) + 1
        for m in range(n_metal_atoms):
            all_atoms.append(self.metal)
            all_positions.append(np.array([0.0, 0.0, m * 2.5]))

        return all_atoms, np.array(all_positions)

    def _apply_tim_barrel(
        self,
        atoms: List[str],
        positions: np.ndarray,
        params: Dict,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Apply TIM barrel transformation: alternating α-helices and β-strands
        forming a barrel with metal-coordinated active site pocket.
        """
        n_repeats = params["n_repeats"]
        all_atoms = []
        all_positions = []

        barrel_radius = 8.0
        helix_radius = 2.3

        for repeat_idx in range(n_repeats):
            angle = 2 * np.pi * repeat_idx / n_repeats

            # β-strand on barrel surface
            strand_atoms = list(atoms[:len(atoms)//2])  # first half
            strand_positions = positions[:len(atoms)//2].copy()

            x_center = barrel_radius * np.cos(angle)
            y_center = barrel_radius * np.sin(angle)

            for i in range(len(strand_positions)):
                z = strand_positions[i][2]
                z_scaled = z / (positions[:len(atoms)//2, 2].max() + 1e-8) * 15.0
                strand_positions[i] = np.array([
                    x_center + 0.3 * np.cos(angle + np.pi/2),
                    y_center + 0.3 * np.sin(angle + np.pi/2),
                    z_scaled,
                ])

            all_atoms.extend(strand_atoms)
            all_positions.extend(strand_positions.tolist())

            # α-helix connecting to next strand
            helix_atoms = list(atoms[len(atoms)//2:])
            helix_positions = self._apply_alpha_helix(
                helix_atoms, positions[len(atoms)//2:].copy(),
                SILO_DIHEDRALS["alpha_helix"]
            )

            # Position helix outside barrel
            next_angle = 2 * np.pi * (repeat_idx + 0.5) / n_repeats
            hx = (barrel_radius + 5.0) * np.cos(next_angle)
            hy = (barrel_radius + 5.0) * np.sin(next_angle)

            for i in range(len(helix_positions)):
                helix_positions[i][0] += hx
                helix_positions[i][1] += hy

            all_atoms.extend(helix_atoms)
            all_positions.extend(helix_positions.tolist())

        # Metal center in active site pocket
        all_atoms.append(self.metal)
        all_positions.append(np.array([0.0, 0.0, 7.5]))

        return all_atoms, np.array(all_positions)

    def build_fold(
        self,
        n_residues: int,
        fold_type: str,
        metal: Optional[str] = None,
    ) -> SilicoProteinFold:
        """
        Build a complete silico-protein fold.

        Args:
            n_residues: Number of amino acid residues
            fold_type: One of 'alpha_helix', 'beta_sheet', 'beta_barrel',
                       'coiled_coil', 'tim_barrel'
            metal: Transition metal for cross-links

        Returns:
            SilicoProteinFold with 3D structure
        """
        if metal is None:
            metal = self.metal

        # Build extended backbone
        atoms, positions = self._build_backbone_chain(n_residues)

        # Apply fold transformation
        if fold_type == "alpha_helix":
            folded_positions = self._apply_alpha_helix(
                atoms, positions, SILO_DIHEDRALS["alpha_helix"]
            )
            folded_atoms = atoms
        elif fold_type == "beta_sheet":
            folded_atoms, folded_positions = self._apply_beta_sheet(
                atoms, positions, SILO_DIHEDRALS["beta_sheet"]
            )
        elif fold_type == "beta_barrel":
            folded_atoms, folded_positions = self._apply_beta_barrel(
                atoms, positions, SILO_DIHEDRALS["beta_barrel"]
            )
        elif fold_type == "coiled_coil":
            folded_atoms, folded_positions = self._apply_coiled_coil(
                atoms, positions, SILO_DIHEDRALS["coiled_coil"]
            )
        elif fold_type == "tim_barrel":
            folded_atoms, folded_positions = self._apply_tim_barrel(
                atoms, positions, SILO_DIHEDRALS["tim_barrel"]
            )
        else:
            raise ValueError(f"Unknown fold type: {fold_type}")

        # Estimate properties
        metal_info = METAL_CROSSLINK.get(metal, METAL_CROSSLINK["Fe"])
        homo_lumo = self._estimate_fold_homo_lumo(folded_atoms, metal)
        conductivity = self._estimate_conductivity(homo_lumo, metal)
        stability = self._estimate_fold_stability(folded_atoms, fold_type)
        solvent_compat = self._estimate_solvent_compatibility(folded_atoms)

        # Find metal positions
        metal_indices = [i for i, a in enumerate(folded_atoms) if a in METAL_ELEMENTS]
        if metal_indices:
            metal_positions = folded_positions[metal_indices]
        else:
            metal_positions = np.zeros((1, 3))

        # Warnings
        warnings = []
        si_count = folded_atoms.count("Si")
        if si_count > 0:
            warnings.append(
                "⚠️ STABILITY WARNING: Silico-protein will hydrolyze violently "
                "in H₂O/O₂. Strictly stable only in reducing cryogenic solvents."
            )
        if homo_lumo < 0.5:
            warnings.append(
                f"⚡ CONDUCTIVITY: HOMO-LUMO gap ({homo_lumo:.2f} eV) is narrow. "
                "This silico-protein may function as a biological nanowire "
                "via electron tunneling."
            )

        fold_id = f"SiProtein-{fold_type}-{metal}-{n_residues}res"

        return SilicoProteinFold(
            fold_id=fold_id,
            fold_type=fold_type,
            n_residues=n_residues,
            metal_center=metal,
            backbone_atoms=folded_atoms,
            positions=folded_positions,
            dihedrals=SILO_DIHEDRALS.get(fold_type, {}),
            metal_positions=metal_positions,
            coordination_geometry=metal_info["geometry"],
            estimated_stability=stability,
            homo_lumo_gap=homo_lumo,
            conductivity_estimate=conductivity,
            solvent_compatibility=solvent_compat,
            warnings=warnings,
        )

    def _estimate_fold_homo_lumo(self, atoms: List[str], metal: str) -> float:
        """Estimate HOMO-LUMO gap for folded silico-protein."""
        # Metal centers dramatically narrow the gap
        base_gap = 1.5  # eV for pure silazane chain
        metal_count = sum(1 for a in atoms if a in METAL_ELEMENTS)
        if metal_count > 0:
            base_gap *= 0.3  # metals create mid-gap states
        # Si-Si bonds narrow gap
        si_count = atoms.count("Si")
        base_gap -= si_count * 0.01
        # S atoms provide lone pairs that narrow gap
        s_count = atoms.count("S")
        base_gap -= s_count * 0.005
        return max(0.05, base_gap)

    def _estimate_conductivity(self, homo_lumo: float, metal: str) -> float:
        """
        Estimate electrical conductivity from HOMO-LUMO gap.
        Uses Arrhenius-type relation: σ = σ₀ exp(-Eg/2kT)
        """
        kT = 8.617e-5 * self.temperature_K  # eV
        sigma_0 = 1e4  # S/cm prefactor
        conductivity = sigma_0 * np.exp(-homo_lumo / (2 * kT))
        return min(conductivity, 1e6)  # cap at metallic

    def _estimate_fold_stability(self, atoms: List[str], fold_type: str) -> float:
        """Estimate fold stability score (0-1)."""
        # Helical structures are more stable at low T
        stability_map = {
            "alpha_helix": 0.85,
            "beta_sheet": 0.75,
            "beta_barrel": 0.70,
            "coiled_coil": 0.80,
            "tim_barrel": 0.65,
        }
        base = stability_map.get(fold_type, 0.5)

        # Metal cross-links increase stability
        metal_count = sum(1 for a in atoms if a in METAL_ELEMENTS)
        base += metal_count * 0.05

        # Low temperature increases stability
        if self.temperature_K < 250:
            base += 0.1

        return min(1.0, base)

    def _estimate_solvent_compatibility(self, atoms: List[str]) -> Dict[str, float]:
        """Estimate compatibility with different solvents."""
        composition = Counter(atoms)
        n = len(atoms)
        return {
            "liquid_ammonia": min(1.0, 0.5 + 0.3 * composition.get("N", 0) / n),
            "liquid_methane": min(1.0, 0.4 + 0.2 * (composition.get("Si", 0) + composition.get("H", 0)) / n),
            "liquid_hydrogen_sulfide": min(1.0, 0.5 + 0.3 * composition.get("S", 0) / n),
            "supercritical_ammonia": min(1.0, 0.6 + 0.3 * composition.get("N", 0) / n + 0.1 * composition.get("Al", 0) / n),
        }

    def build_all_folds(
        self,
        n_residues: int = 12,
        metals: Optional[List[str]] = None,
    ) -> List[SilicoProteinFold]:
        """
        Build all fold types for all specified metals.

        Args:
            n_residues: Number of residues per fold
            metals: List of transition metals to use

        Returns:
            List of SilicoProteinFold objects
        """
        if metals is None:
            metals = METAL_ELEMENTS

        fold_types = ["alpha_helix", "beta_sheet", "beta_barrel", "coiled_coil", "tim_barrel"]
        folds = []

        for metal in metals:
            for fold_type in fold_types:
                try:
                    fold = self.build_fold(n_residues, fold_type, metal=metal)
                    folds.append(fold)
                    logger.info(f"Built {fold.fold_id}: {fold.n_residues} residues, "
                               f"HOMO-LUMO={fold.homo_lumo_gap:.3f} eV, "
                               f"σ={fold.conductivity_estimate:.2e} S/cm")
                except Exception as e:
                    logger.warning(f"Failed to build {fold_type} with {metal}: {e}")

        return folds

    def export_folds(self, folds: List[SilicoProteinFold], output_path: str):
        """Export fold structures to JSON."""
        data = []
        for f in folds:
            d = asdict(f)
            d["positions"] = f.positions.tolist()
            d["metal_positions"] = f.metal_positions.tolist()
            data.append(d)

        with open(output_path, "w") as f_out:
            json.dump(data, f_out, indent=2, default=str)

        logger.info(f"Exported {len(data)} fold structures to {output_path}")

    def to_cif(self, fold: SilicoProteinFold) -> str:
        """
        Convert a silico-protein fold to CIF format.

        CIF is the standard crystallographic format and can represent
        molecular structures with arbitrary unit cells.
        """
        atoms = fold.backbone_atoms
        positions = fold.positions

        # Use a large box as unit cell
        box = 50.0
        lines = [
            f"data_{fold.fold_id}",
            f"_cell_length_a    {box:.4f}",
            f"_cell_length_b    {box:.4f}",
            f"_cell_length_c    {box:.4f}",
            f"_cell_angle_alpha  90.000",
            f"_cell_angle_beta   90.000",
            f"_cell_angle_gamma  90.000",
            f"_symmetry_space_group_name_H-M  'P 1'",
            "",
            "loop_",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        ]

        for i, (atom, pos) in enumerate(zip(atoms, positions)):
            # Convert to fractional coordinates
            fx = (pos[0] + box/2) / box
            fy = (pos[1] + box/2) / box
            fz = (pos[2] + box/2) / box
            label = f"{atom}{i+1}"
            lines.append(f"{label:>8s}  {atom:>2s}  {fx:.6f}  {fy:.6f}  {fz:.6f}")

        return "\n".join(lines)

    def to_xyz(self, fold: SilicoProteinFold) -> str:
        """Convert a silico-protein fold to XYZ format."""
        atoms = fold.backbone_atoms
        positions = fold.positions
        n_atoms = len(atoms)

        lines = [str(n_atoms), f"# {fold.fold_id}"]
        for atom, pos in zip(atoms, positions):
            lines.append(f"{atom:>2s}  {pos[0]:12.8f}  {pos[1]:12.8f}  {pos[2]:12.8f}")

        return "\n".join(lines)
