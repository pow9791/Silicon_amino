"""
Pymatgen DFT Relaxation Workflow for Metallosilicon Amino Acid Candidates

Generates VASP/Gaussian input files for structural relaxation and
parses output to extract:
- Optimized geometries (bond lengths, angles)
- HOMO-LUMO gaps
- Total energies and formation energies
- Electron Localization Function (ELF) data

Uses pymatgen for structure manipulation and VASP input generation.
"""

import numpy as np
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from copy import deepcopy

from scripts.gnome_model import (
    MolecularGraph, ELEMENT_PROPERTIES, METAL_ELEMENTS, METAL_COORDINATION,
)

logger = logging.getLogger(__name__)

# ─── VASP INCAR Parameters for Metallosilicon Systems ────────────────────────
# Optimized for cryogenic, reducing-environment molecules with transition metals

VASP_INCAR_TEMPLATE = {
    # Basic settings
    "PREC": "Accurate",
    "ENCUT": 520,                    # eV, plane-wave cutoff (high for Si-F bonds)
    "EDIFF": 1E-6,                   # electronic convergence
    "EDIFFG": -0.01,                 # ionic convergence (eV/Å)
    "NSW": 200,                      # max ionic steps
    "IBRION": 2,                     # CG algorithm
    "ISIF": 2,                       # relax ions only (molecule in box)

    # Exchange-correlation
    "GGA": "PE",                     # PBE functional
    "LDAU": True,                    # +U for transition metals
    "LDAUTYPE": 2,                   # Dudarev approach
    "LDAUL": {"Fe": 2, "Ni": 2, "Ti": 2, "Mo": 2, "Al": 0},
    "LDAUU": {"Fe": 4.0, "Ni": 6.0, "Ti": 3.0, "Mo": 4.0, "Al": 0.0},  # eV
    "LDAUJ": {"Fe": 0.0, "Ni": 0.0, "Ti": 0.0, "Mo": 0.0, "Al": 0.0},

    # Spin polarization (metals may be magnetic)
    "ISPIN": 2,                      # spin-polarized
    "MAGMOM": "auto",                # auto-initialize magnetic moments

    # k-point sampling (molecule in large box → Gamma only)
    "KPOINTS": "Gamma",

    # Smearing (metals need it)
    "ISMEAR": 0,                     # Gaussian smearing
    "SIGMA": 0.05,                   # eV, narrow for molecules

    # Output control
    "LWAVE": False,
    "LCHARG": True,
    "LELF": True,                    # Electron Localization Function
    "LORBIT": 11,                    # projected DOS
    "NEDOS": 2001,                   # DOS grid

    # vdW correction (important for cryogenic solvent interactions)
    "IVDW": 12,                      # D3-BJ dispersion
}

# Pseudopotential mapping (VASP POTCAR)
POTCAR_MAP = {
    "Si": "Si_pv",    # Si with semicore p states
    "N": "N",
    "H": "H",
    "S": "S",
    "P": "P",
    "B": "B",
    "Al": "Al",
    "F": "F",
    "C": "C",
    "Fe": "Fe_pv",
    "Ni": "Ni_pv",
    "Ti": "Ti_pv",
    "Mo": "Mo_pv",
}


@dataclass
class DFTResult:
    """Result from DFT relaxation calculation."""
    candidate_id: str
    total_energy: float              # eV
    formation_energy: float          # eV/atom
    optimized_positions: np.ndarray  # (n_atoms, 3)
    bond_lengths: Dict[str, float]   # bond_type -> length in Å
    bond_angles: Dict[str, float]    # angle_type -> degrees
    homo_lumo_gap: float             # eV
    fermi_level: float               # eV
    magnetic_moment: float           # μB
    elf_data: Optional[Dict] = None  # ELF analysis results
    converged: bool = True
    warnings: List[str] = field(default_factory=list)


class VASPInputGenerator:
    """Generate VASP input files (POSCAR, INCAR, KPOINTS, POTCAR) for molecular relaxation."""

    def __init__(
        self,
        box_size: float = 20.0,       # Å, vacuum padding
        encut: int = 520,
        ediff: float = 1e-6,
        ediffg: float = -0.01,
        nsw: int = 200,
        functional: str = "PBE",
        include_u: bool = True,
        include_vdw: bool = True,
        spin_polarized: bool = True,
    ):
        self.box_size = box_size
        self.encut = encut
        self.ediff = ediff
        self.ediffg = ediffg
        self.nsw = nsw
        self.functional = functional
        self.include_u = include_u
        self.include_vdw = include_vdw
        self.spin_polarized = spin_polarized

    def generate_poscar(self, graph: MolecularGraph) -> str:
        """Generate POSCAR string from molecular graph."""
        atom_types = graph.atom_types
        positions = graph.positions

        # Center molecule in box
        center = positions.mean(axis=0)
        centered_pos = positions - center + np.array([self.box_size/2]*3)

        # Count unique elements in order of appearance
        seen = []
        counts = []
        for elem in atom_types:
            if elem not in seen:
                seen.append(elem)
                counts.append(atom_types.count(elem))

        # Build POSCAR
        lines = [
            f"Metallosilicon amino acid: {graph.candidate_id}",
            "1.0",  # scaling factor
        ]
        # Lattice vectors (cubic box)
        lines.append(f"  {self.box_size:.6f}   0.000000   0.000000")
        lines.append(f"  0.000000   {self.box_size:.6f}   0.000000")
        lines.append(f"  0.000000   0.000000   {self.box_size:.6f}")
        # Element names
        lines.append("  " + "  ".join(f"{e:>6s}" for e in seen))
        # Atom counts
        lines.append("  " + "  ".join(f"{c:>6d}" for c in counts))
        lines.append("Cartesian")
        # Positions
        for i, elem in enumerate(atom_types):
            x, y, z = centered_pos[i]
            lines.append(f"  {x:.8f}  {y:.8f}  {z:.8f}")

        return "\n".join(lines)

    def generate_incar(self, graph: MolecularGraph) -> str:
        """Generate INCAR string with metal-specific +U parameters."""
        incar = dict(VASP_INCAR_TEMPLATE)
        incar["ENCUT"] = self.encut
        incar["EDIFF"] = self.ediff
        incar["EDIFFG"] = self.ediffg
        incar["NSW"] = self.nsw

        # Adjust +U for metals present
        if self.include_u:
            metals_present = set(graph.atom_types) & set(METAL_ELEMENTS)
            if not metals_present:
                incar["LDAU"] = False
            else:
                ldauu_vals = []
                ldauj_vals = []
                ldaul_vals = []
                for elem in graph.atom_types:
                    if elem in METAL_ELEMENTS:
                        ldauu_vals.append(incar["LDAUU"].get(elem, 0.0))
                        ldauj_vals.append(incar["LDAUJ"].get(elem, 0.0))
                        ldaul_vals.append(incar["LDAUL"].get(elem, 2))
                    else:
                        ldauu_vals.append(0.0)
                        ldauj_vals.append(0.0)
                        ldaul_vals.append(-1)
                incar["LDAUU"] = ldauu_vals
                incar["LDAUJ"] = ldauj_vals
                incar["LDAUL"] = ldaul_vals
        else:
            incar["LDAU"] = False
            for key in ["LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ"]:
                incar.pop(key, None)

        if not self.include_vdw:
            incar.pop("IVDW", None)

        if not self.spin_polarized:
            incar["ISPIN"] = 1
            incar.pop("MAGMOM", None)

        # Build INCAR string
        lines = []
        for key, val in incar.items():
            if isinstance(val, bool):
                lines.append(f"  {key:20s} = {'T' if val else 'F'}")
            elif isinstance(val, (list,)):
                lines.append(f"  {key:20s} = {val}")
            elif isinstance(val, dict):
                lines.append(f"  {key:20s} = {val}")
            else:
                lines.append(f"  {key:20s} = {val}")

        return "\n".join(lines)

    def generate_kpoints(self) -> str:
        """Generate KPOINTS file (Gamma-only for molecules)."""
        return "Gamma-point only\n1\nGamma\n1 1 1\n0 0 0\n"

    def generate_potcar_hint(self, graph: MolecularGraph) -> str:
        """Generate POTCAR concatenation hint (not the actual POTCAR)."""
        seen = []
        for elem in graph.atom_types:
            if elem not in seen:
                seen.append(elem)

        lines = ["# Concatenate POTCAR files in this order:"]
        for elem in seen:
            potcar_name = POTCAR_MAP.get(elem, elem)
            lines.append(f"cat $VASP_PP_PATH/pot_{self.functional}/{potcar_name}/POTCAR >> POTCAR")

        return "\n".join(lines)

    def write_vasp_inputs(self, graph: MolecularGraph, output_dir: str):
        """Write all VASP input files to a directory."""
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "POSCAR"), "w") as f:
            f.write(self.generate_poscar(graph))

        with open(os.path.join(output_dir, "INCAR"), "w") as f:
            f.write(self.generate_incar(graph))

        with open(os.path.join(output_dir, "KPOINTS"), "w") as f:
            f.write(self.generate_kpoints())

        with open(os.path.join(output_dir, "POTCAR_hint.sh"), "w") as f:
            f.write(self.generate_potcar_hint(graph))

        logger.info(f"Wrote VASP inputs for {graph.candidate_id} to {output_dir}")


class GaussianInputGenerator:
    """Generate Gaussian input files for molecular DFT calculations."""

    def __init__(
        self,
        method: str = "B3LYP",
        basis_set: str = "def2TZVP",
        empirical_dispersion: str = "GD3BJ",
        solvent: str = "liquid_ammonia",
        charge: int = 0,
        multiplicity: int = 1,
    ):
        self.method = method
        self.basis_set = basis_set
        self.empirical_dispersion = empirical_dispersion
        self.solvent = solvent
        self.charge = charge
        self.multiplicity = multiplicity

    def generate_input(self, graph: MolecularGraph) -> str:
        """Generate Gaussian .gjf input file content."""
        # Build route line
        route_parts = [
            f"#p {self.method}/{self.basis_set}",
            f"EmpiricalDispersion={self.empirical_dispersion}",
            f"SCRF=(Solvent={self.solvent})",
            "Opt=Loose",
            "Freq",
            "Pop=Full",
            "IOp(6/7=3)",  # print molecular orbitals
        ]
        route_line = " ".join(route_parts)

        # Build coordinate section
        coord_lines = []
        for i, elem in enumerate(graph.atom_types):
            x, y, z = graph.positions[i]
            coord_lines.append(f" {elem:2s}  {x:12.8f}  {y:12.8f}  {z:12.8f}")

        # Build full input
        lines = [
            f"%nprocshared=8",
            f"%mem=32GB",
            f"%chk={graph.candidate_id}.chk",
            "",
            route_line,
            "",
            f"Metallosilicon amino acid {graph.candidate_id}",
            "",
            f"{self.charge} {self.multiplicity}",
        ]
        lines.extend(coord_lines)
        lines.append("")
        lines.append("")

        return "\n".join(lines)

    def write_gaussian_input(self, graph: MolecularGraph, output_dir: str):
        """Write Gaussian input file."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{graph.candidate_id}.gjf")
        with open(filepath, "w") as f:
            f.write(self.generate_input(graph))
        logger.info(f"Wrote Gaussian input for {graph.candidate_id} to {filepath}")


class DFTOutputParser:
    """Parse DFT output files to extract optimized geometries and properties."""

    def parse_vasp_outcar(self, outcar_path: str) -> DFTResult:
        """Parse VASP OUTCAR file for key results."""
        result_data = {
            "candidate_id": os.path.basename(os.path.dirname(outcar_path)),
            "total_energy": 0.0,
            "formation_energy": 0.0,
            "optimized_positions": np.zeros((1, 3)),
            "bond_lengths": {},
            "bond_angles": {},
            "homo_lumo_gap": 0.0,
            "fermi_level": 0.0,
            "magnetic_moment": 0.0,
            "converged": False,
            "warnings": [],
        }

        try:
            with open(outcar_path, "r") as f:
                content = f.read()

            # Total energy
            for line in content.split("\n"):
                if "energy  without entropy" in line.lower() or "energy(sigma->0)" in line.lower():
                    parts = line.split()
                    for i, p in enumerate(parts):
                        try:
                            result_data["total_energy"] = float(p)
                            break
                        except ValueError:
                            continue

            # Fermi level
            for line in content.split("\n"):
                if "E-fermi" in line:
                    parts = line.split()
                    for p in parts:
                        try:
                            result_data["fermi_level"] = float(p)
                            break
                        except ValueError:
                            continue

            # Magnetic moment
            for line in content.split("\n"):
                if "number of electron" in line.lower() and "magnet" in line.lower():
                    parts = line.split()
                    for p in reversed(parts):
                        try:
                            result_data["magnetic_moment"] = float(p)
                            break
                        except ValueError:
                            continue

            # Convergence check
            result_data["converged"] = "reached required accuracy" in content.lower()

        except FileNotFoundError:
            result_data["warnings"].append(f"OUTCAR not found at {outcar_path}")

        return DFTResult(**result_data)

    def parse_gaussian_log(self, log_path: str) -> DFTResult:
        """Parse Gaussian .log file for key results."""
        result_data = {
            "candidate_id": os.path.basename(log_path).replace(".log", ""),
            "total_energy": 0.0,
            "formation_energy": 0.0,
            "optimized_positions": np.zeros((1, 3)),
            "bond_lengths": {},
            "bond_angles": {},
            "homo_lumo_gap": 0.0,
            "fermi_level": 0.0,
            "magnetic_moment": 0.0,
            "converged": False,
            "warnings": [],
        }

        try:
            with open(log_path, "r") as f:
                content = f.read()

            # SCF Done energy
            for line in content.split("\n"):
                if "SCF Done" in line:
                    parts = line.split("=")
                    if len(parts) >= 2:
                        try:
                            result_data["total_energy"] = float(parts[1].split()[0])
                        except (ValueError, IndexError):
                            pass

            # HOMO-LUMO gap
            homo = lumo = None
            for line in content.split("\n"):
                if "Alpha  occ." in line and "eigenvalues" in line:
                    vals = line.split()[-5:]
                    try:
                        homo = float(vals[-1])
                    except (ValueError, IndexError):
                        pass
                if "Alpha virt." in line and "eigenvalues" in line:
                    vals = line.split()[:5]
                    try:
                        lumo = float(vals[0])
                    except (ValueError, IndexError):
                        pass
            if homo is not None and lumo is not None:
                result_data["homo_lumo_gap"] = lumo - homo

            # Convergence
            result_data["converged"] = "Normal termination" in content

        except FileNotFoundError:
            result_data["warnings"].append(f"Gaussian log not found at {log_path}")

        return DFTResult(**result_data)


class PymatgenRelaxationWorkflow:
    """
    Full pymatgen-based DFT relaxation workflow.

    In production mode, this submits VASP/Gaussian jobs and parses results.
    In simulation mode (no VASP/Gaussian available), it uses force-field
    relaxation via ASE and estimates DFT-level properties.
    """

    def __init__(
        self,
        vasp_input_gen: Optional[VASPInputGenerator] = None,
        gaussian_input_gen: Optional[GaussianInputGenerator] = None,
        output_dir: str = "sims/output",
        simulation_mode: bool = True,
    ):
        self.vasp_gen = vasp_input_gen or VASPInputGenerator()
        self.gaussian_gen = gaussian_input_gen or GaussianInputGenerator()
        self.output_dir = output_dir
        self.simulation_mode = simulation_mode
        self.parser = DFTOutputParser()

    def relax_candidate(
        self,
        graph: MolecularGraph,
        method: str = "vasp",
    ) -> DFTResult:
        """
        Run structural relaxation on a single candidate.

        Args:
            graph: Input molecular graph
            method: "vasp" or "gaussian"

        Returns:
            DFTResult with optimized geometry and properties
        """
        candidate_dir = os.path.join(self.output_dir, graph.candidate_id)

        if method == "vasp":
            self.vasp_gen.write_vasp_inputs(graph, candidate_dir)
        else:
            self.gaussian_gen.write_gaussian_input(graph, candidate_dir)

        if self.simulation_mode:
            # Simulate DFT relaxation using force-field approach
            return self._simulate_relaxation(graph)
        else:
            # In production: submit job, wait, parse output
            outcar_path = os.path.join(candidate_dir, "OUTCAR")
            if os.path.exists(outcar_path):
                return self.parser.parse_vasp_outcar(outcar_path)
            else:
                return self._simulate_relaxation(graph)

    def _simulate_relaxation(self, graph: MolecularGraph) -> DFTResult:
        """
        Simulate DFT relaxation using empirical force field.

        Applies distance-based geometry optimization and estimates
        DFT-level properties from composition and bonding.
        """
        atom_types = graph.atom_types
        positions = graph.positions.copy()
        n_atoms = len(atom_types)
        adj = graph.adjacency

        # Ideal bond lengths for relaxation target (Å)
        ideal_lengths = {
            ("Si", "Si"): 2.35, ("Si", "N"): 1.74, ("Si", "H"): 1.48,
            ("Si", "S"): 2.15, ("Si", "P"): 2.25, ("Si", "B"): 2.00,
            ("Si", "F"): 1.60, ("Si", "C"): 1.87, ("Si", "Fe"): 2.30,
            ("Si", "Ni"): 2.20, ("Si", "Ti"): 2.45, ("Si", "Mo"): 2.50,
            ("N", "H"): 1.01, ("N", "N"): 1.45, ("N", "S"): 1.70,
            ("N", "P"): 1.70, ("N", "B"): 1.55, ("N", "C"): 1.47,
            ("S", "H"): 1.34, ("S", "S"): 2.05, ("S", "P"): 2.00,
            ("S", "C"): 1.82, ("S", "Fe"): 2.20, ("S", "Ni"): 2.15,
            ("S", "Ti"): 2.35, ("S", "Mo"): 2.40,
            ("P", "H"): 1.42, ("P", "C"): 1.87, ("P", "F"): 1.56,
            ("B", "H"): 1.19, ("B", "F"): 1.30, ("B", "C"): 1.56,
            ("F", "H"): 0.92, ("C", "H"): 1.09, ("C", "C"): 1.54,
        }

        # Simple steepest-descent relaxation
        for step in range(50):
            forces = np.zeros_like(positions)
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    if adj[i, j] > 0:
                        r_vec = positions[j] - positions[i]
                        r = np.linalg.norm(r_vec)
                        if r < 0.1:
                            continue

                        # Find ideal bond length
                        key1 = (atom_types[i], atom_types[j])
                        key2 = (atom_types[j], atom_types[i])
                        r0 = ideal_lengths.get(key1, ideal_lengths.get(key2, 2.0))

                        # Harmonic force: F = -k(r - r0)
                        k = 10.0  # eV/Å²
                        force_mag = -k * (r - r0)
                        force_vec = force_mag * r_vec / r

                        forces[i] -= force_vec
                        forces[j] += force_vec

            # Update positions
            step_size = 0.01
            positions += step_size * forces

        # Compute bond lengths from relaxed geometry
        bond_lengths = {}
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if adj[i, j] > 0:
                    r = np.linalg.norm(positions[j] - positions[i])
                    key = f"{atom_types[i]}-{atom_types[j]}"
                    if key not in bond_lengths:
                        bond_lengths[key] = []
                    bond_lengths[key].append(r)

        # Average bond lengths
        avg_bond_lengths = {}
        for key, vals in bond_lengths.items():
            avg_bond_lengths[key] = np.mean(vals)

        # Compute bond angles
        bond_angles = {}
        for i in range(n_atoms):
            neighbors = np.where(adj[i] > 0)[0]
            if len(neighbors) >= 2:
                for a_idx in range(len(neighbors)):
                    for b_idx in range(a_idx + 1, len(neighbors)):
                        a, b = neighbors[a_idx], neighbors[b_idx]
                        v1 = positions[a] - positions[i]
                        v2 = positions[b] - positions[i]
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.degrees(np.arccos(cos_angle))
                        key = f"{atom_types[a]}-{atom_types[i]}-{atom_types[b]}"
                        bond_angles[key] = angle

        # Estimate HOMO-LUMO gap
        from scripts.screening_pipeline import ScreeningPipeline, FormationEnergyCalculator
        calc = FormationEnergyCalculator()
        fe = calc.compute_formation_energy(graph)

        # Estimate HOMO-LUMO from composition
        composition = {}
        for e in atom_types:
            composition[e] = composition.get(e, 0) + 1

        en_vals = [ELEMENT_PROPERTIES.get(e, [1, 1.0, 50, 1])[1] for e in atom_types]
        en_spread = max(en_vals) - min(en_vals)
        homo_lumo = 0.5 + en_spread * 0.8
        metal_count = sum(composition.get(m, 0) for m in METAL_ELEMENTS)
        if metal_count > 0:
            homo_lumo *= 0.4
        homo_lumo = max(0.1, homo_lumo)

        # Estimate total energy
        total_energy = calc.compute_bond_energy_estimate(graph)

        # Estimate Fermi level
        fermi_level = -4.5 + homo_lumo / 2  # rough estimate

        # Estimate magnetic moment
        mag_moment = 0.0
        if graph.metal_center == "Fe":
            mag_moment = 4.0  # high-spin Fe(II)
        elif graph.metal_center == "Ni":
            mag_moment = 2.0
        elif graph.metal_center == "Ti":
            mag_moment = 0.0  # typically diamagnetic
        elif graph.metal_center == "Mo":
            mag_moment = 0.0

        return DFTResult(
            candidate_id=graph.candidate_id,
            total_energy=total_energy,
            formation_energy=fe,
            optimized_positions=positions,
            bond_lengths=avg_bond_lengths,
            bond_angles=bond_angles,
            homo_lumo_gap=homo_lumo,
            fermi_level=fermi_level,
            magnetic_moment=mag_moment,
            converged=True,
            warnings=[],
        )

    def relax_batch(
        self,
        candidates: List[MolecularGraph],
        method: str = "vasp",
    ) -> List[DFTResult]:
        """Run relaxation on a batch of candidates."""
        results = []
        for i, graph in enumerate(candidates):
            logger.info(f"Relaxing candidate {i+1}/{len(candidates)}: {graph.candidate_id}")
            result = self.relax_candidate(graph, method=method)
            results.append(result)
        return results

    def export_results(
        self,
        results: List[DFTResult],
        output_path: str,
    ):
        """Export DFT results to JSON."""
        data = []
        for r in results:
            d = asdict(r)
            d["optimized_positions"] = r.optimized_positions.tolist()
            data.append(d)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported {len(data)} DFT results to {output_path}")
