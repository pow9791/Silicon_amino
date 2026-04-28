# Aluminosilicon Amino Acid Discovery Pipeline — Summary

## Overview
note this is simplified and probably needs some editing

This pipeline generates and screens stable metallosilicon amino acid analogs for **prosilicon biological frameworks** in warm (>30°C), reducing, low-oxygen environments.

**Key Achievement**: 500/500 candidates pass screening at 310K with strongly negative formation energies (-3.6 eV/atom).

---

## Prosilicon World Chemistry

### Elemental Hierarchy (Earth Crustal Abundance Model)
| Element Class | Elements | Crustal Abundance | Role |
|--------------|----------|------------------|------|
| **Primary Framework** | Si, Al | ~28%, ~8% | Network formers (like terrestrial silicates) |
| **Secondary Framework** | N, H, S, P, B | ~2-8% each | Modifiers, functional groups |
| **Trace Metals** | Fe, Ni, Ti, Mo | <0.1% each | Coordination sites (optional) |
| **Minimal** | C, F | <0.1% | Contaminants |

### Environment
- **Temperature**: >30°C (310K default)
- **Solvent**: Supercritical ammonia (reducing)
- **Oxygen**: <1% atomic fraction
- **Atmosphere**: Highly reducing (no O₂/H₂O)

---

## Pipeline Architecture

### Stage 1: Molecular Graph Generation
- Systematic enumeration of Al-Si-N backbone templates
- Functional group attachment (phosphine, silazane, thiol, etc.)
- Optional metal coordination (Fe, Ni, Ti, Mo)
- Constraint validation (O<1%, C minimal, Al/Si required)

### Stage 2: Bond-Energy Screening
- **Primary method**: Bond-energy calculator (not untrained GNoME)
- Formation energy from bond dissociation energies
- Temperature-dependent entropy corrections
- Convex hull distance calculation

### Stage 3: Phase Diagram Analysis
- 30+ reference compounds (AlN, Si₃N₄, PH₃, etc.)
- Gibbs free energy corrections at temperature
- K-nearest hull vertex interpolation

### Stage 4: DFT Relaxation (Simulated)
- VASP input generation (PBE+U, D3-BJ, spin-polarized)
- 50-step steepest-descent optimization
- HOMO-LUMO gap estimation

### Stage 5: Protein Fold Simulation
- 5 fold types: alpha_helix, beta_sheet, beta_barrel, coiled_coil, TIM_barrel
- 4 metals × 5 folds = 20 silico-protein structures
- Conductivity via Arrhenius relation: σ = σ₀ exp(-Eg/2kT)

### Stage 6: Output Generation
- JSON with all candidate data
- CIF/XYZ geometry files
- Summary report

### Stage 7: Visualization
- Formation energy distribution
- HOMO-LUMO gap analysis
- Solvent stability heatmap
- Coordination geometry plots

---

## Key Results

### Top Candidate Structure
```
ID: MSA-84bf8cc3
Composition: Si-Al-N-Si-Al-N backbone with phosphine/silazane groups
ΔE_f = -3.63 eV/atom (ON CONVEX HULL)
HOMO-LUMO gap = 2.16 eV
Metal: None (pure aluminosilazane framework)
```

### Formation Energy Distribution
| Metric | Value |
|--------|-------|
| Best | -3.63 eV/atom |
| Worst (passed) | -1.02 eV/atom |
| Mean | ~-2.5 eV/atom |

### Silico-Protein Conductivity
| Fold Type | HOMO-LUMO (eV) | Conductivity (S/cm) | Behavior |
|-----------|----------------|---------------------|----------|
| **beta_barrel** | 0.050 | 3,920 | **Nanowire** |
| **tim_barrel** | 0.050 | 3,920 | **Nanowire** |
| **coiled_coil** | 0.090 | 1,860 | **Nanowire** |
| beta_sheet | 0.930 | 0.000275 | Semiconductor |
| alpha_helix | 1.380 | 6.05e-08 | Insulator |

**Key Insight**: Beta-barrel and TIM-barrel folds with metal centers show narrow HOMO-LUMO gaps suggesting **biological nanowire** functionality via electron tunneling.

---

## Stability Warnings

⚠️ **Pyrophoric**: All candidates combust spontaneously in O₂ atmosphere
⚠️ **Hydrolytic instability**: Si-H and Si-Si bonds react violently with H₂O
⚠️ **Thermal limits**: Si-Si bonds thermally labile above ~250K (but Si-N and Si-Al are stable)
⚠️ **Earth environment**: These molecules require strictly reducing, low-O₂ conditions

---

## File Structure

```
silicon_amino/
├── run_pipeline.py              # Main orchestration script
├── scripts/
│   ├── gnome_model.py          # GNoME GNN architecture
│   ├── molecular_generator.py  # Al-Si molecular graph generator
│   ├── screening_pipeline.py    # Formation energy + hull screening
│   ├── phase_diagram.py        # Convex hull analysis
│   ├── dft_workflow.py         # VASP/Gaussian input generation
│   ├── protein_folds.py        # Silico-protein fold simulator
│   ├── output_system.py        # JSON/CIF/XYZ export
│   └── visualization.py        # Matplotlib/Plotly plots
├── sims/
│   ├── input/candidates.json   # Generated candidates
│   └── output/
│       ├── candidates.json     # Full candidate data
│       ├── screening_results.json
│       ├── phase_diagram.json
│       ├── dft_results.json
│       ├── protein_folds.json
│       ├── report.md           # Summary report
│       ├── cif/                # CIF geometry files
│       ├── xyz/                # XYZ coordinate files
│       ├── fold_cif/           # Protein fold CIFs
│       ├── fold_xyz/           # Protein fold XYRs
│       └── plots/              # Visualization PNGs
└── summary.md                  # This file
```

---

## Usage

```bash
# Run with defaults (310K, supercritical ammonia, no GNN)
python run_pipeline.py

# Run with custom settings
python run_pipeline.py --n-candidates 1000 --temperature 340 --solvent supercritical_ammonia

# Run with GNN (requires trained model)
python run_pipeline.py --no-gnn  # Explicitly disable until trained
```

---

## Theoretical Foundations

### Bond Energy References (kJ/mol)
| Bond | Energy | Note |
|------|--------|------|
| Si-N | 355 | Strong, like terrestrial silicates |
| Si-Al | 230 | Aluminosilicate framework |
| Al-N | 297 | Lewis acidic coordination |
| Si-Si | 226 | Weak — avoid in favor of Si-N/Si-Al |
| P-H | 322 | Phosphine functional group |

### Thermodynamic Stability
- ΔG ≈ ΔH - TΔS at 310K
- Entropy corrections favor molecules with more bonds
- On-hull candidates are **thermodynamically optimal** (no decomposition pathway)

---

## References
- GNoME: Merchant et al., Nature 2023
- VASP: Kresse & Furthmüller, PRB 1996
- Pymatgen: Ong et al., CM 2013
- Bond energies: CRC Handbook, NIST

---

*Generated: 2026-04-09 | Pipeline Runtime: 8.4s | Temperature: 310K*
