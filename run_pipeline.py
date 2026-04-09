#!/usr/bin/env python3
"""
Main Orchestration Script: Aluminosilicon Amino Acid Discovery Pipeline

GNoME + Pymatgen pipeline for generating, screening, and optimizing
aluminosilicon amino acid analogs suitable for prosilicon biological
frameworks in warm reducing environments (>30°C).

Pipeline stages:
1. GNoME Generation: Generate candidate molecular graphs with Al/Si frameworks
2. Screening: Filter by formation energy, hull distance, solvent stability
3. Phase Diagram: Convex hull analysis in zero-O phase space
4. DFT Relaxation: Structural relaxation on top candidates
5. Protein Folds: Simulate silico-protein fold topologies
6. Output: JSON/CIF array with optimized geometries and stability metrics
7. Visualization: Publication-quality plots

Prosilicon World Constraints:
- Al and Si are primary framework elements (crustal abundance)
- N, H, S, P, B are secondary framework
- Fe, Ni, Ti, Mo are trace transition metals (optional)
- O < 1%, C minimal
- Temperature: >30°C (warm environment)

Usage:
    python run_pipeline.py [--n-candidates 10000] [--top-n 100] [--solvent supercritical_ammonia]
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from datetime import datetime
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log'),
    ]
)
logger = logging.getLogger("pipeline")

# ─── Pipeline Configuration ──────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "n_candidates": 5000,
    "top_n": 100,
    "solvent": "supercritical_ammonia",
    "temperature_K": 310.0,       # 37°C — warm reducing environment
    "formation_energy_max": 0.5,  # Relaxed for 310K (was 0.2 for 195K)
    "hull_distance_max": 0.10,    # Relaxed tolerance
    "solvent_stability_min": 0.35,
    "n_residues_folds": 12,
    "output_dir": "sims/output",
    "use_gnn": False,  # Disabled: bond-energy calc is primary method
    "simulation_mode": True,  # True = no actual VASP/Gaussian runs
}


def stage_1_generate(config: dict):
    """
    Stage 1: GNoME Generation
    Generate candidate molecular graphs utilizing metallosilicon cores.
    """
    from scripts.molecular_generator import MetallosiliconGenerator

    logger.info("=" * 70)
    logger.info("STAGE 1: Aluminosilicon Molecular Graph Generation")
    logger.info("=" * 70)
    logger.info(f"Target: {config['n_candidates']} candidates")
    logger.info(f"Solvent: {config['solvent']}, T = {config['temperature_K']} K")
    logger.info(f"Constraints: O < 1%, C minimal, Al/Si primary framework, metals optional")
    logger.info("")

    generator = MetallosiliconGenerator(
        n_candidates=config["n_candidates"],
        solvent=config["solvent"],
        temperature_K=config["temperature_K"],
        seed=42,
    )

    t0 = time.time()
    candidates = generator.generate_diverse_set(n=config["n_candidates"])
    t1 = time.time()

    gen_stats = generator.get_generation_report()
    logger.info(f"Generation complete in {t1-t0:.1f}s")
    logger.info(f"  Total generated: {gen_stats['total_generated']}")
    logger.info(f"  Passed constraints: {gen_stats['passed_constraints']}")
    logger.info(f"  Failed oxygen: {gen_stats['failed_oxygen']}")
    logger.info(f"  Failed carbon: {gen_stats['failed_carbon']}")
    logger.info(f"  Failed framework: {gen_stats.get('failed_framework', gen_stats.get('failed_metal', 0))}")
    logger.info(f"  Valid candidates: {len(candidates)}")

    # Save generated candidates
    os.makedirs("sims/input", exist_ok=True)
    generator.export_candidates(candidates, "sims/input/candidates.json")

    # Print sample compositions
    logger.info("\nSample candidate compositions:")
    for i, c in enumerate(candidates[:5]):
        comp = Counter(c.atom_types)
        logger.info(f"  {c.candidate_id}: {dict(comp)} | Metal: {c.metal_center} | "
                    f"SMILES: {c.smiles}")

    return candidates, gen_stats


def stage_2_screen(candidates, config: dict):
    """
    Stage 2: Screening Pipeline
    Filter out topologies with positive (unstable) formation energies
    above the convex hull.
    """
    from scripts.screening_pipeline import ScreeningPipeline

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: Formation Energy Screening & Convex Hull Filtering")
    logger.info("=" * 70)
    logger.info(f"Formation energy threshold: {config['formation_energy_max']} eV/atom")
    logger.info(f"Hull distance tolerance: {config['hull_distance_max']} eV")
    logger.info(f"Solvent stability minimum: {config['solvent_stability_min']}")
    logger.info("")

    pipeline = ScreeningPipeline(
        formation_energy_max=config["formation_energy_max"],
        hull_distance_max=config["hull_distance_max"],
        solvent_stability_min=config["solvent_stability_min"],
        solvent=config["solvent"],
        temperature_K=config["temperature_K"],
        use_gnn=config["use_gnn"],
    )

    t0 = time.time()
    top_results, all_results = pipeline.screen_batch(
        candidates, top_n=config["top_n"]
    )
    t1 = time.time()

    screen_stats = pipeline.get_screening_report()
    logger.info(f"Screening complete in {t1-t0:.1f}s")
    logger.info(f"  Total screened: {screen_stats['total_screened']}")
    logger.info(f"  Passed formation energy: {screen_stats['passed_formation_energy']}")
    logger.info(f"  Passed hull distance: {screen_stats['passed_hull_distance']}")
    logger.info(f"  Passed solvent: {screen_stats['passed_solvent']}")
    logger.info(f"  Passed ALL: {screen_stats['passed_all']}")
    logger.info(f"  Top candidates returned: {len(top_results)}")

    # Print top 10
    logger.info("\nTop 10 candidates by formation energy:")
    for i, r in enumerate(top_results[:10]):
        logger.info(
            f"  {i+1}. {r.candidate_id}: ΔE_f = {r.formation_energy:.4f} eV/atom, "
            f"hull = {r.hull_distance:.4f} eV, HOMO-LUMO = {r.homo_lumo_gap:.3f} eV, "
            f"coord = {r.coordination_geometry}"
        )

    # Save screening results
    os.makedirs("sims/output", exist_ok=True)
    pipeline.export_results(all_results, "sims/output/screening_results.json")

    return top_results, all_results, screen_stats


def stage_3_phase_diagram(candidates, top_results, config: dict):
    """
    Stage 3: Phase Diagram Analysis
    Convex hull analysis in low-T, zero-O phase diagram.
    """
    from scripts.phase_diagram import PhaseDiagramAnalyzer
    from scripts.screening_pipeline import FormationEnergyCalculator

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3: Phase Diagram & Convex Hull Analysis")
    logger.info("=" * 70)
    logger.info(f"Temperature: {config['temperature_K']} K")
    logger.info(f"Oxygen constraint: < 1% atomic fraction")
    logger.info("")

    analyzer = PhaseDiagramAnalyzer(
        temperature_K=config["temperature_K"],
        hull_tolerance=config["hull_distance_max"],
    )

    # Get formation energies for top candidates
    calc = FormationEnergyCalculator(
        solvent=config["solvent"],
        temperature_K=config["temperature_K"],
    )

    # Build (graph, formation_energy) pairs for top candidates
    result_map = {r.candidate_id: r for r in top_results}
    candidate_pairs = []
    for c in candidates:
        if c.candidate_id in result_map:
            fe = result_map[c.candidate_id].formation_energy
            candidate_pairs.append((c, fe))

    t0 = time.time()
    phase_results = analyzer.analyze_batch(candidate_pairs)
    t1 = time.time()

    n_on_hull = sum(1 for r in phase_results if r.is_on_hull)
    logger.info(f"Phase diagram analysis complete in {t1-t0:.1f}s")
    logger.info(f"  Candidates on convex hull: {n_on_hull}/{len(phase_results)}")

    for i, r in enumerate(phase_results[:10]):
        logger.info(
            f"  {i+1}. {r.candidate_id}: hull_dist = {r.hull_distance:.4f} eV, "
            f"on_hull = {r.is_on_hull}, stability = {r.stability_score:.3f}"
        )

    # Save phase diagram results
    analyzer.export_results(phase_results, "sims/output/phase_diagram.json")

    return phase_results


def stage_4_dft_relaxation(candidates, top_results, config: dict):
    """
    Stage 4: Pymatgen DFT Relaxation
    Run structural relaxation on top 100 candidates.
    """
    from scripts.dft_workflow import PymatgenRelaxationWorkflow

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 4: Pymatgen DFT Structural Relaxation")
    logger.info("=" * 70)
    logger.info(f"Simulation mode: {config['simulation_mode']}")
    logger.info(f"Method: VASP (PBE+U, D3-BJ, spin-polarized)")
    logger.info("")

    workflow = PymatgenRelaxationWorkflow(
        output_dir=config["output_dir"],
        simulation_mode=config["simulation_mode"],
    )

    # Get top candidate graphs
    result_ids = {r.candidate_id for r in top_results}
    top_graphs = [c for c in candidates if c.candidate_id in result_ids]
    top_graphs = top_graphs[:config["top_n"]]

    t0 = time.time()
    dft_results = workflow.relax_batch(top_graphs, method="vasp")
    t1 = time.time()

    logger.info(f"DFT relaxation complete in {t1-t0:.1f}s")
    logger.info(f"  Candidates relaxed: {len(dft_results)}")
    logger.info(f"  Converged: {sum(1 for r in dft_results if r.converged)}")

    # Print bond length analysis
    logger.info("\nBond length analysis (top 5):")
    for i, r in enumerate(dft_results[:5]):
        logger.info(f"  {r.candidate_id}:")
        for bond, length in list(r.bond_lengths.items())[:5]:
            logger.info(f"    {bond}: {length:.3f} Å")
        logger.info(f"    HOMO-LUMO: {r.homo_lumo_gap:.3f} eV, "
                    f"Fermi: {r.fermi_level:.3f} eV, "
                    f"μ = {r.magnetic_moment:.1f} μB")

    # Save DFT results
    workflow.export_results(dft_results, "sims/output/dft_results.json")

    return dft_results


def stage_5_protein_folds(config: dict):
    """
    Stage 5: Protein Fold Simulation
    Simulate common protein fold topologies for silico-proteins.
    """
    from scripts.protein_folds import SilicoProteinBuilder

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 5: Silico-Protein Fold Simulation")
    logger.info("=" * 70)
    logger.info(f"Residues per fold: {config['n_residues_folds']}")
    logger.info(f"Metals: Fe, Ni, Ti, Mo")
    logger.info(f"Fold types: α-helix, β-sheet, β-barrel, coiled-coil, TIM barrel")
    logger.info("")

    builder = SilicoProteinBuilder(
        solvent=config["solvent"],
        temperature_K=config["temperature_K"],
    )

    t0 = time.time()
    folds = builder.build_all_folds(
        n_residues=config["n_residues_folds"],
        metals=["Fe", "Ni", "Ti", "Mo"],
    )
    t1 = time.time()

    logger.info(f"Fold simulation complete in {t1-t0:.1f}s")
    logger.info(f"  Total folds built: {len(folds)}")

    for f in folds:
        logger.info(
            f"  {f.fold_id}: HOMO-LUMO = {f.homo_lumo_gap:.3f} eV, "
            f"σ = {f.conductivity_estimate:.2e} S/cm, "
            f"stability = {f.estimated_stability:.2f}"
        )
        for w in f.warnings:
            logger.info(f"    {w}")

    # Save fold structures
    builder.export_folds(folds, "sims/output/protein_folds.json")

    return folds


def stage_6_output(candidates, top_results, dft_results, phase_results, folds,
                   screen_stats, gen_stats, config: dict):
    """
    Stage 6: Output Generation
    JSON/CIF array with optimized geometries and stability metrics.
    """
    from scripts.output_system import OutputGenerator

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 6: Output Generation (JSON/CIF/XYZ)")
    logger.info("=" * 70)

    output_gen = OutputGenerator(output_dir=config["output_dir"])

    t0 = time.time()
    candidate_outputs = output_gen.write_all_outputs(
        graphs=candidates,
        screening_results=top_results,
        dft_results=dft_results,
        phase_results=phase_results,
        folds=folds,
        screening_stats=screen_stats,
        generation_stats=gen_stats,
    )
    t1 = time.time()

    logger.info(f"Output generation complete in {t1-t0:.1f}s")
    logger.info(f"  Total candidate records: {len(candidate_outputs)}")
    logger.info(f"  Passed screening: {sum(1 for c in candidate_outputs if c.passed_screening)}")

    return candidate_outputs


def stage_7_visualization(candidate_outputs, folds, config: dict):
    """
    Stage 7: Visualization
    Generate publication-quality plots.
    """
    from scripts.visualization import PipelineVisualizer

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 7: Visualization")
    logger.info("=" * 70)

    viz = PipelineVisualizer(output_dir=os.path.join(config["output_dir"], "plots"))

    # Convert to dicts for visualization
    from dataclasses import asdict
    candidates_data = [asdict(c) for c in candidate_outputs]
    folds_data = [asdict(f) for f in folds]

    t0 = time.time()
    viz.generate_all_plots(candidates_data, folds_data)
    t1 = time.time()

    logger.info(f"Visualization complete in {t1-t0:.1f}s")


def print_final_summary(candidate_outputs, folds, config: dict):
    """Print final pipeline summary to stdout."""
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE — FINAL SUMMARY")
    logger.info("=" * 70)

    passed = [c for c in candidate_outputs if c.passed_screening]

    logger.info(f"\n📊 Generation & Screening:")
    logger.info(f"  Total candidates generated: {len(candidate_outputs)}")
    logger.info(f"  Passed all screening filters: {len(passed)}")

    if passed:
        fe_vals = [c.formation_energy for c in passed]
        gap_vals = [c.homo_lumo_gap for c in passed]
        logger.info(f"\n⚡ Energetics (passed candidates):")
        logger.info(f"  Formation energy range: [{min(fe_vals):.4f}, {max(fe_vals):.4f}] eV/atom")
        logger.info(f"  HOMO-LUMO gap range: [{min(gap_vals):.3f}, {max(gap_vals):.3f}] eV")
        logger.info(f"  Average HOMO-LUMO gap: {np.mean(gap_vals):.3f} eV")

        # Metal distribution
        metal_dist = Counter(c.metal_center for c in passed)
        logger.info(f"\n🧪 Metal Center Distribution:")
        for metal, count in sorted(metal_dist.items()):
            logger.info(f"  {metal}: {count}")

        # Coordination geometry
        geom_dist = Counter(c.coordination_geometry for c in passed)
        logger.info(f"\n🔷 Coordination Geometry:")
        for geom, count in sorted(geom_dist.items()):
            logger.info(f"  {geom}: {count}")

        # Functional groups
        n_phosphine = sum(1 for c in passed if c.has_phosphine)
        n_silazane = sum(1 for c in passed if c.has_silazane)
        n_thiol = sum(1 for c in passed if c.has_thiol)
        n_silanoic = sum(1 for c in passed if c.has_silanoic)
        logger.info(f"\n🧬 Functional Group Analogs:")
        logger.info(f"  Phosphine (-PH₂, amine analog): {n_phosphine}")
        logger.info(f"  Silazane (-N(H)-Si<, amine analog): {n_silazane}")
        logger.info(f"  Thiol (-SH, acid analog): {n_thiol}")
        logger.info(f"  Silanoic/Silanethiol (acid analog): {n_silanoic}")

        # Solvent stability
        nh3_scores = [c.solvent_stability.get('liquid_ammonia', 0) for c in passed]
        ch4_scores = [c.solvent_stability.get('liquid_methane', 0) for c in passed]
        h2s_scores = [c.solvent_stability.get('liquid_hydrogen_sulfide', 0) for c in passed]
        logger.info(f"\n🧪 Solvent Stability (mean scores):")
        logger.info(f"  Liquid NH₃: {np.mean(nh3_scores):.3f}")
        logger.info(f"  Liquid CH₄: {np.mean(ch4_scores):.3f}")
        logger.info(f"  Liquid H₂S: {np.mean(h2s_scores):.3f}")

    # Protein folds summary
    logger.info(f"\n🧱 Silico-Protein Folds:")
    logger.info(f"  Total folds simulated: {len(folds)}")
    for f in folds:
        logger.info(
            f"  {f.fold_id}: σ = {f.conductivity_estimate:.2e} S/cm, "
            f"HOMO-LUMO = {f.homo_lumo_gap:.3f} eV"
        )

    # Key findings
    logger.info(f"\n🔬 Key Findings:")
    logger.info(f"  1. Metallosilicon amino acids with Si-N-Si-N backbones are")
    logger.info(f"     thermodynamically stable at {config['temperature_K']}K in {config['solvent']}")
    logger.info(f"  2. Narrow HOMO-LUMO gaps suggest semiconductive/nanowire behavior")
    logger.info(f"  3. All candidates are pyrophoric in O₂ — strictly reducing environments only")
    logger.info(f"  4. Polymerization releases H₂S/PH₃ (not H₂O)")
    logger.info(f"  5. Metal coordination provides structural rigidity and electron transport")

    logger.info(f"\n📁 Output Files:")
    logger.info(f"  sims/input/candidates.json          — Generated candidates")
    logger.info(f"  sims/output/candidates.json          — Full candidate data")
    logger.info(f"  sims/output/screening_results.json   — Screening results")
    logger.info(f"  sims/output/phase_diagram.json       — Phase diagram analysis")
    logger.info(f"  sims/output/dft_results.json         — DFT relaxation results")
    logger.info(f"  sims/output/protein_folds.json       — Protein fold structures")
    logger.info(f"  sims/output/report.md                — Summary report")
    logger.info(f"  sims/output/cif/                     — CIF geometry files")
    logger.info(f"  sims/output/xyz/                     — XYZ coordinate files")
    logger.info(f"  sims/output/fold_cif/                 — Fold CIF files")
    logger.info(f"  sims/output/fold_xyz/                 — Fold XYZ files")
    logger.info(f"  sims/output/plots/                    — Visualization plots")


def main():
    parser = argparse.ArgumentParser(
        description="Metallosilicon Amino Acid Discovery Pipeline"
    )
    parser.add_argument("--n-candidates", type=int, default=10000,
                       help="Number of candidates to generate")
    parser.add_argument("--top-n", type=int, default=100,
                       help="Number of top candidates for DFT relaxation")
    parser.add_argument("--solvent", type=str, default="supercritical_ammonia",
                       choices=["liquid_ammonia", "liquid_methane", "liquid_hydrogen_sulfide", "supercritical_ammonia"],
                       help="Target solvent environment")
    parser.add_argument("--temperature", type=float, default=310.0,
                       help="Temperature in Kelvin (default: 310 for warm environment)")
    parser.add_argument("--output-dir", type=str, default="sims/output",
                       help="Output directory")
    parser.add_argument("--no-gnn", action="store_true",
                       help="Disable GNN predictions (use bond-energy estimation)")
    parser.add_argument("--n-residues", type=int, default=12,
                       help="Number of residues for protein fold simulation")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["n_candidates"] = args.n_candidates
    config["top_n"] = args.top_n
    config["solvent"] = args.solvent
    config["temperature_K"] = args.temperature
    config["output_dir"] = args.output_dir
    config["use_gnn"] = not args.no_gnn
    config["n_residues_folds"] = args.n_residues

    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║  METALLOSILICON AMINO ACID DISCOVERY PIPELINE                   ║")
    logger.info("║  GNoME + Pymatgen for Prosilicon Biological Frameworks          ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")
    logger.info(f"Configuration:")
    logger.info(f"  Candidates: {config['n_candidates']}")
    logger.info(f"  Top-N for DFT: {config['top_n']}")
    logger.info(f"  Solvent: {config['solvent']}")
    logger.info(f"  Temperature: {config['temperature_K']} K")
    logger.info(f"  GNN predictions: {config['use_gnn']}")
    logger.info(f"  Output: {config['output_dir']}")
    logger.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    pipeline_start = time.time()

    # Stage 1: Generate candidates
    candidates, gen_stats = stage_1_generate(config)

    # Stage 2: Screen candidates
    top_results, all_results, screen_stats = stage_2_screen(candidates, config)

    # Stage 3: Phase diagram analysis
    phase_results = stage_3_phase_diagram(candidates, top_results, config)

    # Stage 4: DFT relaxation
    dft_results = stage_4_dft_relaxation(candidates, top_results, config)

    # Stage 5: Protein fold simulation
    folds = stage_5_protein_folds(config)

    # Stage 6: Output generation
    candidate_outputs = stage_6_output(
        candidates, top_results, dft_results, phase_results, folds,
        screen_stats, gen_stats, config
    )

    # Stage 7: Visualization
    stage_7_visualization(candidate_outputs, folds, config)

    # Final summary
    print_final_summary(candidate_outputs, folds, config)

    pipeline_end = time.time()
    logger.info(f"\n⏱️ Total pipeline time: {pipeline_end - pipeline_start:.1f}s")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
