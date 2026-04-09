"""
Visualization Module for Metallosilicon Amino Acid Pipeline

Generates publication-quality plots for:
- Formation energy distribution
- HOMO-LUMO gap analysis
- Convex hull phase diagrams
- Metal coordination geometry
- Solvent stability heatmaps
- Protein fold structures
- Electron Localization Function (ELF) contours
"""

import numpy as np
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter

from scripts.gnome_model import ELEMENT_PROPERTIES

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available. Install with: pip install matplotlib")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning("plotly not available. Install with: pip install plotly")


class PipelineVisualizer:
    """Generate visualizations for pipeline results."""

    def __init__(self, output_dir: str = "sims/output/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_formation_energy_distribution(
        self,
        formation_energies: List[float],
        hull_distances: List[float],
        passed: List[bool],
        candidate_ids: List[str],
        filename: str = "formation_energy_distribution.png",
    ):
        """Plot formation energy distribution with hull distance coloring."""
        if not HAS_MATPLOTLIB:
            logger.warning("Skipping plot: matplotlib not available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Formation energy histogram
        passed_fe = [fe for fe, p in zip(formation_energies, passed) if p]
        failed_fe = [fe for fe, p in zip(formation_energies, passed) if not p]

        ax1.hist(passed_fe, bins=30, alpha=0.7, color='#2ecc71', label='Stable')
        ax1.hist(failed_fe, bins=30, alpha=0.5, color='#e74c3c', label='Unstable')
        ax1.axvline(x=0.2, color='orange', linestyle='--', label='Threshold (0.2 eV)')
        ax1.set_xlabel('Formation Energy (eV/atom)', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Formation Energy Distribution', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Hull distance scatter
        colors = ['#2ecc71' if p else '#e74c3c' for p in passed]
        sizes = [30 if p else 10 for p in passed]
        ax2.scatter(formation_energies, hull_distances, c=colors, s=sizes, alpha=0.6)
        ax2.axhline(y=0.05, color='orange', linestyle='--', label='Hull tolerance')
        ax2.set_xlabel('Formation Energy (eV/atom)', fontsize=12)
        ax2.set_ylabel('Hull Distance (eV)', fontsize=12)
        ax2.set_title('Formation Energy vs Hull Distance', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved formation energy plot to {filepath}")

    def plot_homo_lumo_analysis(
        self,
        homo_lumo_gaps: List[float],
        metals: List[str],
        passed: List[bool],
        filename: str = "homo_lumo_analysis.png",
    ):
        """Plot HOMO-LUMO gap analysis by metal center."""
        if not HAS_MATPLOTLIB:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # HOMO-LUMO distribution by metal
        metal_colors = {'Fe': '#e74c3c', 'Ni': '#3498db', 'Ti': '#2ecc71', 'Mo': '#9b59b6'}
        for metal in ['Fe', 'Ni', 'Ti', 'Mo']:
            gaps = [g for g, m, p in zip(homo_lumo_gaps, metals, passed) if m == metal and p]
            if gaps:
                ax1.hist(gaps, bins=15, alpha=0.6, color=metal_colors.get(metal, 'gray'),
                        label=f'{metal} (n={len(gaps)})')

        ax1.axvline(x=0.5, color='orange', linestyle='--', label='Semiconductor threshold')
        ax1.axvline(x=0.1, color='red', linestyle='--', label='Nanowire regime')
        ax1.set_xlabel('HOMO-LUMO Gap (eV)', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('HOMO-LUMO Gap by Metal Center', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Box plot by metal
        data_by_metal = {}
        for metal in ['Fe', 'Ni', 'Ti', 'Mo']:
            gaps = [g for g, m, p in zip(homo_lumo_gaps, metals, passed) if m == metal and p]
            if gaps:
                data_by_metal[metal] = gaps

        if data_by_metal:
            bp = ax2.boxplot(data_by_metal.values(), labels=data_by_metal.keys(),
                           patch_artist=True)
            for metal, patch in zip(data_by_metal.keys(), bp['boxes']):
                patch.set_facecolor(metal_colors.get(metal, 'gray'))
                patch.set_alpha(0.6)
        ax2.set_ylabel('HOMO-LUMO Gap (eV)', fontsize=12)
        ax2.set_xlabel('Metal Center', fontsize=12)
        ax2.set_title('HOMO-LUMO Gap Distribution', fontsize=14)
        ax2.grid(alpha=0.3)

        # Add conductivity annotations
        ax2.axhspan(0, 0.1, alpha=0.1, color='red', label='Nanowire')
        ax2.axhspan(0.1, 0.5, alpha=0.1, color='orange', label='Semiconductor')
        ax2.axhspan(0.5, 2.0, alpha=0.1, color='green', label='Insulator')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved HOMO-LUMO plot to {filepath}")

    def plot_solvent_stability_heatmap(
        self,
        candidates_data: List[Dict],
        filename: str = "solvent_stability_heatmap.png",
    ):
        """Plot solvent stability heatmap for top candidates."""
        if not HAS_MATPLOTLIB:
            return

        # Take top 20 candidates
        top = sorted(candidates_data, key=lambda c: c.get('formation_energy', 999))[:20]

        solvents = ['liquid_ammonia', 'liquid_methane', 'liquid_hydrogen_sulfide']
        short_names = ['NH₃(l)', 'CH₄(l)', 'H₂S(l)']

        matrix = np.zeros((len(top), len(solvents)))
        labels = []
        for i, c in enumerate(top):
            ss = c.get('solvent_stability', {})
            for j, s in enumerate(solvents):
                matrix[i, j] = ss.get(s, 0.0)
            labels.append(c.get('formula') or c.get('candidate_id', f'C{i}'))

        fig, ax = plt.subplots(figsize=(8, 10))
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(range(len(short_names)))
        ax.set_xticklabels(short_names, fontsize=11)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)

        # Add text annotations
        for i in range(len(top)):
            for j in range(len(solvents)):
                text = f'{matrix[i, j]:.2f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=8,
                       color='black' if matrix[i, j] > 0.5 else 'white')

        plt.colorbar(im, label='Stability Score')
        ax.set_title('Solvent Stability Heatmap (Top 20 Candidates)', fontsize=14)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved solvent heatmap to {filepath}")

    def plot_coordination_geometry(
        self,
        candidates_data: List[Dict],
        filename: str = "coordination_geometry.png",
    ):
        """Plot metal coordination geometry distribution."""
        if not HAS_MATPLOTLIB:
            return

        passed = [c for c in candidates_data if c.get('passed_screening', False)]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Coordination geometry pie chart
        geom_counts = Counter(c.get('coordination_geometry', 'unknown') for c in passed)
        ax = axes[0, 0]
        if geom_counts:
            ax.pie(geom_counts.values(), labels=geom_counts.keys(), autopct='%1.1f%%',
                  colors=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        ax.set_title('Coordination Geometry Distribution', fontsize=12)

        # 2. Metal center distribution
        metal_counts = Counter(c.get('metal_center', 'None') for c in passed)
        ax = axes[0, 1]
        metals = ['Fe', 'Ni', 'Ti', 'Mo']
        counts = [metal_counts.get(m, 0) for m in metals]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        ax.bar(metals, counts, color=colors)
        ax.set_xlabel('Metal', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Metal Center Distribution', fontsize=12)

        # 3. Coordination number distribution
        cn_counts = Counter(c.get('metal_coordination_number', 0) for c in passed)
        ax = axes[1, 0]
        if cn_counts:
            cns = sorted(cn_counts.keys())
            ax.bar([str(cn) for cn in cns], [cn_counts[cn] for cn in cns],
                  color='#3498db', alpha=0.7)
        ax.set_xlabel('Coordination Number', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Metal Coordination Number', fontsize=12)

        # 4. Metal vs HOMO-LUMO gap
        ax = axes[1, 1]
        for metal, color in zip(['Fe', 'Ni', 'Ti', 'Mo'], colors):
            gaps = [c.get('homo_lumo_gap', 0) for c in passed if c.get('metal_center') == metal]
            if gaps:
                ax.scatter([metal]*len(gaps), gaps, c=color, s=50, alpha=0.6, label=metal)
        ax.set_ylabel('HOMO-LUMO Gap (eV)', fontsize=11)
        ax.set_title('Metal Center vs HOMO-LUMO Gap', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.suptitle('Metal Coordination Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved coordination geometry plot to {filepath}")

    def plot_protein_folds(
        self,
        folds_data: List[Dict],
        filename: str = "protein_folds_summary.png",
    ):
        """Plot protein fold simulation results."""
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        fold_types = ['alpha_helix', 'beta_sheet', 'beta_barrel', 'coiled_coil', 'tim_barrel']
        metals = ['Fe', 'Ni', 'Ti', 'Mo']
        metal_colors = {'Fe': '#e74c3c', 'Ni': '#3498db', 'Ti': '#2ecc71', 'Mo': '#9b59b6'}

        # 1. HOMO-LUMO by fold type
        ax = axes[0, 0]
        for fold_type in fold_types:
            gaps = [f['homo_lumo_gap'] for f in folds_data if f['fold_type'] == fold_type]
            if gaps:
                ax.hist(gaps, bins=10, alpha=0.6, label=fold_type.replace('_', ' '))
        ax.set_xlabel('HOMO-LUMO Gap (eV)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('HOMO-LUMO by Fold Type', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # 2. Conductivity by fold type
        ax = axes[0, 1]
        for fold_type in fold_types:
            conds = [f['conductivity_estimate'] for f in folds_data if f['fold_type'] == fold_type]
            if conds:
                ax.hist(np.log10([max(c, 1e-20) for c in conds]), bins=10,
                       alpha=0.6, label=fold_type.replace('_', ' '))
        ax.set_xlabel('log₁₀(Conductivity) (S/cm)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Conductivity by Fold Type', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # 3. Stability by fold type
        ax = axes[0, 2]
        for fold_type in fold_types:
            stabs = [f['estimated_stability'] for f in folds_data if f['fold_type'] == fold_type]
            if stabs:
                ax.hist(stabs, bins=10, alpha=0.6, label=fold_type.replace('_', ' '))
        ax.set_xlabel('Stability Score', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Fold Stability', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # 4. Metal vs conductivity
        ax = axes[1, 0]
        for metal in metals:
            conds = [f['conductivity_estimate'] for f in folds_data if f['metal_center'] == metal]
            if conds:
                ax.boxplot([np.log10([max(c, 1e-20) for c in conds])],
                          positions=[metals.index(metal)],
                          patch_artist=True,
                          boxprops=dict(facecolor=metal_colors[metal], alpha=0.6))
        ax.set_xticks(range(len(metals)))
        ax.set_xticklabels(metals)
        ax.set_ylabel('log₁₀(σ) (S/cm)', fontsize=10)
        ax.set_title('Conductivity by Metal', fontsize=11)
        ax.grid(alpha=0.3)

        # 5. Solvent compatibility
        ax = axes[1, 1]
        solvents = ['liquid_ammonia', 'liquid_methane', 'liquid_hydrogen_sulfide']
        solvent_short = ['NH₃', 'CH₄', 'H₂S']
        for i, (sol, short) in enumerate(zip(solvents, solvent_short)):
            scores = [f['solvent_compatibility'].get(sol, 0) for f in folds_data]
            ax.hist(scores, bins=15, alpha=0.5, label=short)
        ax.set_xlabel('Compatibility Score', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Solvent Compatibility', fontsize=11)
        ax.legend()
        ax.grid(alpha=0.3)

        # 6. Fold type vs metal heatmap
        ax = axes[1, 2]
        matrix = np.zeros((len(fold_types), len(metals)))
        for i, ft in enumerate(fold_types):
            for j, m in enumerate(metals):
                count = sum(1 for f in folds_data if f['fold_type'] == ft and f['metal_center'] == m)
                matrix[i, j] = count
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(metals)))
        ax.set_xticklabels(metals)
        ax.set_yticks(range(len(fold_types)))
        ax.set_yticklabels([ft.replace('_', ' ') for ft in fold_types], fontsize=9)
        plt.colorbar(im, ax=ax, label='Count')
        ax.set_title('Fold × Metal Matrix', fontsize=11)

        plt.suptitle('Silico-Protein Fold Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved protein folds plot to {filepath}")

    def plot_3d_molecule(
        self,
        atom_types: List[str],
        positions: np.ndarray,
        candidate_id: str,
        filename: str = None,
    ):
        """Create interactive 3D molecular visualization using plotly."""
        if not HAS_PLOTLY:
            logger.warning("Skipping 3D plot: plotly not available")
            return

        element_colors = {
            'Si': '#6c5ce7', 'N': '#0984e3', 'H': '#dfe6e9',
            'S': '#fdcb6e', 'P': '#e17055', 'B': '#00b894',
            'F': '#55efc4', 'C': '#2d3436',
            'Fe': '#d63031', 'Ni': '#636e72', 'Ti': '#74b9ff', 'Mo': '#a29bfe',
        }
        element_sizes = {
            'Si': 12, 'N': 10, 'H': 5, 'S': 11, 'P': 11,
            'B': 9, 'F': 8, 'C': 10,
            'Fe': 14, 'Ni': 13, 'Ti': 14, 'Mo': 14,
        }

        fig = go.Figure()

        for i, (atom, pos) in enumerate(zip(atom_types, positions)):
            color = element_colors.get(atom, '#b2bec3')
            size = element_sizes.get(atom, 8)
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers+text',
                marker=dict(size=size, color=color, opacity=0.9),
                text=[atom],
                textposition='top center',
                name=f'{atom}{i+1}',
                showlegend=False,
            ))

        fig.update_layout(
            title=f'Metallosilicon Amino Acid: {candidate_id}',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data',
            ),
            width=800,
            height=600,
        )

        if filename is None:
            filename = f"molecule_{candidate_id}.html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        logger.info(f"Saved 3D molecule plot to {filepath}")

    def plot_elf_contour(
        self,
        positions: np.ndarray,
        atom_types: List[str],
        candidate_id: str,
        filename: str = "elf_contour.png",
    ):
        """
        Plot Electron Localization Function (ELF) contour map.

        Uses a simplified model: ELF ≈ 1/(1 + (D/D₀)²) where D is
        the electron density gradient proxy based on distance to nuclei.
        """
        if not HAS_MATPLOTLIB:
            return

        # Create 2D slice through the molecular plane
        # Project onto xy plane (z = mean z)
        mean_z = positions[:, 2].mean()

        # Grid
        x_range = [positions[:, 0].min() - 3, positions[:, 0].max() + 3]
        y_range = [positions[:, 1].min() - 3, positions[:, 1].max() + 3]

        nx, ny = 100, 100
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)

        # Compute simplified ELF
        # ELF = 1 / (1 + (D/D₀)²)
        # D = sum of |∇ρ| contributions from each atom
        D = np.zeros_like(X)
        for atom, pos in zip(atom_types, positions):
            # Distance from grid point to atom
            r = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2 + (mean_z - pos[2])**2)
            r = np.maximum(r, 0.1)

            # Simplified electron density: ρ ∝ Z * exp(-αr)
            Z_eff = ELEMENT_PROPERTIES.get(atom, [1, 1, 50, 1])[0]
            alpha = 0.5  # decay constant
            rho = Z_eff * np.exp(-alpha * r)

            # Gradient proxy
            D += np.abs(np.gradient(rho, axis=0)) + np.abs(np.gradient(rho, axis=1))

        D0 = np.median(D[D > 0]) + 1e-8
        ELF = 1.0 / (1.0 + (D / D0)**2)

        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(X, Y, ELF, levels=20, cmap='RdYlBu_r')
        plt.colorbar(contour, label='ELF')

        # Plot atom positions
        element_colors_2d = {
            'Si': 'purple', 'N': 'blue', 'H': 'gray', 'S': 'yellow',
            'P': 'orange', 'B': 'green', 'F': 'cyan', 'C': 'black',
            'Fe': 'red', 'Ni': 'silver', 'Ti': 'lightblue', 'Mo': 'violet',
        }
        for atom, pos in zip(atom_types, positions):
            color = element_colors_2d.get(atom, 'gray')
            size = 50 if atom in ['Fe', 'Ni', 'Ti', 'Mo'] else 30
            ax.scatter(pos[0], pos[1], c=color, s=size, zorder=5,
                      edgecolors='black', linewidth=0.5)
            ax.annotate(atom, (pos[0], pos[1]), fontsize=7,
                       ha='center', va='bottom')

        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        ax.set_title(f'Electron Localization Function — {candidate_id}', fontsize=14)
        ax.set_aspect('equal')

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved ELF contour to {filepath}")

    def generate_all_plots(
        self,
        candidates_data: List[Dict],
        folds_data: List[Dict],
    ):
        """Generate all visualization plots."""
        # Formation energy distribution
        fe = [c.get('formation_energy', 0) for c in candidates_data]
        hd = [c.get('hull_distance', 0) for c in candidates_data]
        passed = [c.get('passed_screening', False) for c in candidates_data]
        ids = [c.get('formula') or c.get('candidate_id', '') for c in candidates_data]
        self.plot_formation_energy_distribution(fe, hd, passed, ids)

        # HOMO-LUMO analysis
        gaps = [c.get('homo_lumo_gap', 0) for c in candidates_data]
        metals = [c.get('metal_center', 'None') for c in candidates_data]
        self.plot_homo_lumo_analysis(gaps, metals, passed)

        # Solvent stability heatmap
        self.plot_solvent_stability_heatmap(candidates_data)

        # Coordination geometry
        self.plot_coordination_geometry(candidates_data)

        # Protein folds
        if folds_data:
            self.plot_protein_folds(folds_data)

        # 3D molecule for top candidate
        if HAS_PLOTLY and candidates_data:
            best = sorted(candidates_data, key=lambda c: c.get('formation_energy', 999))[0]
            if 'positions' in best and 'atom_types' in best:
                self.plot_3d_molecule(
                    best['atom_types'],
                    np.array(best['positions']),
                    best.get('formula') or best['candidate_id'],
                )

        # ELF contour for top candidate
        if candidates_data:
            best = sorted(candidates_data, key=lambda c: c.get('formation_energy', 999))[0]
            if 'positions' in best and 'atom_types' in best:
                self.plot_elf_contour(
                    np.array(best['positions']),
                    best['atom_types'],
                    best.get('formula') or best['candidate_id'],
                )

        logger.info(f"All plots saved to {self.output_dir}")
