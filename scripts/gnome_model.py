"""
GNoME (Graph Networks for Materials Exploration) Architecture
Adapted for Metallosilicon Molecular Topology Prediction

This module implements a graph neural network that predicts formation energies
and stability of metallosilicon amino acid analogs from molecular graphs.

Reference: Merchant et al., "Scaling deep learning for materials discovery", Nature 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)

# ─── Elemental Configuration for Metallosilicon Chemistry ────────────────────

# ─── Elemental Hierarchy for Prosilicon Life ────────────────────────────────────
# Based on planetary crustal composition (Earth-like but with 100x less O₂):
# Primary framework: Si (~28%), Al (~8%) — like Earth's crust, Al is abundant
# Secondary framework: N, H, S, P, B — significant crustal/atmosphere abundance
# Trace metals: Fe, Ni, Ti, Mo — scarce transition metals (<0.1% each)
# Minimal: F (very reactive), C (strictly limited)

# Primary framework elements (network formers like Si/Al in aluminosilicates)
PRIMARY_FRAMEWORK = ["Si", "Al"]

# Secondary framework elements (modifiers, ~2-8% crustal abundance)
SECONDARY_FRAMEWORK = ["N", "H", "S", "P", "B"]

# All backbone elements
BACKBONE_ELEMENTS = PRIMARY_FRAMEWORK + SECONDARY_FRAMEWORK + ["F"]

# Trace transition metals (scarce in crust, form coordination complexes)
METAL_ELEMENTS = ["Fe", "Ni", "Ti", "Mo"]

# Extremely limited
TRACE_ELEMENTS = ["C"]

ALL_ELEMENTS = PRIMARY_FRAMEWORK + SECONDARY_FRAMEWORK + METAL_ELEMENTS + TRACE_ELEMENTS + ["F"]

# Element properties: [atomic_number, electronegativity, covalent_radius_pm, valence_electrons]
ELEMENT_PROPERTIES = {
    "Si": [14, 1.90, 111, 4],
    "N":  [7,  3.04, 71,  5],
    "H":  [1,  2.20, 31,  1],
    "S":  [16, 2.58, 105, 6],
    "P":  [15, 2.19, 106, 5],
    "B":  [5,  2.04, 84,  3],
    "Al": [13, 1.61, 121, 3],
    "F":  [9,  3.98, 57,  7],
    "Fe": [26, 1.83, 132, 8],
    "Ni": [28, 1.91, 124, 10],
    "Ti": [22, 1.54, 147, 4],
    "Mo": [42, 2.16, 154, 6],
    "C":  [6,  2.55, 77,  4],
}

# Bond energy lookup (kJ/mol) for formation energy estimation
BOND_ENERGIES = {
    ("Si", "Si"): 226, ("Si", "N"): 355, ("Si", "H"): 323, ("Si", "S"): 293,
    ("Si", "P"): 314, ("Si", "B"): 289, ("Si", "Al"): 230, ("Si", "F"): 597, ("Si", "C"): 318,
    ("Si", "Fe"): 268, ("Si", "Ni"): 274, ("Si", "Ti"): 301, ("Si", "Mo"): 285,
    ("N", "H"): 391, ("N", "N"): 163, ("N", "S"): 272, ("N", "P"): 293,
    ("N", "B"): 389, ("N", "Al"): 297, ("N", "F"): 272, ("N", "C"): 305,
    ("S", "H"): 363, ("S", "S"): 268, ("S", "P"): 272, ("S", "B"): 301,
    ("S", "Al"): 251, ("S", "F"): 327, ("S", "C"): 272, ("S", "Fe"): 339, ("S", "Ni"): 347,
    ("S", "Ti"): 318, ("S", "Mo"): 355,
    ("P", "H"): 322, ("P", "P"): 201, ("P", "B"): 289, ("P", "Al"): 213,
    ("P", "F"): 490, ("P", "C"): 264, ("P", "Fe"): 272, ("P", "Ni"): 280, ("P", "Ti"): 293,
    ("B", "H"): 389, ("B", "B"): 293, ("B", "Al"): 251, ("B", "F"): 613, ("B", "C"): 356,
    ("Al", "H"): 285, ("Al", "Al"): 186, ("Al", "F"): 582, ("Al", "C"): 255,
    ("Al", "Fe"): 230, ("Al", "Ni"): 234, ("Al", "Ti"): 243, ("Al", "Mo"): 226,
    ("F", "H"): 565, ("F", "C"): 485, ("F", "Fe"): 350, ("F", "Ni"): 360,
    ("F", "Ti"): 380, ("F", "Mo"): 370,
    ("C", "H"): 413, ("C", "C"): 347,
    ("Fe", "Fe"): 168, ("Ni", "Ni"): 176, ("Ti", "Ti"): 130, ("Mo", "Mo"): 218,
}

# Coordination geometry templates for metal centers
METAL_COORDINATION = {
    "Fe": {"preferred_cn": 4, "geometries": ["tetrahedral", "square_planar"], "typical_bond_len": 2.15},
    "Ni": {"preferred_cn": 4, "geometries": ["tetrahedral", "square_planar"], "typical_bond_len": 2.05},
    "Ti": {"preferred_cn": 6, "geometries": ["octahedral", "trigonal_prismatic"], "typical_bond_len": 2.30},
    "Mo": {"preferred_cn": 6, "geometries": ["octahedral", "trigonal_prismatic"], "typical_bond_len": 2.35},
    "Al": {"preferred_cn": 4, "geometries": ["tetrahedral", "square_planar"], "typical_bond_len": 2.05},
}


@dataclass
class MolecularGraph:
    """Represents a metallosilicon molecular graph."""
    node_features: np.ndarray       # (n_atoms, n_features)
    edge_features: np.ndarray       # (n_bonds, n_edge_features)
    adjacency: np.ndarray            # (n_atoms, n_atoms) sparse
    atom_types: List[str]            # element symbols
    positions: np.ndarray            # (n_atoms, 3) Cartesian coordinates
    metal_center: Optional[str] = None
    metal_index: Optional[int] = None
    smiles: str = ""
    candidate_id: str = ""
    formula: str = ""
    display_name: str = ""
    oxygen_fraction: float = 0.0
    backbone_template: str = ""
    amine_analog: str = ""
    acid_analog: str = ""
    side_chain: str = ""
    has_phosphine: bool = False
    has_silazane: bool = False
    has_thiol: bool = False
    has_silanoic: bool = False


@dataclass
class PredictionResult:
    """Output from GNoME model prediction."""
    candidate_id: str
    formation_energy: float          # eV/atom
    stability_score: float           # probability of being stable
    hull_distance: float             # eV above convex hull (0 = on hull)
    homo_lumo_gap: float             # eV
    coordination_geometry: str
    solvent_stability: Dict[str, float]  # stability in different solvents
    graph: Optional[MolecularGraph] = None


# ─── GNoME Graph Neural Network Architecture ────────────────────────────────

class InteractionBlock(nn.Module):
    """
    GNoME interaction block: message-passing layer with residual connections.
    Adapted from GemNet/dimenet style interaction networks for molecular graphs.
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head attention for message passing
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Node update network
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge update network
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Layer norms
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: (N, hidden_dim)
            edge_features: (E, hidden_dim)
            edge_index: (2, E) source and target indices
        Returns:
            Updated node and edge features
        """
        src, dst = edge_index

        # Gather source and target features
        src_features = node_features[src]  # (E, hidden_dim)
        dst_features = node_features[dst]  # (E, hidden_dim)

        # Compute messages
        message_input = torch.cat([src_features, dst_features, edge_features], dim=-1)
        messages = self.message_net(message_input)

        # Aggregate messages at target nodes
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, dst, messages)
        degree = torch.zeros(node_features.size(0), 1, device=node_features.device)
        degree.index_add_(0, dst, torch.ones(messages.size(0), 1, device=node_features.device))
        aggregated = aggregated / (degree + 1e-8)

        # Update nodes with residual
        node_input = torch.cat([node_features, aggregated], dim=-1)
        node_update = self.node_update(node_input)
        node_features = self.node_norm(node_features + self.dropout(node_update))

        # Update edges with residual
        edge_input = torch.cat([src_features, dst_features, edge_features], dim=-1)
        edge_update = self.edge_update(edge_input)
        edge_features = self.edge_norm(edge_features + self.dropout(edge_update))

        return node_features, edge_features


class GNoMEModel(nn.Module):
    """
    Graph Networks for Materials Exploration (GNoME) model
    adapted for metallosilicon molecular topology prediction.

    Architecture:
    1. Node/edge embedding layers with elemental property encoding
    2. N interaction blocks for message passing
    3. Global pooling for graph-level predictions
    4. Multi-task heads: formation energy, stability, HOMO-LUMO gap
    """

    def __init__(
        self,
        n_elements: int = len(ALL_ELEMENTS),
        n_bond_types: int = 8,
        hidden_dim: int = 128,
        num_interactions: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_elements = n_elements
        self.hidden_dim = hidden_dim
        self.num_interactions = num_interactions

        # Node embedding: one-hot element + 4 elemental properties
        self.node_embedding = nn.Sequential(
            nn.Linear(n_elements + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge embedding: bond type + distance + 2 angle features
        self.edge_embedding = nn.Sequential(
            nn.Linear(n_bond_types + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Interaction blocks
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_interactions)
        ])

        # Global pooling
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Prediction heads
        self.formation_energy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.stability_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.homo_lumo_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        self.coordination_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),  # 4 geometry classes
        )

        self.solvent_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # NH3, CH4, H2S solvents
        )

    def _embed_nodes(self, atom_types: List[str]) -> torch.Tensor:
        """Convert element symbols to feature vectors."""
        n_atoms = len(atom_types)
        features = np.zeros((n_atoms, self.n_elements + 4))

        for i, elem in enumerate(atom_types):
            idx = ALL_ELEMENTS.index(elem) if elem in ALL_ELEMENTS else 0
            features[i, idx] = 1.0
            props = ELEMENT_PROPERTIES.get(elem, [1, 1.0, 50, 1])
            features[i, self.n_elements:] = [
                props[0] / 42.0,       # normalized atomic number
                props[1] / 4.0,         # normalized electronegativity
                props[2] / 160.0,       # normalized covalent radius
                props[3] / 10.0,        # normalized valence electrons
            ]

        return torch.tensor(features, dtype=torch.float32)

    def _embed_edges(
        self,
        bond_types: np.ndarray,
        distances: np.ndarray,
        angles: np.ndarray,
    ) -> torch.Tensor:
        """Convert bond information to feature vectors."""
        n_bonds = len(bond_types)
        n_bond_types = 8
        features = np.zeros((n_bonds, n_bond_types + 3))

        for i in range(n_bonds):
            bt = int(bond_types[i]) % n_bond_types
            features[i, bt] = 1.0
            features[i, n_bond_types] = distances[i] / 3.0      # normalized distance
            features[i, n_bond_types + 1] = angles[i] / np.pi   # normalized angle
            features[i, n_bond_types + 2] = np.cos(angles[i])   # angular feature

        return torch.tensor(features, dtype=torch.float32)

    def forward(
        self,
        atom_types: List[str],
        edge_index: torch.Tensor,
        bond_types: np.ndarray,
        distances: np.ndarray,
        angles: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GNoME model.

        Args:
            atom_types: List of element symbols
            edge_index: (2, E) edge indices
            bond_types: (E,) bond type indices
            distances: (E,) interatomic distances
            angles: (E,) bond angles

        Returns:
            Dictionary of predictions
        """
        # Embed nodes and edges
        node_features = self.node_embedding(self._embed_nodes(atom_types))
        edge_features = self.edge_embedding(self._embed_edges(bond_types, distances, angles))

        # Message passing through interaction blocks
        for interaction in self.interactions:
            node_features, edge_features = interaction(
                node_features, edge_features, edge_index
            )

        # Global pooling (mean + max)
        mean_pool = node_features.mean(dim=0, keepdim=True)
        max_pool = node_features.max(dim=0, keepdim=True).values
        graph_features = torch.cat([mean_pool, max_pool], dim=-1)  # (1, 2*hidden_dim)

        # Expand for per-node predictions
        graph_expanded = graph_features.expand(node_features.size(0), -1)

        # Multi-task predictions
        formation_energy = self.formation_energy_head(graph_features).squeeze(-1)
        stability = self.stability_head(graph_features).squeeze(-1)
        homo_lumo = self.homo_lumo_head(graph_features).squeeze(-1)
        coordination = self.coordination_head(graph_features)
        solvent = self.solvent_head(graph_features)

        return {
            "formation_energy": formation_energy,
            "stability": stability,
            "homo_lumo_gap": homo_lumo,
            "coordination_logits": coordination,
            "solvent_stability": solvent,
            "node_features": node_features,
            "graph_features": graph_features,
        }

    def predict(self, graph: MolecularGraph) -> PredictionResult:
        """
        Run full prediction pipeline on a MolecularGraph.

        Args:
            graph: Input molecular graph

        Returns:
            PredictionResult with all predicted properties
        """
        self.eval()
        with torch.no_grad():
            # Build edge index from adjacency
            adj = graph.adjacency
            edges = np.array(np.where(adj > 0))
            edge_index = torch.tensor(edges, dtype=torch.long)

            # Compute bond properties
            n_bonds = edge_index.size(1)
            if n_bonds > 0:
                src, dst = edges
                distances = np.linalg.norm(
                    graph.positions[dst] - graph.positions[src], axis=1
                )
                angles = np.zeros(n_bonds)  # simplified; real impl would compute
                bond_types = adj[src, dst].astype(int)
            else:
                distances = np.array([1.0])
                angles = np.array([0.0])
                bond_types = np.array([1])

            # Forward pass
            output = self.forward(
                graph.atom_types, edge_index, bond_types, distances, angles
            )

            # Decode coordination geometry
            geom_names = ["tetrahedral", "square_planar", "octahedral", "trigonal_prismatic"]
            coord_idx = output["coordination_logits"].argmax(dim=-1).item()
            coord_geom = geom_names[coord_idx]

            # Decode solvent stability
            solvent_vals = output["solvent_stability"].squeeze().tolist()
            solvent_dict = {
                "liquid_ammonia": float(solvent_vals[0]),
                "liquid_methane": float(solvent_vals[1]),
                "liquid_hydrogen_sulfide": float(solvent_vals[2]),
            }

            return PredictionResult(
                candidate_id=graph.candidate_id,
                formation_energy=float(output["formation_energy"].item()),
                stability_score=float(output["stability"].item()),
                hull_distance=max(0.0, float(output["formation_energy"].item())),
                homo_lumo_gap=float(output["homo_lumo_gap"].item()),
                coordination_geometry=coord_geom,
                solvent_stability=solvent_dict,
                graph=graph,
            )


class GNoMETrainer:
    """Training pipeline for GNoME model on metallosilicon molecular data."""

    def __init__(
        self,
        model: GNoMEModel,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def train_step(self, batch: List[MolecularGraph], targets: Dict[str, torch.Tensor]):
        """Single training step on a batch of molecular graphs."""
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        for graph, target in zip(batch, targets):
            adj = graph.adjacency
            edges = np.array(np.where(adj > 0))
            edge_index = torch.tensor(edges, dtype=torch.long).to(self.device)

            n_bonds = edge_index.size(1)
            if n_bonds > 0:
                src, dst = edges
                distances = np.linalg.norm(
                    graph.positions[dst] - graph.positions[src], axis=1
                )
                angles = np.zeros(n_bonds)
                bond_types = adj[src, dst].astype(int)
            else:
                distances = np.array([1.0])
                angles = np.array([0.0])
                bond_types = np.array([1])

            output = self.model(
                graph.atom_types, edge_index, bond_types, distances, angles
            )

            # Multi-task loss
            fe_loss = F.mse_loss(output["formation_energy"], target["formation_energy"])
            stab_loss = F.binary_cross_entropy(output["stability"], target["stability"])
            gap_loss = F.mse_loss(output["homo_lumo_gap"], target["homo_lumo_gap"])

            loss = fe_loss + 0.5 * stab_loss + 0.3 * gap_loss
            total_loss += loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item() / len(batch)


def create_pretrained_model(hidden_dim: int = 128, num_interactions: int = 6) -> GNoMEModel:
    """
    Create a GNoME model with Xavier-initialized weights.
    In production, this would load weights pre-trained on Materials Project data.
    """
    model = GNoMEModel(
        hidden_dim=hidden_dim,
        num_interactions=num_interactions,
    )

    # Xavier initialization for stable training
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    logger.info(f"Created GNoME model with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def save_model(model: GNoMEModel, path: str):
    """Save model weights and configuration."""
    config = {
        "hidden_dim": model.hidden_dim,
        "num_interactions": model.num_interactions,
        "n_elements": model.n_elements,
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str, device: str = "cpu") -> GNoMEModel:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = GNoMEModel(
        hidden_dim=config["hidden_dim"],
        num_interactions=config["num_interactions"],
        n_elements=config["n_elements"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {path}")
    return model
