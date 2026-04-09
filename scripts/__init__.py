# Metallosilicon Amino Acid Discovery Pipeline
# GNoME + Pymatgen Integration for Prosilicon Biological Frameworks

"""
Pipeline for generating, screening, and optimizing metallosilicon amino acid analogs
suitable for cryogenic, reducing environments (liquid methane/ammonia solvents).

Core Components:
- GNoME Graph Neural Network for stability prediction
- Molecular graph generator with elemental constraints (Si, N, H, S, P, B, F, Fe/Ni/Ti/Mo)
- Pymatgen DFT relaxation workflow
- Phase diagram analysis with convex hull filtering
"""

__version__ = "1.0.0"
__author__ = "Silicon Biogeochemistry Lab"
