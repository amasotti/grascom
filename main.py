# # # Playground and testing
from src.classes.Grammar import Grammar
from src.gsc.gsc_network import Net
import torch
# Set seed for reproducibility
torch.manual_seed(123)

#---------------------------------------
#       GRAMMAR AND CONSTRAINTS
#---------------------------------------
# Fillers and Roles
fillers = ["bh", "b", "u", "d", "dh"]
roles = ["s1", "s2", "s3", "s4"]
"""
similarities = torch.tensor([[1, 0.75, 0, 0, 0, 0],
                             [0.75, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0.75, 0],
                             [0, 0, 0, 0.75, 1, 0],
                             [0, 0, 0, 0, 0, 1]])
"""
# Build Grammar
G = Grammar(fillers, roles, emtpyFiller="#")
#G = Grammar(fillers, roles, emtpyFiller="#", fillerSimilarities=similarities)

# Single Harmony constraints
# This is a matrix (nF, nR)
"""cons = [("#/s4", 1),  # Prefer simple codas
        ("#/s3", .1),
        ("bh/s1", 1),  # b, bh are good onsets. This constraints a little bit the search after well-formed structures.
        ("b/s1", 1),
        ("d/s3", 1),
        ("dh/s3", 1), # d,dh are good codas
        ("u/s2", 5)]
G.update_Hc(cons)"""

# Pairwise Harmony
# Matrix dim: (nF, nR, nF, nR)
"""cons = [("bh/s1", "d/s3", 2),
        ("b/s3", "dh/s3", 3),
        ("bh/s1", "dh/s3", -5),
        ("dh/s4", "u/s2", 5),
        ("dh/s4", "b/s1", 5)]
G.update_Hcc(cons)"""


#---------------------------------------
#           GSC NET
#---------------------------------------
# Initialize
N = Net(G)

# Run
p = N()
