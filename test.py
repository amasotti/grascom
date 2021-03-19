# # # Playground and testing
from src.classes.Grammar import Grammar
from src.gsc.gsc_network import Net
import torch

# Fillers and Roles
fillers = ["bh", "b", "u", "d", "dh"]
roles = ["s1", "s2", "s3", "s4", "s5"]
similarities = torch.tensor([[1, 0.75, 0, 0, 0, 0],
                             [0.75, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0.75, 0],
                             [0, 0, 0, 0.75, 1, 0],
                             [0, 0, 0, 0, 0, 1]])
# Build Grammar
G = Grammar(fillers, roles, emtpyFiller="#", fillerSimilarities=similarities)

# Single Harmony constraints
cons = [("#/s4", 2),  # Prefer simple codas
        ("#/s5", 2),
        ("d/s3", 1),  # Prefer aspirate in the coda
        ("dh/s3", 2),
        ("u/s2", 15)]

G.update_Hc(cons)

# Pairwise Harmony
cons = [("bh/s1", "b/s1", -5),
        ("d/s3", "dh/s3", -5),
        ("bh/s1", "dh/s3", -5),
        ("bh/s1", "d/s3", 5),
        ("dh/s4", "b/s1", 5)]
G.update_Hcc(cons)

N = Net(G)


p = N()
