# # # Playground and testing
from src.classes.Grammar import Grammar
from src.gsc.gsc_network import Net

# Fillers and Roles
fillers = ["bh", "b", "u", "d", "dh", "p", "-", "c", "a"]
roles = ["s1", "s2", "s3", "s4", "s5"]

# Build Grammar
G = Grammar(fillers, roles, emtpyFiller="#")

# Single Harmony constraints
cons = [("#/s4", 2),  # Prefer simple codas
        ("#/s5", 2),
        ("d/s3", 1),  # Prefer aspirate in the coda
        ("dh/s3", 2)]
G.update_Hc(cons)

# Pairwise Harmony
cons = [("bh/s1", "b/s1", -5),
        ("d/s3", "dh/s3", -5),
        ("bh/s1", "dh/s3", -10),
        ("bh/s1", "d/s3", 5),
        ("dh/s4", "b/s1", 5)]
G.update_Hcc(cons)

N = Net(G)


p = N()
