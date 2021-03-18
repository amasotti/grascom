# # # Playground and testing
from src.classes.Grammar import Grammar

from src.gsc.gsc_network import Net

fillers = ["bh", "b", "u", "d", "dh", "p"]
roles = ["s1", "s2", "s3", "s4", "s5"]

G = Grammar(fillers, roles)
# scons = [("bh/r1", -2), ("b/r1", -2), ("d/r3", 1), ("dh/r3", -3)]
cons = [("bh/s1", "b/s1", -2), ("d/s3", "dh/s3", -1),
        ("bh/s1", "dh/s3", -1.5), ("bh/s1", "d/s3", 1.5), ("dh/s4", "b/s1", 7.5)]
G.update_Hcc(cons)

N = Net(G)
