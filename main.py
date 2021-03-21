# # # Playground and testing
from src.gsc.plotting import Plot
from src.classes.Grammar import Grammar
from src.gsc.gsc_network import Net
import torch
# Set seed for reproducibility
# torch.manual_seed(111)

# ---------------------------------------
#       GRAMMAR AND CONSTRAINTS
# ---------------------------------------
# Fillers and Roles
fillers = ["bh", "b", "u", "d", "dh"]
roles = ["s1", "s2", "s3", "s4"]

similarities = torch.tensor([[1, 0.75, 0, 0, 0, 0],
                             [0.75, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0.75, 0],
                             [0, 0, 0, 0.75, 1, 0],
                             [0, 0, 0, 0, 0, 1]])

# Build Grammar
# G = Grammar(fillers, roles, emtpyFiller="#")
G = Grammar(fillers, roles, emtpyFiller="_", fillerSimilarities=similarities)

# Single Harmony constraints
# This is a matrix (nF, nR)
cons = [("_/s4", 3), ("u/s2", 15), ("b/s1", 3), ("d/s1", 30), ("d/s1", 3.5)]
G.update_Hc(cons)


"""
# Pairwise Harmony
# Matrix dim: (nF, nR, nF, nR)
cons = [("bh/s1", "d/s3", 2),
        ("dh/s4", "b/s1", 5),
        ("u/s2", "dh/s3", 5)]
G.update_Hcc(cons)
"""

# ---------------------------------------
#           GSC NET
# ---------------------------------------

custom_settings = {"epochs": 3,
                   "tgtStd": 1e-4,
                   "emaFactor": 0.05,
                   "dt": 5e-3,
                   "T_decay_rate": 1.25e-2,
                   "maxSteps": 7000}
# Initialize
N = Net(G, custom_settings=custom_settings, extData_path="data/inp2.csv")


# Run
p = N()

# ---------------------------------------
#           Plots
# ---------------------------------------
plot = Plot(net=N, fp_traces="data/full_traces.pt")
plot.plotTP_probs(stim=0, epoch=2)
print("Done")
