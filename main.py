# # # Playground and testing
from src.gsc.plotting import Plot
from src.classes.Grammar import Grammar
from src.gsc.gsc_network import Net
import torch
# Set seed for reproducibility
torch.manual_seed(123)

# ---------------------------------------
#       GRAMMAR AND CONSTRAINTS
# ---------------------------------------
# Fillers and Roles
fillers = ["bh", "b", "u", "d", "dh"]
roles = ["s1", "s2", "s3"]

"""similarities = torch.tensor([[1, 0.75, 0, 0, 0, 0],
                             [0.75, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0.75, 0],
                             [0, 0, 0, 0.75, 1, 0],
                             [0, 0, 0, 0, 0, 1]])"""

# Build Grammar
G = Grammar(fillers, roles, emtpyFiller="_")
#G = Grammar(fillers, roles, emtpyFiller="_", fillerSimilarities=similarities)

# Single Harmony constraints
# This is a matrix (nF, nR)
cons = [("b/s1", 2), ("bh/s1", 2), ("u/s2", 5), ("d/s3", 2), ("dh/s3", 5)]
G.update_Hc(cons)
"""

# Pairwise Harmony
# Matrix dim: (nF, nR, nF, nR)
cons = [("b/s1" "u/s2", 50),
        ("u/s2", "d/s3", 50),
        ("bh/s1", "dh/s3", -50)]
G.update_Hcc(cons)
"""

# ---------------------------------------
#           GSC NET
# ---------------------------------------

custom_settings = {"epochs": 5,
                   "tgtStd": 0.00125,
                   "emaFactor": 0.05,
                   "emaSpeedTol": 0.002,
                   "dt": 1e-4,
                   "T_decay_rate": 0.05,
                   "maxSteps": 3000,
                   "printInterval": 1000}
# Initialize
N = Net(G, custom_settings=custom_settings)


# Run
p = N()

# ---------------------------------------
#           Plots
# ---------------------------------------
plot = Plot(net=N, fp_traces="data/full_traces.pt")
plot.plotTP_probs(stim=0, epoch=0)
print("Done")
