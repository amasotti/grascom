# # # Playground and testing
import matplotlib.pyplot as plt
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
roles = ["s1", "s2", "s3", "s4"]

"""similarities = torch.tensor([[1, 0.75, 0, 0, 0,0],
                             [0.75, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0.75, 0],
                             [0, 0, 0, 0.75, 1, 0],
                             [0, 0, 0, 0, 0, 1]])"""

# Build Grammar
G = Grammar(fillers, roles, emtpyFiller="#")
#G = Grammar(fillers, roles, emtpyFiller="_", fillerSimilarities=similarities)


# Single Harmony constraints
# This is a matrix (nF, nR)
# u should be the vowel, occupying the 2nd position
cons = [("u/s2", 5), ("d/s2", -4)]
G.update_Hc(cons)
"""

# Pairwise Harmony
# Matrix dim: (nF, nR, nF, nR)
cons = [("b/s1" "d/s2", -4),
        ("bh/s2", "dh/s3", -10),
        ("b/s1", "dh/s3", 10)]
G.update_Hcc(cons)"""


# ---------------------------------------
#           GSC NET
# ---------------------------------------

custom_settings = {"epochs": 2,
                   "tgtStd": 0.00125,
                   "emaFactor": 0.1,
                   "emaSpeedTol": .5,
                   "dt": 1e-5,
                   "TDecayRate": 0.05,
                   "TInit": 1e-5,
                   "lambdaDecayRate": 0.75,
                   "lambdaMin": 0.01,
                   "maxSteps": 20000,
                   "printInterval": 10000,
                   'bowl_center': 0.4,
                   'beta_min_offset': .01,
                   'q_init': 16}
# Initialize
N = Net(G, custom_settings=custom_settings, extData_path="data/inp_pandas.csv")


# Run
N()


# ---------------------------------------
#           Plots
# ---------------------------------------
last_epoch = custom_settings['epochs'] - 1
fp = "data/full_traces.pt"
nr = len(roles)
nf = len(fillers) + 1
statesDict = G.bind.states
inputNames = N.inputNames


p = Plot(fp_traces="data/full_traces.pt", nf=nf, nr=nr,
         inputNames=inputNames, statesDict=statesDict)

# Plot states
plt.figure(figsize=(16, 10))
df = p.plot_epoch(0, last_epoch)

plt.figure(figsize=(10, 8))
df = p.plot_final_states(save=True)
