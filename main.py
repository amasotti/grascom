# # # Playground and testing
import matplotlib.pyplot as plt
import torch

from src.classes.Grammar import Grammar
from src.gsc.gsc_network import Net
from src.gsc.plotting import Plot

# Set seed for reproducibility
torch.manual_seed(1111)

# ---------------------------------------
#       GRAMMAR AND CONSTRAINTS
# ---------------------------------------
# Fillers and Roles
fillers = ["bhud", "budh", "bhut", "bud", "bhudi", "budhi", "ta", "dha"]
roles = ["r", "s"]  # root & suffix

# Build Grammar
G = Grammar(fillers, roles, emtpyFiller="#")
# G = Grammar(fillers, roles, emtpyFiller="_", fillerSimilarities=similarities)


# Single Harmony constraints
# This is a matrix (nF, nR)

# 1. attempt : activations here
# distinguish between suffix and root (I know... very naive attempt)
"""cons = [("bhud/r", .6), ("budh/r", .9), ("bud/r", .4),
        ("bhut/r", .1), ("dha/s", .5), ("ta/s", .6),
        
        ("bhud/s", -10), ("budh/s", -10), ("bud/s", -10),
        ("bhut/s", -10),
        ("dha/r", -10), ("ta/r", -10)
        ]"""
# 2. Attempt : Harmonies here (agnostic) and activations in the input file
cons = [("bhud/r", 1), ("budh/r", 1), ("bud/r", 1), ("bhut/r", 1),
        ("bhudi/r", 1), ("bhudi/r", 1), ("dha/s", 1), ("ta/s", 1),
        ("bhud/s", -10), ("budh/s", -10), ("bud/s", -10), ("bhut/s", -10),
        ("bhudi/s", -10), ("bhudi/s", -10), ("dha/r", -10), ("ta/r", -10)]
G.update_Hc(cons)

# Pairwise Harmony
# Matrix dim: (nF, nR, nF, nR)
"""
# Distinguish between roots and suffixes (naive attempt, I know...)
suffRoot = [("bhud/r", "bhud/s", -10),  # bhud
            ("bhud/r", "budh/s", -10),
            ("bhud/r", "bhudi/s", -10),
            ("bhud/r", "budhi/s", -10),
            ("bhud/r", "bhut/s", -10),
            ("bhut/r", "bhut/s", -10),  # bhut
            ("bhut/r", "budh/s", -10),
            ("bhut/r", "bhudi/s", -10),
            ("bhut/r", "budhi/s", -10),
            ("budh/r", "bhud/s", -10),  # budh
            ("budh/r", "bhut/s", -10),
            ("budh/r", "budh/s", -10),
            ("budh/r", "bhudi/s", -10),
            ("budh/r", "budhi/s", -10),
            ("budhi/r", "bhut/s", -10),
            ("bhudi/r", "bhut/s", -10),
            ("bhudi/r", "budhi/s", -10),
            ("budhi/r", "budhi/s", -10),
            ("bhudi/r", "bhudi/s", -10),
            ("ta/s", "ta/s", -10),  # ta
            ("ta/s", "dha/s", -10)
            ]
G.update_Hcc(suffRoot)
"""

# Implements the Harmonic Constraints (see Optimization project written in Julia)
cons = [("bhud/r", "ta/s", -3),  # violates voice contrast
        ("budh/r", "ta/s", -3),  # violates voice contrast and License(laryngeal)
        ("bhud/r", "dha/s", -1),  # lazyness
        ("budh/r", "dha/s", -4),  # License & Lazyness
        ("bhut/r", "dha/s", -4),  # Lazyness and voice contrast
        ("bud/r", "dha/s", -1),  # Ident(lar)
        ("bud/r", "ta/s", - 4),  # voice contrast & Ident(lar)
        ("bhud/r", "ta/s", -10.5),
        ("bhudi/r", "ta/s", - 3),  # DepIO
        ("budhi/r", "ta/s", - 3),  # DepIO
        ("bhudi/r", "ta/s", - 5),  # DepIO & Ident(lar)
        ("budhi/r", "ta/s", - 5),  # DepIO & Ident(lar)
        ("bud/r", "dha/s", 10),  # Give the optimal candidate more harmony
        ]


G.update_Hcc(cons)

# ---------------------------------------
#           GSC NET
# ---------------------------------------

custom_settings = {"epochs": 3,
                   "tgtStd": 0.00125,
                   "emaFactor": 0.001,
                   "emaSpeedTol": .00051,
                   "dt": 1e-3,
                   "TDecayRate": 0.02,
                   "TInit": 8e-3,
                   "lambdaDecayRate": 0.75,
                   "lambdaMin": 0.01,
                   "maxSteps": 20000,
                   "printInterval": 10000,
                   'bowl_center': 0.4,
                   'beta_min_offset': .01,
                   'q_init': 4.5,
                   'clamp': False}
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

df = p.plot_act_stim(0, 2)
#df = p.plot_harmonies(0, 2)
