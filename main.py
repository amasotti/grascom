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


"""# Single Harmony constraints
# This is a matrix (nF, nR)
cons = [("u/s2", 5)]  # u should be the vowel, occupying the 2nd position
G.update_Hc(cons)


# Pairwise Harmony
# Matrix dim: (nF, nR, nF, nR)
cons = [("b/s1" "d/s2", -4),
        ("bh/s2", "dh/s3", -10),
        ("b/s1", "dh/s3", 10)]
G.update_Hcc(cons)"""


# ---------------------------------------
#           GSC NET
# ---------------------------------------

custom_settings = {"epochs": 5,
                   "tgtStd": 0.00125,
                   "emaFactor": 0.0089,
                   "emaSpeedTol": 0.002,
                   "dt": 1e-4,
                   "TDecayRate": 0.05,
                   "lambdaDecayRate": 0.75,
                   "lambdaMin": 0.01,
                   "maxSteps": 30000,
                   "printInterval": 10000,
                   'bowl_center': 0.4,
                   'beta_min_offset': 2,
                   'q_init': 16.58}
# Initialize
N = Net(G, custom_settings=custom_settings, extData_path="data/inp_pandas.csv")


# Run
N()

# ---------------------------------------
#           Plots
# ---------------------------------------
nF = len(fillers) + 1  # +1 due to the empty filler added automatically
nR = len(roles)
statesDict = G.bind.states
inputNames = N.inputNames

plot = Plot(fp_traces="data/full_traces.pt", nf=nF, nr=nR,
            inputNames=inputNames, statesDict=statesDict)
"""
plot.plot_act_harmony(0, 1)
plot.plot_final_states()
plot.plot_epoch(2, 0)"""

df = plot.plot_input_tstep(3, what="harmony_dev")
