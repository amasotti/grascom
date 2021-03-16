"""
The Gradient Symbolic Computation Network class

"""
import torch
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
# Set seed for reproducibility
torch.seed(123)


class Net(object):
    def __init__(self, grammar):
        # The Harmonic Grammar
        self.grammar = grammar

        #  Set up the settings dictionary
        self._define_settings()

        # Harmony constraints
        self.Hc = self.grammar.Hc
        self.Hcc = self.grammar.Hcc

        # Preprare the network to be run
        self.setup_net()

    def _define_settings(self):
        """Set option variables to default values."""

        self.settings = {}

        ##### ORIGINALLY IN THE DOMAIN FILE #######

        # Bowl center
        self.settings['z'] = 0.3
        # bowl multiplier
        self.settings['q'] = 16.2
        # Maximum input in the C-Space : no constituent can be more than 100% present
        self.settings['maxInp'] = 1

        #### ORIGINALLY IN THE SETTINGS FILE #####
        self.settings["epochs"] = 5
        self.settings["timeStep"] = .0001
        self.settings["tgtStd"] = 0.00125
        self.settings["TInit"] = -1
        self.settings["TMin"] = 0
        self.settings["TdecayRate"] = 0.05
        self.settings["lambdaInit"] = 0.011387
        self.settings["lambdaMin"] = 0.01
        self.settings["lambdaDecayRate"] = 0.75
        self.settings["maxSteps"] = 60000
        self.settings["emaSpeedTol"] = 0.002
        self.settings["emaFactor"] = .05
        self.settings["diary"] = False
        self.settings["printInterval"] = 3000
        self.settings["saveFile"] = 'Simulations/grassman.txt'
        mean = torch.eye(self.grammar.bind.nF,
                         self.grammar.bind.nR)/self.grammar.bind.nF
        self.settings["initStateMean"] = mean
        self.settings["initStateStdev"] = .025
        #self.settings["sequentialMode"] = False

    def readInput(self, fp="data/inp_pandas.csv"):
        inputs = pd.read_csv(fp, sep=",")
        self.nStimuli = len(inputs['id'].unique())
        self.simuli = torch.zeros(
            (self.nStimuli, self.grammar.bind.nF, self.grammar.bind.nR))
        for idx, i in enumerate(inputs['id'].unique()):
            stimulus = inputs[inputs.id == i].to_numpy()[:, 1:]
            for filler in stimulus:
                fidx = self.grammar.bind.fillers.index(filler[0])
                self.stimuli[idx, fidx, :] = stimulus[1:]

    def train(self, stimuli):  # TODO:
        pass

    def __call__(self, stimuli, plot=False):  # TODO:
        self.readInput(stimuli)
        self.train()
        if plot:
            self.plot()

    def plot(self):  # TODO:
        pass
