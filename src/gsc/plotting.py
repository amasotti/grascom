"""
Plotting functions

AT MOMENT JUST A RAW COPY OF THE ORIGINAL FILE....

DO NOT USE

"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch


class Plot(object):
    """Main entry point for visualizing the GSC results"""

    def __init__(self, net, fp_traces):
        self.path = fp_traces
        self.net = net

        self.nF = self.net.grammar.nF
        self.nR = self.net.grammar.nR

        # Data
        self.data = self.load_traces()

    def load_traces(self):
        """Load dictionary of traces"""
        traces = torch.load(self.path)
        return traces

    def plotTP_probs(self, stim, epoch, save=True, outpath="data/plots/"):
        """Plot for a given stimulus and a given training epoch"""
        # Extract the tensor of state numbers
        tp_num = self.data['TPnum_trace']

        # Check at which step this stimulus converged:
        converged_at = int(self.data['reaction_times'][stim, epoch])
        # Extract indices, counts and stateNums
        states, idx, counts = torch.unique(
            tp_num[stim, epoch, 0:converged_at], return_inverse=True, return_counts=True, sorted=True)

        # reverse indices (from unique to original vector)
        idx_rev = torch.empty_like(states)
        for ind, value in enumerate(states.long()):
            idx_rev[ind] = torch.where(tp_num[stim, epoch, :] == value)[0][0]

        tph_trace = self.data['TP_h_trace']
        harmonies = tph_trace[stim, epoch, idx_rev.long()]
        # Counts sorted
        sortVal, sortInd = torch.sort(counts)
        # Harmonies sorted
        self.harmonies = harmonies[sortInd]
        # states sorted
        states_sorted = states[sortInd]
        self.states = states_sorted.long().tolist()
        self.statesNames = self.stateName_list(self.states)
        self.stateCounts = (
            sortVal.long() / torch.sum(sortVal).long()).tolist()

        df = pd.DataFrame({"p(state)": self.stateCounts,
                           "Outputs": self.statesNames})

        prob_plot = sns.barplot(x="Outputs", y="p(state)", data=df)
        prob_plot.set_title(
            f"Output Probability (epoch {epoch}/stimulus {self.net.inputNames[stim]})")
        if save:
            fig = prob_plot.get_figure()
            fig.savefig(outpath + "probabilities_ep" +
                        str(epoch) + "_stimulus_" + str(stim))
        plt.show()

        df = pd.DataFrame({"H(state)": self.harmonies,
                           "Outputs": self.statesNames})
        harm_plot = sns.barplot(x="Outputs", y="H(state)", data=df)
        harm_plot.set_title(
            f"Output Harmonies (epoch {epoch}/stimulus {self.net.inputNames[stim]})")
        if save:
            fig = harm_plot.get_figure()
            fig.savefig(outpath + "harmonies_ep" +
                        str(epoch) + "_stimulus_" + str(stim))

        plt.show()

    def stateName_list(self, states):
        stateNames = []
        for state in states:
            stateNames.append(self.find_names(state))
        return stateNames

    def find_names(self, stateNum):
        winners = self.id_to_filler(stateNum=torch.tensor(stateNum))
        stateName = self.net.find_TPname(winners)
        return stateName

    def id_to_filler(self, stateNum, nR=4, nF=6):
        """Retrieve the state number given ordered winner fillers

        This is achieved treating the winning fillers a number in base nFillers 
        starting from the origin.
        """
        # translate the vector
        winners = torch.zeros(self.nR).long()
        coefficients = torch.tensor(self.nF).pow(
            torch.arange(self.nR - 1, -1, -1))
        for i in range(self.nR):
            winners[i] = torch.floor(stateNum/coefficients[i])
            stateNum = stateNum % coefficients[i]
        return winners
