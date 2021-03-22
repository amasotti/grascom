"""
Plotting functions

AT MOMENT JUST A RAW COPY OF THE ORIGINAL FILE....

DO NOT USE

"""
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import torch


class Plot(object):
    """Main entry point for visualizing the GSC results"""

    def __init__(self, nf, nr, inputNames, statesDict, fp_traces="data/full_traces.pt"):
        self.path = fp_traces

        self.nF = nf
        self.nR = nr
        self.epochs = 4

        self.inputNames = inputNames
        self.statesDict = statesDict
        self.statesDict_rev = {i: s for s, i in self.statesDict.items()}

        # Data
        self.data = self.load_traces()

        self.harmonies = None
        self.stateCounts = None
        self.statesNames = None
        self.states = None

        #sns.set_context(context="paper", font_scale=1.4)
        sns.set_style("darkgrid")

    # ----------------- PLOTTING FUNCS ---------------------------------

    def plot_harmonies(self, stim, epoch, save=False, outpath="data/plot"):
        """Plot harmonies

        Args:
        ------
            - stim : the stimulus number
            - epoch : the training epoch

        """

        # Retrieve states and state Names
        self.get_states(stim, epoch)

        # Create the df and plot
        df = pd.DataFrame({"StateNum": self.states, "Outputs": self.statesNames,
                           "H(output)": self.harmonies}, index=self.statesNames)
        harm_plot = sns.barplot(data=df, x="Outputs", y="H(output)")
        harm_plot.set_title(
            f"Output Harmonies (epoch {epoch}/stimulus {self.inputNames[stim]})")
        plt.show()

        # Save
        if save:
            fig = harm_plot.get_figure()
            fig.savefig(outpath + "Harmonies_ep" +
                        str(epoch) + "_stimulus_" + str(stim))

        return df

    def plot_freq(self, stim, epoch, save=False, outpath="data/plot"):
        """Plot Frequency

        Args:
        ------
            - stim : the stimulus number
            - epoch : the training epoch

        """

        # Retrieve states and state Names
        self.get_states(stim, epoch)

        statesNames = self.stateName_list(self.statesUnique)
        # Create the df and plot
        df = pd.DataFrame(
            {"Outputs": statesNames, "P(output)": self.stateCounts}, index=statesNames)

        # Plot
        prob_plot = sns.barplot(data=df, x="Outputs", y="P(output)")

        # Set title
        prob_plot.set(
            title=f"Output Frequencies (epoch {epoch}/stimulus {self.inputNames[stim]})")
        plt.show()

        # Save
        if save:
            fig = prob_plot.get_figure()
            fig.savefig(outpath + "Frequency_ep" +
                        str(epoch) + "_stimulus_" + str(stim))

        return df

    def plot_final_states(self, save=False, outpath="data/plots"):
        """ Plot harmonies and probabilities for the last epoch"""
        for i in range(len(self.inputNames)):
            self.plot_freq(i, self.epochs-1, save=save, outpath=outpath)
            self.plot_harmonies(i, self.epochs-1, save=save, outpath=outpath)

    # ----------------------------------------------------------------------

    def plot_harmonyProb(self, stim, epoch, save=False, outpath="data/plots", lm=True):
        """Plot the relation between harmony and frequency"""
        # Get data
        self.get_states(stim, epoch)

        # Retrieve rev indices for unique states
        unique, idx = torch.unique(torch.tensor(
            self.states), return_inverse=True)
        # reverse indices (from unique to original vector)
        idx_rev = torch.empty_like(unique)
        for ind, value in enumerate(unique.long()):
            idx_rev[ind] = torch.where(torch.tensor(
                self.states) == value)[0][0]

        harm_unique = torch.tensor(self.harmonies)[idx_rev.long()].tolist()
        names_unique = []
        for i in idx_rev.long().tolist():
            names_unique.append(self.statesNames[i])

        df = pd.DataFrame({"outputs": self.statesUnique,
                           "H(output)": harm_unique, "P(output)": self.stateCounts}, index=names_unique)

        # get coeffs of linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df["H(output)"], df["P(output)"])

        if lm:
            lm_plot = sns.lmplot(
                x="H(output)", y="P(output)", data=df, ci=None, x_estimator=np.mean)
            lm_plot.set(
                title="Correlation between Harmony and Quantization frequency")
            plt.show()
        else:
            lm_plot = sns.regplot(
                x="H(output)", y="P(output)", data=df, ci=None, line_kws={'label': "y={0:.1f}x+{1:.1f}".format(slope, intercept)})
            lm_plot.set(
                title="Correlation between Harmony and Quantization frequency")
            lm_plot.legend()
            # plt.legend()
            plt.show()

        # Save
        if save:
            fig = lm_plot.get_figure()
            fig.savefig(outpath + "Lm_plot_" + "stimulus_" +
                        str(stim) + "(epoch" + str(epoch) + ")")

        print(f"R-coeff: {r_value}")
        print(f"Slope: {slope}")
        print(f"Intercept: {intercept}")

        return df

    # ----------------------------------------------------------------------

    def plot_epoch(self, stim, epoch, save=False, outpath="data/plots/"):
        """Plot the outputs for a given stimulus
            at a given training epoch
        """
        # Load the data
        self.get_states(stim, epoch)

        timestep = torch.arange(0, len(self.states)).tolist()

        df = pd.DataFrame({"outputs": self.statesNames,
                           "step": timestep}, index=self.statesNames)
        epoch_plot = sns.scatterplot(
            data=df, x="step", y="outputs", palette="deep", s=4)
        # Set title
        epoch_plot.set(
            title=f"Training phase for stimulus {self.inputNames[stim]} (epoch {epoch})")
        plt.show()

        # Save
        if save:
            fig = epoch_plot.get_figure()
            fig.savefig(outpath + "Epoch_" +
                        str(epoch) + "_stimulus_" + str(stim))

        return df

    # ----------------------------------------------------------------------

    def plot_matrix(self, matrix, cmap="Blues", alpha=.8, save=False, outpath="data/plots/"):
        """Plot a matrix as heatmap"""
        matrix = matrix.numpy()

        mat_plot = sns.heatmap(matrix, annot=True, alpha=alpha, cmap=cmap)
        plt.show()

        if save:
            fig = mat_plot.get_figure()
            fig.savefig(outpath + "Matrix_representation")

    # ---------------- AUXILIARY FUNCS ------------------------

    def load_traces(self):
        """Load dictionary of traces"""
        traces = torch.load(self.path)
        return traces

    def get_states(self, stim, epoch):
        """Load the state numbers from the backup file

        This can be thought as unique identifier for a given sequence of
        bindings.

        The vectors are cutted at the "converged_at" index, since the training stopped there
        and the rest of the components are meaningless.

        """
        # Extract the tensor of state numbers
        tp_num = self.data['TPnum_trace']

        # Check at which step this stimulus converged:
        converged_at = int(self.data['reaction_times'][stim, epoch])
        # Extract indices, counts and stateNums
        states = tp_num[stim, epoch, 0:converged_at]

        # Counts
        unique, counts = torch.unique(states, return_counts=True)
        counts = counts / counts.sum()
        self.statesUnique = unique.tolist()
        self.stateCounts = counts.tolist()

        # reverse indices (from unique to original vector)
        idx_rev = torch.empty_like(states)
        for ind, value in enumerate(states.long()):
            idx_rev[ind] = torch.where(tp_num[stim, epoch, :] == value)[0][0]

        tph_trace = self.data['TP_h_trace']
        harmonies = tph_trace[stim, epoch, idx_rev.long()]

        self.states = states.tolist()
        self.statesNames = self.stateName_list(self.states)
        self.harmonies = harmonies.tolist()
        self.convergence_index = converged_at

    def stateName_list(self, states):
        stateNames = []
        for state in states:
            stateNames.append(self.find_names(state))
        return stateNames

    def find_names(self, stateNum):
        stateName = self.statesDict_rev[stateNum]
        return stateName

    def id_to_filler(self, stateNum, nR=4, nF=6):
        """Retrieve the state number given ordered winner fillers

        This is achieved treating the winning fillers a number in base nFillers 
        starting from the origin.
        """
        # translate the vector
        #stateNum -= 1
        winners = torch.zeros(self.nR).long()
        coefficients = torch.tensor(self.nF).pow(
            torch.arange(self.nR - 1, -1, -1))
        for i in range(self.nR):
            winners[i] = torch.floor(stateNum/coefficients[i])

            #stateNum = stateNum % coefficients[i]
            stateNum = torch.remainder(stateNum, coefficients[i])
        return winners


"""# --------------- TESTING AREA ------------------------------------------------
# +1 due to the empty filler added automatically
nF = len(N.grammar.fillers)
nR = len(N.grammar.roles)
statesDict = N.grammar.bind.states
inputNames = N.inputNames

plot = Plot(fp_traces="data/full_traces.pt", nf=nF,
            nr=nR, inputNames=inputNames, statesDict=statesDict)


a = plot.plot_harmonyProb(1, 4, lm=False)
b = plot.plot_epoch(1, 2)
"""
