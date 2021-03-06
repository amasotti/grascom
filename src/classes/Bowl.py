"""
The Bowl class

__date__ : Februar 201
__author__ : A. Masotti (on the model of LDNet 1.5)
"""
import torch


class Bowl(object):
    def __init__(self, GSCNet):
        self.Net = GSCNet
        self.center = self.Net.vars['bowl_center'] * \
            torch.ones(self.Net.nSym, dtype=torch.double)
        self.strength = self.recommended_strength_Matlab()
        # Alternative: this here depends on the external input. If you want to reuse this, you should
        # change the order in which variables are initialized.
        # self.strength = self.recommend_strength()  # Python version

    def recommend_strength(self):
        """Calculate the recommended strength for the Bowl.

        This value depends on the external input and will be used either to set the strength of the bowl
        or to check that the chosen values allows the training to converge.

        This value is crucial since the final weight matrix should be negative-definite.
        This value is exactly what ensures that.

        """
        eigenvalues = torch.linalg.eigvalsh(self.Net.Wc)
        largest_eigval = torch.max(eigenvalues)

        if torch.sum(self.center.sum()) > 0:
            if self.Net.nSym == 1:
                beta1 = -(self.Net.Bc + self.Net.inpC) / self.center
                beta2 = (self.Net.Bc + self.Net.inpC +
                         largest_eigval) / (1-self.center)
            else:
                beta1 = torch.min(
                    (self.Net.Bc + self.Net.inpC) / self.center) * -1
                beta2 = torch.max(
                    (self.Net.Bc + self.Net.inpC + largest_eigval) / (1 - self.center))
                value = max(largest_eigval, beta1, beta2)
        else:
            value = largest_eigval

        return value

    def __repr__(self):
        return f"Bowl object with strength {self.strength} and center {self.center}"

    def recommended_strength_Matlab(self):
        """The Matlab version of the function to calculate the recommended Q value"""

        eigMax = torch.max(torch.linalg.eigvalsh(self.Net.Wc))
        q_nd = max(0, eigMax)

        beta_min = -(torch.min(self.Net.Bc) -
                     self.Net.settings['maxInp'])/self.Net.vars['bowl_center']
        beta_max = torch.max(self.Net.Bc) + self.Net.settings['maxInp']
        beta_max = (beta_max + eigMax)/(1 - self.Net.vars['bowl_center'])

        q_rec = max([torch.max(beta_min), torch.max(beta_max), q_nd])

        return q_rec

    def set_biases(self):
        """Set the biases for the Bowl

        The biases in the conceptual space are just
            strength * center * ones(nSymbols)

        In the S-Space the biases are Harmony preserving (no-crosstalk)

        """
        # Matlab version
        bowl_biasesC = float(self.Net.vars['bowl_strength']) * float(
            self.Net.vars['bowl_center']) * torch.ones(self.Net.nSym, 1)
        bowl_biasesS = self.Net.TPinv.T.matmul(bowl_biasesC.double())

        return bowl_biasesC, bowl_biasesS

    def set_weights(self):
        """Set bowl weights.

        Weight in the conceptual space: This is just the Identity matrix
        times the strength.

        The Weights in the Sspace are Harmony-preserving (No cross-talk)
        """

        WC = -float(self.Net.vars['bowl_strength']) * \
            torch.eye(self.Net.nSym, dtype=torch.double)
        WS = self.Net.TPinv.T.mm(WC).mm(self.Net.TPinv)

        return WC, WS

    def calc_bowl_harmony(self, state=None):
        "Compute the bowl harmony"
        if state is None:
            state = self.Net.state

        z = self.Net.vars['zeta_bowl']
        Hbowl = -0.5 * (state - z).T @ self.Net.Gc @ (state - z)
        Hbowl *= self.strength
        return Hbowl

    def calc_bowl_gradient(self, state=None):
        "Compute the bowl harmony gradient"
        if state is None:
            state = self.Net.state

        z = self.Net.vars['zeta_bowl']
        grad_bowl = self.strength * (-self.Net.Gc @ state + self.Net.Gc @ z)
        return grad_bowl
