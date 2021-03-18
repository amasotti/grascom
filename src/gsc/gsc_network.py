"""
The Gradient Symbolic Computation Network class

"""
from src.classes.utilFunc import fortran_reshape
import torch
import numpy as np
import math
#import matplotlib.pyplot as plt
#import seaborn
import pandas as pd
# Set seed for reproducibility
torch.manual_seed(123)


class Net(object):
    """The GSC Network"""

    def __init__(self, grammar):
        # The Harmonic Grammar
        self.grammar = grammar
        self.R = self.grammar.bind.R
        self.F = self.grammar.bind.F
        self.nSym = self.grammar.bind.nF * self.grammar.bind.nR

        # Dictionaries
        # Fillers
        self.filler2index = self.grammar.fillers.filler2index
        self.index2filler = self.grammar.fillers.index2filler

        # Roles
        self.role2index = self.grammar.roles.role2index
        self.index2role = self.grammar.roles.index2role

        # Bindings
        self.bind2index = self.grammar.bind.bind2index
        self.index2bind = self.grammar.bind.index2bind

        #  Set up the settings dictionary
        self._define_settings()
        # Variables that will change/be updated while training
        self._training_vars()

        # Harmony constraints
        self.Hc = self.grammar.Hc
        self.Hcc = self.grammar.Hcc

        # Preprare the network to be run
        self.stimuli = None
        self.setup_net()

    # -----------------------  GENERAL SETTINGS ------------------------------

    def _define_settings(self):
        """Set settings to default values."""

        self.settings = {}

        ##### ORIGINALLY IN THE DOMAIN FILE #######

        # Maximum input in the C-Space : no constituent can be more than 100% present
        self.settings['maxInp'] = 1

        #### ORIGINALLY IN THE SETTINGS FILE #####
        self.settings["epochs"] = 5  # Training epochs
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
        self.settings["summary_file"] = "data/summary.txt"
        mean = torch.eye(self.grammar.bind.nF,
                         self.grammar.bind.nR)/self.grammar.bind.nF
        self.settings["initStateMean"] = mean
        self.settings["initStateStdev"] = .025
        self.settings['clamped'] = False

    def _training_vars(self):
        """Variables to keep track in the learning phase"""
        self.vars = dict()
        self.vars["var_names"] = [
            'Activation', 'Harmony', 'H0', 'Q', 'q', 'Temp', 'time', 'ema_speed', 'speed']

        self.vars['norm_ord'] = torch.tensor([float('inf')])
        self.vars['coord'] = 'N'

        # EMA
        self.vars['ema_factor'] = 0.001
        self.vars['ema_tau'] = -1 / math.log(self.vars['ema_factor'])

        # Temperature params
        self.vars['T_init'] = 1e-3
        self.vars['T_min'] = 0.
        self.vars['T_decay_rate'] = 1e-3
        # Bowl params
        self.vars['q_init'] = 2  # initial strength for the bowl
        self.vars['q_max'] = 200.
        self.vars['q_rate'] = 10.
        self.vars['c'] = 0.5
        self.vars['bowl_center'] = 0.5
        self.vars['bowl_strength'] = None
        self.vars['beta_min_offset'] = 0.1
        # Harmony params
        self.vars['H0_on'] = True
        self.vars['H1_on'] = True
        self.vars['Hq_on'] = True
        self.vars['max_dt'] = 0.01
        self.vars['min_dt'] = 0.0005
        self.vars['q_policy'] = None
        # Time step params
        self.vars['time'] = 0
        self.vars['dt'] = 0.001

    def reset(self):
        pass  # TODO:

    # ----------------------- EXTERNAL INPUT  ----------------------------

    def readInput(self, fp="data/inp_pandas.csv"):
        """Read external input.

        Inputs are provided in a csv table with header:
        id,filler,r1,r2,....,rn

        The id marks the input as whole: many lines with the same ID count as 1 input

        requires Pandas
        """
        # Read dataframe
        inputs = pd.read_csv(fp, sep=",")

        self.nStimuli = len(inputs['id'].unique())
        # Initialize stimuli tensor
        self.stimuli = torch.zeros(
            (self.nStimuli, self.grammar.bind.nF, self.grammar.bind.nR))

        # Loop over the single inputs as whole
        for idx, i in enumerate(inputs['id'].unique()):
            inp_string = ""
            stimulus = inputs[inputs.id == i].to_numpy()[:, 1:]

            # Loop over the fillers in a given input
            for filler in stimulus:
                fidx = self.grammar.bind.fillers.index(filler[0])
                inp_string += filler[0] + "-"
                for roledix in range(self.grammar.bind.nR):
                    self.stimuli[idx, fidx, roledix] = filler[roledix+1]
            print(f"Input processed: {inp_string[:-1]}\n")

    # -----------------------  RETRIEVE BINDINGS -------------------------
    def find_bindings(self, bindName):
        """Retrieve the index of a specific binding"""
        try:
            return self.bind2index[bindName]
        except KeyError:
            raise f"The binding {bindName} is not in the general list... check your input file!"

    def find_role(self, roleName):
        """Retrieve the index of a specific role"""
        try:
            return self.role2index[roleName]
        except KeyError:
            raise f"The role {roleName} is not in the general list... check your input file!"

    def find_filler(self, fillerName):
        """Retrieve the index of a specific binding"""
        try:
            return self.filler2index[fillerName]
        except KeyError:
            raise f"The Filler {fillerName} is not in the general list... check your input file!"

    # -----------------------  SETUP ------------------------------------
    # -----------------------  ENCODINGS --------------------------------
    def _set_encodings(self):
        """Set default variables to default values"""
        self.encodings = dict()
        # Similarity between fillers / roles
        self.encodings['dotP_fillers'] = 0.0
        self.encodings['dotP_roles'] = 0.0

        self.encodings['coord_fillers'] = 'local'
        self.encodings['coord_roles'] = 'local'
        self.encodings['dim_fillers'] = None
        self.encodings['dim_roles'] = None
        self.encodings['fillers_names'] = list(self.filler2index.keys())
        self.encodings['roles_names'] = list(self.role2index.keys())
        self.encodings['FillerSimilarities'] = self.fillers.similarities

    def update_encodings(self, new_encodings):
        for key, value in new_encodings.items():
            if key in self.encodings:
                self.encodings[key] = value
            else:
                print("WARNING: Cannot update a non-existing in Net.encodings!")

    # ----------------------- SETUP ROUTINE ------------------------------------

    def setup_net(self):
        # Create the change-of-basis matrix
        # This will allow us to change from neural distributed to local symbolic repr.
        self.generateTP()

        # Generate Encodings for Fillers and Roles

        # Initialize weights and biases
        self.W = self.compute_neural_weights()
        self.B = self.compute_neural_biases()

        # Activation and Inputs for the RNN-like network
        self.activationC = torch.zeros(
            self.nSym, dtype=torch.double)
        self.activationC_prev = torch.zeros(
            self.nSym, dtype=torch.double)

        self.activation = self.toNeural()
        self.activation_prev = self.toNeural(matrix=self.activationC_prev)

        self.externalInpC = torch.zeros(self.nSym, dtype=torch.double)
        self.externalInpC_prev = torch.zeros(
            self.nSym, dtype=torch.double)

        self.externalInp = self.toNeural(self.externalInpC)
        self.externalInp_prev = self.toNeural(self.externalInpC_prev)

        # Construct weights and biase matrices for the Neural space
        self._set_weights()
        self._set_bias()
        self._set_quantList()
        self.debug_copies()

        # Bowl
        self.bowl = Bowl(self)
        self.vars['bowl_center'] = self.bowl.center
        self.vars['bowl_strength'] = self.bowl.strength + \
            self.vars['beta_min_offset']
        self.vars['zeta_bowl'] = self.toNeural(self.vars['bowl_center'])

        # Calculate the recommended Lambda and Temp values:
        self.check_Q_T_lambda()

        # Log default parameters
        self.logger(default_settings="#"*25 + " DEFAULT SETTINGS" + "#"*25)
        self.logger(default_settings=self.__dict__)

    # ---------------------- MATRIX FACTORY --------------------
    @ staticmethod
    def initializer(shape, dist="zero"):
        """Matrix initializer.

        Given a shape, it returns a matrix initialized with all zeros (default)
        or with values taken from a random distribution.

        Accepted are:
            - zeros
            - ones
            - identity matrix
            - random uniform
            - log normal
            - gamma
        """
        if dist == "zero":
            M = torch.zeros(shape, dtype=torch.double)
        elif dist == "one":
            M = torch.ones(shape, dtype=torch.double)
        elif dist == "id":
            M = torch.eye(shape, dtype=torch.double)
        elif dist == "uniform":
            M = torch.rand(shape, dtype=torch.double)
        elif dist == "log_normal":
            m = torch.distributions.log_normal.LogNormal(
                0, 1)
            M = m.sample(shape).double()
        elif dist == "gamma":
            m = torch.distributions.gamma.Gamma(1, 1)
            M = m.sample(shape).double()
        else:
            print("The distribution you gave is unknown... Matrix initialized with zeros")
        return M

    def generateTP(self):
        """Generate the Matrices for the change-of-basis from Neural to Conceptual
            and the other way round 
        """
        # Approximate to Kronecker Product
        self.TP = torch.kron(self.R, self.F).double()
        # create the inverse if TP is a square matrix:
        # Use the Moore-Penrose pseudoinverse if TP is not square
        self.TPinv = torch.linalg.pinv(self.TP, hermitian=True)
        self.Gc = torch.mm(self.TPinv.T, self.TP)

    def compute_neural_biases(self, dist="zero"):
        """Compute the Biases Matrix for the Neural space.

            This will be derived from the Harmonies specified in self.Hc
            (the single constituent Harmony) and will then be transformed
            into the B matrix for the Constituent space through simple 
            matrix multiplication

        """
        # flatten
        harmonies = fortran_reshape(self.Hc, (torch.numel(self.Hc), 1))
        #harmonies = self.Hc.reshape((torch.numel(self.Hc), 1))

        # Initialize
        biases = self.initializer((self.nSym, 1), dist=dist)

        # Update
        for i in range(self.nSym):
            b_i = self.TP[:, i]
            update_value = (harmonies[i] * b_i) / torch.matmul(b_i.T, b_i)
            biases += update_value.reshape((self.nSym, 1))
        return biases

    def compute_neural_weights(self, dist="zero"):
        """Compute the W Matrix for the Neural space.

            This will be derived from the Harmonies specified in self.Hcc
            (the pairwise constituent Harmony) and will then be transformed
            into the W matrix for the Constituent space through simple 
            matrix multiplication

            #0.5 in the case i == j (because of the symmetry)
        """
        # TODO: Should this begin with zeros? Or can we initialize the matrices with random dist?
        harmonies = fortran_reshape(self.Hcc, (self.nSym, self.nSym))
        W = self.initializer((self.nSym, self.nSym), dist=dist)

        # Update using the Hcc infos:
        for i in range(self.nSym):
            w_i = self.TP[:, i]  # take the i-th binding
            for j in range(i+1):  # just operate in the lower triangle, the rest is symmetric
                w_j = self.TP[:, j]  # take the j-th binding
                if i != j:
                    W += (harmonies[i, j] * (w_i.matmul(w_j.T) + w_j.matmul(w_i.T))
                          ) / (w_i.T.matmul(w_i) * w_j.T.matmul(w_j))
                else:
                    W += .5 * (harmonies[i, j] * (w_i.matmul(w_j.T) + w_j.matmul(w_i.T))) / (
                        w_i.T.matmul(w_i)*w_j.T.matmul(w_j))
        return W

    def _set_weights(self):
        """Transfor neural weights into conceptual weights."""
        self.Wc = torch.mm(self.TP.T, self.W).mm(self.TP)

    def _set_bias(self):
        """Conceptual Bias Matrix from S-biases

            Bc is a n-dimensional array, n = num Bindings
            TP is the change of Base matrix

        """
        # Alternative
        # self.B = self.TPinv.T.matmul(self.Bc)
        self.Bc = self.TP.T.matmul(self.B)

    def debug_copies(self):
        """Create debug copies of the initialized matrices
        #TODO: Why? Can we delete this?
        """
        self.b_debug = self.B.clone().detach()
        self.Bc_debug = self.Bc.clone().detach()
        self.W_debug = self.W.clone().detach()
        self.Wc_debug = self.Wc.clone().detach()

    def _set_quantList(self):
        """Quantization list"""
        self.quantList = []
        for _, index in self.role2index.items():
            self.quantList.append(self.R[:, index])

    # ------------------RECOMMENDED VALUES FOR Q, L, T ------------------

    def check_Q_T_lambda(self):
        """Check the bowl parameters."""
        # Substituted with Bowl.recommend_strength()
        #self.vars['q_rec'], self.vars['q_rec_nd'] = self.recommend_Q()

        self.vars['lambda_rec'] = self.recommend_L()
        # Check lambdas
        # 1e-2 tolerance #TODO: review tol
        if abs(self.settings['lambdaInit'] - self.vars['lambda_rec']) > 0.01:
            print(
                f"LAMBDA RECOMMENDED: {self.vars['lambda_rec']}, ACTUAL LAMBDA = {self.settings['lambdaInit']}")
            choice = input(
                "If you want to change to the recommended value press 'y', else any other key:")
            if choice.lower() == 'y':
                self.settings['lambdaInit'] = self.vars['lambda_rec']
                # TODO: clean this chaos up.... What does belong to settings, what to vars, what to encodings?
                self.vars['lambdaInit'] = self.vars['lambda_rec']
                self.logger(new_lambda=self.vars['lambda_rec'])

        self.vars['T_rec'] = self.recommend_T()
        # Check Temperatures
        if abs(self.vars['T_init'] - self.vars['T_rec']) > 1e-07:  # 1e-7 tolerance
            print(
                f"T RECOMMENDED: {self.vars['T_rec']}, ACTUAL T = {self.vars['T_init']}")
            choice = input(
                "If you want to change to the recommended value press 'y', else any other key:")
            if choice.lower() == 'y':
                self.vars['T_init'] = self.vars['T_rec']
                self.logger(new_temperature=self.vars['T_rec'])

    def recommend_Q(self):
        """ The value of the param Q ensures that the weights of the final weight matrix
        are negative definite, which is an important condition for the distribution to be stationary

        The mechanics is pretty simple: we must just find a value that is larger than the greatest positive eigvalue in Wc

        #TODO: The eigvectors and eigvals differ between MATLAB, Numpy and Pytorch probably due to different
        algorithms viz. error in the numerical algorithm. I should check if this is an issue or it works.

        """
        eigvals_wc = torch.linalg.eigvalsh(self.Wc)
        max_eigvalue = torch.max(eigvals_wc)
        q_nd = max(0, max_eigvalue)
        print(f"Eigenvalues of the weight matrix found: \n {eigvals_wc}\n")

        # Find the smallest q that puts the Harmony maximum between [0,1]
        bmin = torch.min(self.Bc) - self.settings['maxInp']
        bmax = torch.max(self.Bc) + self.settings['maxInp']

        # Use the bowl parameter to calculate q
        q_0 = -bmin/self.domain.z
        q_1 = (bmax + max_eigvalue) / \
            (1 - self.vars['bowl_center'])  # Will this work?
        q_range = torch.max([q_0, q_1, q_nd])

        return q_range, q_nd

    def recommend_L(self):
        """Recommended lambda init value

        This function is mathematically problematic. See Issue #2
        # TODO: see above eigvals Pytorch
        """

        min_eigenvalue = torch.min(torch.linalg.eigvalsh(self.Wc))
        l = 1 / (1 + 4*torch.abs(min_eigenvalue - self.vars['q_init']))
        return l

    def recommend_T(self):
        """ Compute the recommended temperature

        Finds the value of T that makes the maximum stdev of the stationary
        gaussian distribution equal to the passed-in value. The precision of
        eigendirection k is (q-eig(k))/T.

        """
        max_eigvalue = torch.max(torch.linalg.eigvalsh(self.Wc))
        T = (self.settings["tgtStd"] ** 2) * \
            (self.vars['bowl_strength'] - max_eigvalue)
        return T

    # ---------------------- UPDATE WEIGHT AND BIASES --------------------
    def set_singleWeight(self, bind1, bind2, weight, symmetric=True):
        idx1 = self.find_bindings(bind1)
        idx2 = self.find_bindings(bind2)
        if symmetric:
            self.Wc[idx1, idx2] = self.Wc[idx2, idx1] = weight
        else:
            self.Wc[idx1, idx2] = weight

        # Update general Weight Matrix in the Neural Space
        self._set_weights()

    def set_bias(self, bind, bias):
        idx = self.find_bindings(bind)
        self.bC[idx] = bias

        # Update the neural Bias matrix
        self._set_bias()

    # ----------------------- CHANGE OF BASIS  ---------------------------

    def toNeural(self, matrix=None):
        """Transform Conceptual vectors into Neural vectors"""
        if matrix is None:
            matrix = self.activationC
        if len(matrix.shape) > 1:
            return torch.mm(self.TP, matrix)
        else:
            return torch.matmul(self.TP, matrix)

    def toConceptual(self, matrix=None):
        """Transform Neural vectors into local vectors."""

        if matrix is None:
            matrix = self.activation
        if len(matrix.shape) > 1:
            return torch.mm(self.TPinv, matrix)
        else:
            return torch.matmul(self.TPinv, matrix)

    # -----------------------  TRAIN ------------------------------------

    def train(self, stimuli):  # TODO:
        pass

    # -----------------------  RUNNING ROUTINE ---------------------------

    def __call__(self, stimuli, plot=False):  # TODO:
        self.readInput(stimuli)
        self.train()
        if plot:
            self.plot()

    # -----------------------  READ DATA -----------------------------------
    """Functions to retrieve biases, weights and grid points. Preliminary to
    plotting."""

    # -----------------------  VISUALIZATION -------------------------------

    def plot(self):  # TODO:
        pass

    # -----------------------  SAVE -----------------------------------------

    def logger(self, **kwargs):
        fp = self.settings['summary_file']
        with open(fp, "a+", encoding="utf-8") as summary:
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        summary.write(str(k))
                        summary.write("\n")
                        summary.write(str(v))
                        summary.write("\n" + "-"*80 + "\n\n")
                else:
                    summary.write(str(key))
                    summary.write("\n")
                    summary.write(str(value))
                    summary.write("\n" + "-"*80 + "\n\n")

    def save_net(self):
        pass

    def __repr__(self):
        return "GSC Network"


class Bowl(object):
    def __init__(self, GSCNet):
        self.Net = GSCNet
        self.center = self.Net.vars['bowl_center'] * \
            torch.ones(self.Net.nSym, dtype=torch.double)
        #self.strength = self.recommend_strength()
        self.strength = self.recommended_strength_Matlab()
        print(
            f"recommended pyton: {self.recommend_strength()}\nRecommended Matlab: {self.strength}")

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
                beta1 = -(self.Net.Bc + self.Net.externalInpC) / self.center
                beta2 = (self.Net.Bc + self.Net.externalInpC +
                         largest_eigval) / (1-self.center)
            else:
                beta1 = torch.min(
                    (self.Net.Bc + self.Net.externalInpC) / self.center) * -1
                beta2 = torch.max(
                    (self.Net.Bc + self.Net.externalInpC + largest_eigval) / (1 - self.center))
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

        q_rec = max([beta_min, beta_max, q_nd])

        return q_rec
