"""
The Gradient Symbolic Computation Network class

"""
from src.classes.utilFunc import fortran_reshape
import torch
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
        # Create all possible bindings
        self.set_all_bindings()

        #  Set up the settings dictionary
        self._define_settings()
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
        self.vars['q_init'] = 0.
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
        inputs = pd.read_csv(fp, sep=",")
        self.nStimuli = len(inputs['id'].unique())
        self.stimuli = torch.zeros(
            (self.nStimuli, self.grammar.bind.nF, self.grammar.bind.nR))
        for idx, i in enumerate(inputs['id'].unique()):
            inp_string = ""
            stimulus = inputs[inputs.id == i].to_numpy()[:, 1:]
            for filler in stimulus:
                fidx = self.grammar.bind.fillers.index(filler[0])
                inp_string += filler[0] + "-"
                for roledix in range(self.grammar.bind.nR):
                    self.stimuli[idx, fidx, roledix] = filler[roledix+1]
            print(f"Input processed: {inp_string[:-1]}\n")
        # TODO: Should this be left so, or do we need (nInp, nR, nF)?

    # -----------------------  RETRIEVE BINDINGS -------------------------
    def set_all_bindings(self, sep="/"):
        """Build a list of all possible bindings"""
        self.binding_names = [
            filler + sep + role for role in self.grammar.roles.rolesNames for filler in self.grammar.fillers.fillersNames]

    def find_bindings(self, binding):
        """Retrieve the index of a specific binding"""
        try:
            return self.binding_names.index(binding)
        except:
            raise f"The binding {binding} is not in the general list... check your input file!"

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
        self.encodings['fillers_names'] = self.grammar.fillers.fillersNames
        self.encodings['roles_names'] = self.grammar.roles.rolesNames
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
        # TODO: Change below, the set_weights and set_biases. These should now give the conceptual
        # matrices
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

        # Bowl
        self.bowl = Bowl(self)
        self.vars['bowl_center'] = self.bowl.center
        self.vars['bowl_strength'] = self.bowl.strength + \
            self.vars['beta_min_offset']
        self.vars['zeta_bowl'] = self.toNeural(self.vars['bowl_center'])

    def generateTP(self):
        """Generate the Matrices for the change-of-basis from Neural to Conceptual
            and the other way round 
        """
        # Approximate to Kronecker Product
        # TODO: Check that the dimension are right!
        self.TP = torch.kron(self.R, self.F).double()
        # create the inverse if TP is a square matrix:
        # Use the Moore-Penrose pseudoinverse if TP is not square
        self.TPinv = torch.linalg.pinv(self.TP, hermitian=True)
        self.Gc = torch.mm(self.TPinv.T, self.TP)

    def compute_neural_biases(self):
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
        biases = torch.zeros((self.nSym, 1), dtype=torch.double)

        # Update
        for i in range(self.nSym):
            b_i = self.TP[:, i]
            update_value = (harmonies[i] * b_i) / torch.matmul(b_i.T, b_i)
            biases += update_value.reshape((self.nSym, 1))
        return biases

    def compute_neural_weights(self):
        """Compute the W Matrix for the Neural space.

            This will be derived from the Harmonies specified in self.Hcc
            (the pairwise constituent Harmony) and will then be transformed
            into the W matrix for the Constituent space through simple 
            matrix multiplication

            #QUESTION: Why that 0.5 if i != j ? 
        """
        harmonies = fortran_reshape(self.Hcc, (self.nSym, self.nSym))
        W = torch.zeros((self.nSym, self.nSym), dtype=torch.double)

        # Update using the Hcc infos:
        for i in range(self.nSym):
            w_i = self.TP[:, i]  # take the i-th filler
            for j in range(i+1):  # just operate in the lower triangle, the rest is symmetric
                w_j = self.TP[:, j]
                if i == j:
                    W = W + harmonies[i, j] * (torch.matmul(w_i, w_j.T) + torch.matmul(
                        w_j, w_i.T)) / (torch.matmul(w_i.T, w_j)*torch.matmul(w_j.T, w_i))
                else:
                    W = W + .5 * harmonies[i, j] * (torch.matmul(w_i, w_j.T) + torch.matmul(
                        w_j, w_i.T)) / (torch.matmul(w_i.T, w_j)*torch.matmul(w_j.T, w_i))
        return W

    def _set_weights(self):
        self.Wc = torch.mm(self.TP.T, self.W).mm(self.TP)

    def _set_bias(self):
        """Neural Bias Matrix from C-biases

            Bc is a n-dimensional array, n = num Bindings
            TP is the change of Base matrix

        """
        # Alternative
        # self.B = self.TPinv.T.matmul(self.Bc)
        self.Bc = self.TP.T.matmul(self.B)

    def _set_quantList(self):
        """Quantization list"""
        self.quantList = []
        for i, r in enumerate(self.grammar.roles.rolesNames):
            self.quantList.append(self.R[:, i])

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

    def logger(fp="data/summary.txt", **kwargs):
        with open(fp, "a+", encoding="utf-8") as summary:
            for key, value in kwargs.items():
                summary.write(key)
                summary.write("\n")
                summary.write(value)
                summary.write("-"*80 + "\n\n")

    def save_net(self):
        pass


class Bowl(object):
    def __init__(self, GSCNet):
        self.Net = GSCNet
        self.center = self.Net.vars['bowl_center'] * \
            torch.ones(self.Net.nSym, dtype=torch.double)
        self.strength = self.recommend_strength()

    def recommend_strength(self):
        """Calculate the recommended strength for the Bowl"""
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
