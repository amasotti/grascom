"""
The Gradient Symbolic Computation Network class

__date__ : Februar 201
__author__ : A. Masotti (on the model of LDNet 1.5)

"""
import scipy
from src.classes.utilFunc import column_max, fortran_reshape
from src.classes.Bowl import Bowl
import torch
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
# Set seed for reproducibility
# torch.manual_seed(123)


class Net(object):
    """The GSC Network"""

    def __init__(self, grammar, extData_path="data/inp_pandas.csv", custom_settings=None):

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
        self.stateDict = self.grammar.bind.states
        self.stateDict_rev = self.grammar.bind.state_rev

        #  Set up the settings dictionary
        self.custom_settings = custom_settings
        self._define_settings()
        # Variables that will change/be updated while training
        self._training_vars()

        # Harmony constraints
        self.Hc = self.grammar.Hc
        self.Hcc = self.grammar.Hcc

        # Preprare the network to be run
        self.stimuli = None
        self.extData_path = extData_path

        # Read external data:
        self.readInput()

        # General Setup
        self.setup_net()

    # -----------------------  GENERAL SETTINGS ------------------------------

    def _define_settings(self):
        """Set settings to default values."""

        self.settings = {}

        ##### ORIGINALLY IN THE DOMAIN FILE #######

        # Maximum input in the C-Space : no constituent can be more than 100% present
        self.settings['maxInp'] = 1

        #### ORIGINALLY IN THE SETTINGS FILE #####
        self.settings["epochs"] = 3  # Training epochs
        self.settings["tgtStd"] = 12e-6
        self.settings['T_init'] = 1e-6
        self.settings["TMin"] = 0
        self.settings["TdecayRate"] = 0.05
        self.settings["lambdaInit"] = 1
        self.settings["lambdaMin"] = 0.01
        self.settings["lambdaDecayRate"] = 0.60
        self.settings["maxSteps"] = 300000
        self.settings["emaSpeedTol"] = 0.009
        self.settings["emaFactor"] = .005
        self.settings["printInterval"] = 3000
        self.settings["summary_file"] = "data/summary.txt"
        mean = torch.ones(self.grammar.bind.nF,
                          self.grammar.bind.nR)/self.grammar.bind.nF
        self.settings["initStateMean"] = mean
        self.settings["initStateStdev"] = .015
        self.settings['clamped'] = False

        if self.custom_settings is not None:
            for key, value in self.custom_settings.items():
                if key in self.settings:
                    self.settings[key] = value

    def _training_vars(self):
        """Variables to keep track in the learning phase

        It may seem a little bit redundant but to have a "settings" and a "vars"
        dictionary has its reasons.
        The settings are reference values, which cannot be updated or modified while training.
        Vars stores the traces, constantly changing in the training loop. These are reset to the
        reference values in settings after each epoch.

        """
        self.vars = dict()
        # Temperature params
        self.vars['T_init'] = 1e-4
        self.vars['T_min'] = 0.0
        self.vars['T_decay_rate'] = 1e-2
        self.vars['lambda_min'] = 0.0001
        # Bowl params
        self.vars['q_init'] = 16  # initial strength for the bowl
        self.vars['q_max'] = 150.
        #self.vars['q_rate'] = 10.
        # Check if we can improve learning, adjusting this value
        self.vars['bowl_center'] = 0.4
        self.vars['bowl_strength'] = None
        self.vars['beta_min_offset'] = 2
        # Time step params
        self.vars['max_dt'] = 0.01
        self.vars['min_dt'] = 0.0005
        self.vars['dt'] = 0.009
        # Training traces
        self.vars['prev_s'] = None
        self.vars['Harmony_trace'] = None
        self.vars['speed_trace'] = None
        self.vars['ema_trace'] = None
        self.vars['lambda_trace'] = None
        self.vars['temp_trace'] = None
        self.vars['TP_trace'] = None
        self.vars['TPnum_trace'] = None
        self.vars['TP_h_trace'] = None
        self.vars['TP_dist_trace'] = None
        self.vars['S_trace'] = None

        if self.custom_settings is not None:
            for key, value in self.custom_settings.items():
                if key in self.vars:
                    self.vars[key] = value

    def create_full_traces(self):
        self.full_traces = dict()
        self.full_traces['Harmony_trace'] = torch.zeros(
            (self.nStimuli, self.settings['epochs'], self.settings['maxSteps']))
        self.full_traces['speed_trace'] = torch.zeros(
            (self.nStimuli, self.settings['epochs'], self.settings['maxSteps']))
        self.full_traces['ema_trace'] = torch.zeros(
            (self.nStimuli, self.settings['epochs'], self.settings['maxSteps']))
        self.full_traces['lambda_trace'] = torch.zeros(
            (self.nStimuli, self.settings['epochs'], self.settings['maxSteps']))
        self.full_traces['time_trace'] = torch.zeros(
            (self.nStimuli, self.settings['epochs'], self.settings['maxSteps']))
        self.full_traces['TP_trace'] = dict()
        self.full_traces['TPnum_trace'] = torch.empty(
            (self.nStimuli, self.settings['epochs'], self.settings['maxSteps']))
        self.full_traces['TP_h_trace'] = torch.zeros(
            (self.nStimuli, self.settings['epochs'], self.settings['maxSteps']))
        self.full_traces['TP_dist_trace'] = torch.zeros(
            (self.nStimuli, self.settings['epochs'], self.settings['maxSteps']))
        self.full_traces['S_trace'] = torch.zeros(
            (self.nStimuli, self.settings['epochs'], self.settings['maxSteps'], self.nSym, 1))
        self.full_traces['reaction_times'] = torch.zeros(
            (self.nStimuli, self.settings['epochs']))
        self.full_traces['divergence'] = torch.zeros(
            (self.nStimuli, self.settings['epochs']))
        self.full_traces['winners_idx'] = torch.empty(
            (self.nStimuli, self.settings['epochs'], 1, self.grammar.nR))

    # -----------------------  RETRIEVE BINDINGS -------------------------
    def find_bindings(self, bindName):
        """Retrieve the index of a specific binding"""
        try:
            return self.bind2index[bindName]
        except:
            raise KeyError(
                f"The binding {bindName} is not in the general list... check your input file!")

    def find_role(self, roleName):
        """Retrieve the index of a specific role"""
        try:
            return self.role2index[roleName]
        except:
            raise KeyError(
                f"The role {roleName} is not in the general list... check your input file!")

    def find_filler(self, fillerName):
        """Retrieve the index of a specific binding"""
        try:
            return self.filler2index[fillerName]
        except:
            raise KeyError(
                f"The Filler {fillerName} is not in the general list... check your input file!")

    # ----------------------- SETUP ROUTINE ------------------------------------

    def setup_net(self):
        # Create the change-of-basis matrix
        # This will allow us to change from neural distributed to local symbolic repr.
        self.generateTP()

        # Initialize weights and biases
        self.W = self.compute_neural_weights()
        self.B = self.compute_neural_biases()

        # Construct weights and biase matrices for the Conceptual space
        self._set_weights()
        self._set_bias()
        # self.debug_copies()

        # States and Inputs for the RNN-like network
        self.stateC = torch.zeros(self.nSym, dtype=torch.double)
        self.stateC_prev = torch.zeros(self.nSym, dtype=torch.double)
        self.state = self.toNeural()
        self.state_prev = self.toNeural(matrix=self.stateC_prev)

        # Initialize vectors for the external input
        self.inpC = torch.zeros(self.nSym, dtype=torch.double)
        self.inpC_prev = torch.zeros(self.nSym, dtype=torch.double)
        self.inpS = self.toNeural(self.inpC)
        self.inpS_prev = self.toNeural(self.inpC_prev)

        # Bowl
        # Calculate the weights and biases for the Bowl:
        self._bowl_params()
        self.Bowl_bC, self.Bowl_bS = self.bowl.set_biases()
        self.Bowl_WC, self.Bowl_WS = self.bowl.set_weights()

        # Calculate the recommended Lambda and Temp values:
        self.check_Q_T_lambda()

        # Add the bowl weights to the Harmoniy weights
        self.merge_bowl()

        # Initialize states
        self.initialize_state()
        self.create_result_states()

        # Log default parameters
        self.logger(default_settings="#"*25 + " DEFAULT SETTINGS" + "#"*25)
        self.logger(default_settings=self.settings)
        self.logger(bowl="#"*25 + " BOWL PARAMETER" + "#"*25)
        self.logger(bowl=self.bowl.__dict__)

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
        TP = np.kron(self.R.numpy(), self.F.numpy())
        self.TP = torch.tensor(TP, dtype=torch.double)
        # create the inverse if TP is a square matrix:
        # Use the Moore-Penrose pseudoinverse if TP is not square
        self.TPinv = torch.inverse(self.TP)

    def compute_neural_biases(self, dist="zero"):
        """Compute the Biases Matrix for the Neural space.

            This will be derived from the Harmonies specified in self.Hc
            (the single constituent Harmony) and will then be transformed
            into the B matrix for the Constituent space through simple
            matrix multiplication

        """
        # flatten
        harmonies = fortran_reshape(self.Hc, (torch.numel(self.Hc), 1))
        # harmonies = self.Hc.reshape((torch.numel(self.Hc), 1))

        # Initialize
        biases = self.initializer((self.nSym, 1), dist=dist)

        # Update
        for i in range(self.nSym):
            b_i = self.TP[:, i].unsqueeze(1)
            update_value = (harmonies[i] * b_i) / torch.matmul(b_i.T, b_i)
            biases += update_value.reshape((self.nSym, 1))
        return biases

    def compute_neural_weights(self, dist="zero"):
        """Compute the W Matrix for the Neural space.

            This will be derived from the Harmonies specified in self.Hcc
            (the pairwise constituent Harmony) and will then be transformed
            into the W matrix for the Constituent space through simple
            matrix multiplication

            # 0.5 in the case i == j (because of the symmetry)
        """
        # TODO: Should this begin with zeros? Or can we initialize the matrices with random dist?
        harmonies = fortran_reshape(self.Hcc, (self.nSym, self.nSym))
        W = self.initializer((self.nSym, self.nSym), dist=dist)

        # Update using the Hcc infos:
        for i in range(self.nSym):
            w_i = self.TP[:, i].unsqueeze(1)  # take the i-th binding
            for j in range(i+1):  # just operate in the lower triangle, the rest is symmetric
                w_j = self.TP[:, j].unsqueeze(1)  # take the j-th binding
                if i != j:
                    k = 1
                else:
                    k = 0.5
                W += k * (harmonies[i, j] * (w_i @ w_j.T + w_j @
                                             w_i.T)) / ((w_i.T @ w_i) @ (w_j.T @ w_j))
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

    def _bowl_params(self):
        """Set up the bowl parameters.
            center : passed by the user as hyperparameter.
            strength: a very important value, that constraints the quantization.
            zeta : the neural version of the center of the bowl

        """
        self.bowl = Bowl(self)
        self.vars['bowl_strength'] = self.vars['q_init'] = self.bowl.strength + \
            self.vars['beta_min_offset']
        if self.vars['bowl_strength'] <= self.vars['beta_min_offset']:
            print(
                f"Bowl overflow -- Set to the minimum value :  {self.vars['beta_min_offset']}")
            # aise ValueError("Bowl overflow... strength lower than set tolerance. Modify the tolerance or fix the bug!")
            self.vars['bowl_strength'] = self.vars['beta_min_offset']
        if self.vars['bowl_strength'] > self.vars['q_max']:
            self.vars['bowl_strength'] = self.vars['q_max']

        """
        if self.vars['q_init'] != self.bowl.strength + self.vars['beta_min_offset']:
            print(f"Recommended bowl strength : {self.bowl.strength} \t Actual value: {self.vars['q_init']}")
            choice = input("Do you want to set the value to the recommended one: (y/n): ")
            if choice.lower() == "y":
                self.vars['bowl_strength'] = self.vars['q_init'] = self.bowl.strength + self.vars['beta_min_offset']
            else:
                self.vars['bowl_strength'] = self.vars['q_init']
        else:
            self.vars['bowl_strength'] = self.vars['q_init']
        """
        self.vars['zeta_bowl'] = self.toNeural(self.bowl.center)
        print(f"Value for Q set to {self.vars['bowl_strength']}")

    def merge_bowl(self):
        """Add bowl biases and weights to the Harmony biases and weights."""
        self.B += self.Bowl_bS
        self.Bc += self.Bowl_bC
        self.W += self.Bowl_WS
        self.Wc += self.Bowl_WS

    def debug_copies(self):
        """Create debug copies of the initialized matrices
        # TODO: Why? Can we delete this?
        """
        self.b_debug = self.B.clone().detach()
        self.Bc_debug = self.Bc.clone().detach()
        self.W_debug = self.W.clone().detach()
        self.Wc_debug = self.Wc.clone().detach()

    # ------------------ LOTKA VOLTERRA MATRICES ------------------------------------
    # Probably it'd be better to create a separate class for the LV Dynamics #TODO:

    def LV_Matrices(self):
        """Initialize the Lotka-Volterra Dynamics
            and its weights
        """
        self.LV_inhM = self.LotkaVolterra_InhibitMatrix()  # (nF, nF)
        self.LV_c, self.LV_s = self.LotkaVolterra_Dynamics()  # (nS, 1)
        # TODO: The LV weights are incredibly slow...
        # self.LV_W = self.LotkaVolterra_Weights2()

    def LotkaVolterra_InhibitMatrix(self):
        """Create the Lotka Volterra Matrix in the C-Space

        This is an inhibitory matrix with 0 on the main diagonal and
        -2 everywhere else.

        """
        LV = -2 * (torch.ones(len(self.filler2index),
                              len(self.filler2index)) - torch.eye(len(self.filler2index)))
        LV = LV.double()
        return LV

    def LotkaVolterra_Dynamics(self):
        """Lotka Volterra matrices in the C- and S-space

        """
        LV_c = self.toConceptual(self.state)  # (nF, nR)
        LV_c = LV_c.mul(1 - LV_c + self.LV_inhM.matmul(LV_c))
        LV_s = self.toNeural(LV_c)

        return LV_c, LV_s

    def LotkaVolterra_Weights(self):
        """Lotka-Volterra Weights.

        Params:
        nS = the dimension of the s-Space (nRoles * nFillers)
        nF = the number of Fillers
        TP = the Tensor Product Matrix
        TPinv = the inverse matrix of TP

        Return:
        ------------------
        W : the matrix of Weights for the Lotka-Volterra space

        # ACHTUNG! : The following function is an O(nS^5) algorithm!!!
        There is probably a more efficient way to implement this!
        """

        # Initialize the 3D Array
        W = torch.zeros((self.nSym, self.nSym, self.nSym)).double()
        for k in tqdm(range(self.nSym), desc="Calculate LV Weights", total=self.nSym):
            for p in range(self.nSym):
                for n in range(self.nSym):
                    for i in range(self.nSym):
                        check_i = np.floor(i/len(self.filler2index))
                        for j in range(self.nSym):
                            check_j = np.floor(j/len(self.filler2index))
                            if check_i == check_j:
                                delta = 1 if i == j else 0
                                W[n, p, k] += self.TP[i, k] * self.TPinv[p,
                                                                         i] * self.TPinv[n, j] * (delta-2)
        return W

    def LotkaVolterra_Weights2(self):
        """Lotka-Volterra Weights.

        Params:
        nS = the dimension of the s-Space (nRoles * nFillers)
        nF = the number of Fillers
        TP = the Tensor Product Matrix
        TPinv = the inverse matrix of TP

        Return:
        ------------------
        W : the matrix of Weights for the Lotka-Volterra space

        # FIXME: Improved by delimiting the range where we now the floor division for i and j
        will be equal. Still very slow.

        # TODO: Check that the dimensions are right!
        """

        # Initialize the 3D Array
        W = torch.zeros((self.nSym, self.nSym, self.nSym)).double()
        for k in tqdm(range(self.nSym), desc="Calculate LV Weights", total=self.nSym):
            for p in range(self.nSym):
                for n in range(self.nSym):
                    for r in range(0, self.grammar.nR):
                        rng = np.array([0, 7]) + (r * self.grammar.nF)
                        for i in range(rng[0], rng[1]):
                            for j in range(rng[0], rng[1]):
                                delta = 1 if i == j else 0
                                W[n, p, k] += self.TP[i, k] * \
                                    self.TPinv[p, i] * \
                                    self.TPinv[n, j] * (delta-2)
        return W

    # ------------------RECOMMENDED VALUES FOR Q, L, T ------------------------------------

    def check_Q_T_lambda(self):
        """Check the bowl parameters."""
        # Substituted with Bowl.recommend_strength()
        # self.vars['q_rec'], self.vars['q_rec_nd'] = self.recommend_Q()

        self.vars['lambda_rec'] = self.recommend_L()
        # Check lambdas
        # 1e-2 tolerance #TODO: review tol
        if abs(self.settings["lambdaInit"] - self.vars['lambda_rec']) > 1e-3:
            print(
                f"LAMBDA RECOMMENDED: {self.vars['lambda_rec']}, ACTUAL LAMBDA = {self.settings['lambdaInit']}")
            choice = input(
                "If you want to change to the recommended Lambda value press 'y', else any other key:")
            if choice.lower() == 'y':
                self.vars['lambdaInit'] = self.vars['lambda_rec']
            else:
                self.vars['lambdaInit'] = self.settings["lambdaInit"]
            self.logger(lambda_value=self.vars['lambdaInit'])
        else:
            self.vars['lambdaInit'] = self.settings["lambdaInit"]

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
        else:
            self.vars['T_init'] = self.settings['T_init']

    def recommend_L(self):
        """Recommended lambda init value

        This function is mathematically problematic. See Issue #2
        # TODO: The eigvectors and eigvals differ between MATLAB, Numpy and Pytorch probably due to different
        algorithms viz. error in the numerical algorithm. I should check if this is an issue or it works.
        """

        min_eigenvalue = torch.min(torch.linalg.eigvalsh(self.Wc))
        lamBda = 1 / (1 + 4*torch.abs(min_eigenvalue - self.vars['q_init']))
        return lamBda

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
    # ---------------------- STATES AND INPUTS --------------------

    def initialize_state(self):
        """Initialize network states and load external inputs"""
        self.state = self.initializer((self.nSym, 1))
        self.inpS = self.initializer((self.nSym, 1))
        self.inpC = self.initializer((self.nSym, 1))

        # Create full traces
        self.create_full_traces()

        # Initialize Lotka Volterra
        self.LV_Matrices()

        # Initialize Temperature and Lambda
        self.vars['T'] = 0
        self.vars['lambda'] = 0

    def create_result_states(self):
        """Create a kind of dataframe to store results."""

        # Dictionary of Final TP States (== the winners)
        self.final_TPStates = dict()
        for stimulus in self.inputNames:
            for i in range(self.settings['epochs']):
                key = stimulus + "/" + str(i)
                self.final_TPStates[key] = 0

    # ----------------------- EXTERNAL INPUT  ----------------------------

    def readInput(self):
        """Read external input.

        Inputs are provided in a csv table with header:
        id,filler,r1,r2,....,rn

        The id marks the input as whole: many lines with the same ID count as 1 input

        requires Pandas
        """
        fp = self.extData_path
        # Read dataframe
        inputs = pd.read_csv(fp, sep=",")
        self.inputNames = []
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
                fidx = self.filler2index[filler[0]]
                inp_string += filler[0] + "-"
                for roledix in range(self.grammar.bind.nR):
                    self.stimuli[idx, fidx, roledix] = filler[roledix+1]
            print(f"Input processed: {inp_string[:-1]}\n")
            # Store the names for later plotting
            self.inputNames.append(inp_string[:-1])

            # ----------------------- CHANGE OF BASIS  ---------------------------

    def toNeural(self, matrix=None):
        """Transform Conceptual vectors into Neural vectors
        The dimension should be (nSym, 1)
        """
        if matrix is None:
            matrix = self.stateC

        matrix = fortran_reshape(matrix, (matrix.numel(), 1))
        return torch.matmul(self.TP, matrix)

    def toConceptual(self, matrix=None):
        """Transform Neural vectors into local vectors."""

        if matrix is None:
            matrix = self.state
        if len(matrix.shape) > 1:
            C = torch.mm(self.TPinv, matrix)
        else:
            C = torch.matmul(self.TPinv, matrix)
        C = fortran_reshape(C, (self.grammar.nF, self.grammar.nR))
        return C

    # -----------------------  TRAIN ------------------------------------

    def __call__(self, plot=False):
        """Train the network, plot the results.

        When the Net is constructed all the weights, biases, bowl params etc..
        are initialized. The external inputs are read and represented as tensors.

        Then we can simply call the Network and we will start the training loop


        """
        # save copy of the initialized matrices
        scipy.io.savemat("data/initialized_mats.mat", mdict={"B": self.B.numpy(), "Bc": self.Bc.numpy(),
                                                             "TP": self.TP.numpy(), "TPinv": self.TPinv.numpy(),
                                                             "W": self.W.numpy(), "Wc": self.Wc.numpy(),
                                                             "Hcc": self.Hcc.numpy(), "Hc": self.Hc.numpy(),
                                                             "stimuli": self.stimuli.numpy(),
                                                             "bowl_center": self.vars['bowl_center'],
                                                             "bowl_strength": self.vars['bowl_strength'],
                                                             "LVinh": self.LV_inhM.numpy(), "LVc": self.LV_c.numpy(),
                                                             "LVs": self.LV_s.numpy()
                                                             })
        for epoch in trange(self.settings['epochs'], desc="Epoch routine:"):
            for stimulus in trange(self.nStimuli, desc=f"Stimulus routine:"):
                stim_vec = self.stimuli[stimulus, :, :]
                diverge, harmony = self.process_stimulus(
                    stim_vec, epoch, stimulus)
                # Update after stimulus processing
                self.update_after_stim(
                    nStimulus=stimulus, epoch=epoch, diverge=diverge)
                print(
                    f"\nLast best Harmony: {float(harmony)}\n")
                self.logger(title="#"*50 + " Training routine " + "#"*50)
                self.logger(epoch=epoch, stimulus=stimulus)
                self.logger(default_settings=self.vars)
            # Update values after each epoch
        self.final_update()
        self.save_final_traces()

    # -----------------------  PROCESSING AND UPDATE ---------------------------------
    def process_stimulus(self, stimulus, epoch, stimNum):
        """Process a single stimulus."""
        diverge = False
        self.init_for_run(stimulus)
        harmony, state, stateC = self.calc_max_Harmony()

        # Update bowl parameters (They depend on the stimulus)
        # self._bowl_params()

        # Update after each step
        self.updateAfterStep(harmony)
        self.consoleLog_step(epoch=epoch+1, stimNum=stimNum+1)

        # Loop over steps ~30 000 steps
        for step in range(1, self.settings['maxSteps']):
            self.vars['step'] = step

            # Compute step components
            self.Hg = self.calc_harmony_dynamics()  # Harmony gradient
            self.LV_c, self.LV_s = self.LotkaVolterra_Dynamics()  # Lotka gradient
            self.add_noise()  # add noise

            # Copy the actual tensor to prev_S (RNN)
            self.stateC = self.toConceptual(self.state)
            self.state_prev = self.state.clone().detach()
            self.stateC_prev = self.stateC.clone().detach()

            # Now update the actual state
            self.state += self.vars['dt'] * (self.vars['lambda'] * self.Hg + (
                1-self.vars['lambda']) * self.LV_s) + self.noise
            self.stateC = self.toConceptual(self.state)

            if self.check_overflow():
                diverge = True
                print(f"Net overflowed at step: {step}")
                break

            harmony = self.calc_harmony()
            self.updateAfterStep(harmony=harmony)
            # Update Animation
            # Print information if step % print_every == 0
            self.consoleLog_step(epoch=epoch+1, stimNum=stimNum+1)

            # Update Lambda and Temperature!
            self.update_lambda_T()

            if self.check_convergency():
                print(
                    f"The net has converged at step {self.vars['step']}, nearest State: {self.vars['TP_trace']}")
                break

        if diverge:
            # Don't use this step
            self.vars['step'] -= 1
        if self.vars['step'] == self.settings['maxSteps']:
            print("FINAL STEP")
            print(
                f"The net has reached the maximum number of steps {self.settings['maxSteps']}")
            self.consoleLog_step()

        return diverge, harmony

    def updateAfterStep(self, harmony):
        """Record state after each step.

        This updates the traces initialized in init_for_run()

        """
        TP_state, winners, state_name, binding, Cdist, Sdist, state_num, TP_h = self.calc_nearest_state()
        self.vars['winners'] = winners
        self.vars['symBindings_winners'] = binding
        self.vars['S_trace'][self.vars['step'], :, :] = self.state
        self.vars['Harmony_trace'][self.vars['step']] = harmony
        self.vars['speed_trace'][self.vars['step']] = self.calc_speed()
        self.vars['ema_trace'][self.vars['step']] = self.calc_ema()
        self.vars['lambda_trace'][self.vars['step']] = self.vars['lambda']
        self.vars['temp_trace'][self.vars['step']] = self.vars['T']
        self.vars['TP_trace'].append(state_name)
        self.vars['TPnum_trace'][self.vars['step']] = state_num
        self.vars['TP_h_trace'][self.vars['step']] = TP_h
        self.vars['TP_dist_trace'][self.vars['step']] = Cdist

    def update_after_stim(self, nStimulus, epoch, diverge):
        """Update values and traces after each stimulus"""
        H, s, C = self.calc_max_Harmony()

        # Update traces
        self.vars['maxHarmony'][:, :, nStimulus, epoch] = C

        # Update finalStates traces
        inp = self.inputNames[nStimulus] + "/" + str(epoch)
        self.final_TPStates[inp] = self.vars['TP_trace'][-1]
        self.update_full_traces(nStimulus=nStimulus,
                                epoch=epoch, diverge=diverge)

        self.logger(fp="data/stimuli_summary.txt",
                    stimulus_epoch=str(nStimulus)+"_"+str(epoch))
        self.logger(fp="data/stimuli_summary.txt",
                    winners=self.final_TPStates)

    def update_full_traces(self, nStimulus, epoch, diverge):
        """Update full traces tensors"""
        self.full_traces['Harmony_trace'][nStimulus,
                                          epoch, :] = self.vars['Harmony_trace']
        #self.full_traces['speed_trace'][nStimulus, epoch, : ] = self.vars['speed_trace']
        #self.full_traces['ema_trace'][nStimulus, epoch, : ] = self.vars['ema_trace']

        self.full_traces['lambda_trace'][nStimulus,
                                         epoch, :] = self.vars['lambda_trace']
        self.full_traces['time_trace'][nStimulus,
                                       epoch, :] = self.vars['temp_trace']
        self.full_traces['TP_h_trace'][nStimulus,
                                       epoch, :] = self.vars['TP_h_trace']
        self.full_traces['TP_dist_trace'][nStimulus,
                                          epoch, :] = self.vars['TP_dist_trace']
        self.full_traces['S_trace'][nStimulus,
                                    epoch, :, :, :] = self.vars['S_trace']
        self.full_traces['reaction_times'][nStimulus,
                                           epoch] = self.vars['step']
        self.full_traces['divergence'][nStimulus, epoch] = diverge
        self.full_traces['winners_idx'][nStimulus,
                                        epoch, :, :] = self.vars['winners']

        key = self.inputNames[nStimulus] + "/rep_" + str(epoch)
        self.full_traces['TP_trace'][key] = self.vars['TP_trace']
        # TODO: change this!
        self.full_traces['TPnum_trace'][nStimulus,
                                        epoch, :] = self.vars['TPnum_trace']

    def update_after_epoch(self, winnerlist):
        # TODO: Do some update after each epoch
        pass

    def final_update(self):
        "Final update after all epochs"
        pass

    # ----------------------- AUXILIARY TO PROCESSING ----------------------

    def init_for_run(self, stimulus):
        """Prepare the network to process an external stimulus.

        We start with a local representation in the C-space and
        initialize the net.

        Most importantly this function updates the representations
        of the external inputs

        """
        # Initialize training traces
        max_steps = self.settings['maxSteps']
        self.vars['S_trace'] = torch.zeros((max_steps, self.nSym, 1))
        self.vars['prev_s'] = torch.tensor(
            [float('inf')]) * torch.ones((self.nSym, 1))
        self.vars['Harmony_trace'] = torch.zeros(max_steps)
        self.vars['speed_trace'] = torch.zeros(max_steps)
        self.vars['ema_trace'] = torch.zeros(max_steps)
        self.vars['lambda_trace'] = torch.zeros(max_steps)
        self.vars['temp_trace'] = torch.zeros(max_steps)
        self.vars['TP_trace'] = list()
        self.vars['TPnum_trace'] = torch.zeros(max_steps)
        self.vars['TP_h_trace'] = torch.zeros(max_steps)
        self.vars['TP_dist_trace'] = torch.zeros(max_steps)
        self.vars['maxHarmony'] = torch.empty(
            self.grammar.nF, self.grammar.nR, self.nStimuli, self.settings['epochs'])  # check this size

        # Create the representations for the stimulus
        stimulus = fortran_reshape(
            stimulus, (torch.numel(stimulus), 1)).double()
        self.inpS = self.TP.matmul(stimulus)
        self.inpC = self.TP.T.matmul(self.inpS)

        # ----------- ALTERNATIVE ---------------------
        # Let's try the matrix version : TPR = Fillers * Bind * Roles
        #self.inpS = self.F.matmul(stimulus).matmul(self.R)
        #self.inpC = self.inpS @ torch.pinverse(self.R)
        # Reshape to make it suitable for MM.
        #self.inpC = self.inpC.reshape((self.inpC.numel(), 1))
        #self.inpS = self.inpS.reshape((self.inpS.numel(), 1))
        # ----------- END ALTERNATIVE ---------------------

        self.inpS = self.inpS.double()
        self.inpC = self.inpC.double()

        # Initialize state
        self.initial_state = self.settings['initStateMean'] * torch.rand(
            (self.grammar.nF, self.grammar.nR)) * self.settings["initStateStdev"]
        self.initial_state = self.initial_state.double()
        self.state = self.toNeural(self.initial_state)
        self.stateC = self.toConceptual(self.state)

        # Set Lambda and Temperature
        self.vars['T'] = self.vars['T_init']
        self.vars['lambda'] = self.vars['lambdaInit']
        self.vars['step'] = 0

    def calc_max_Harmony(self):
        """Calculate the maximum Harmony state for the actual state.

        Returns:
        ---------
            - A conceptual representation of the actual state
            - A neural representation of the actual state
            - the maximum Harmony of the actual state

        """
        inpS = self.toNeural(self.inpC)
        state = torch.linalg.pinv(self.W).matmul(-self.B - inpS)
        stateC = self.toConceptual(state)
        harmony = self.calc_harmony(state=state)
        return harmony, state, stateC

    def calc_harmony(self, state=None):
        """Calculate Harmony value"""

        if state is None:
            state = self.state

        harmony = (self.B + self.inpS).T.matmul(state)
        harmony += .5 * state.T.matmul(self.W).matmul(state)
        return harmony

    def calc_harmony_dynamics(self):
        """Calculate the Harmony Gradient

        HarmonyG = bias + input + W * s 
        """
        Hg = self.B + self.inpS + self.W.matmul(self.state)
        return Hg

    def calc_speed(self):
        """Calc the speed at which the Learning is improving"""
        if self.vars['step'] > 0:
            target_tensor = torch.abs(self.state - self.state_prev)
            speed = column_max(target_tensor) / self.vars['dt']
        else:  # this is the first step, no calculation is possible
            speed = float('NaN')
        return speed

    def calc_ema(self):
        """Calc the ema value"""
        emaFactor = self.settings['emaFactor']
        stepFactor = emaFactor ** self.vars['dt']
        if self.vars['step'] == 0:
            ema = float('NaN')
        elif self.vars['step'] == 1:
            ema = self.vars['speed_trace'][1]
        else:
            ema = stepFactor * self.vars['ema_trace'][self.vars['step'] - 1] + (
                1 - stepFactor)*self.vars['speed_trace'][self.vars['step']]
        return ema

    def calc_nearest_state(self):
        """Calc the nearest TP state and the most probable winner"""
        self.stateC = self.toConceptual(self.state)
        CTP, winners = self.find_winner()

        state_name = self.find_TPname(filleridx=winners)
        binding = self.find_symBinding(filleridx=winners)
        state_num = self.find_TPnum(stateName=state_name)
        TP_state = self.TP.matmul(fortran_reshape(CTP, (torch.numel(CTP), 1)))
        Cdist = torch.norm(CTP - self.stateC, p='fro')  # Frobenius Norm
        Sdist = self.L2norm(TP_state - self.state)
        TP_h = self.calc_harmony(state=TP_state)

        return TP_state, winners, state_name, binding, Cdist, Sdist, state_num, TP_h

    def find_winner(self):
        """Return a binary Matrix that implements the winner-takes-all strategy
        and the indices of the winning fillers.

        """

        # Extract the max value of each col in the conceptual state matrix
        winners_idx = column_max(self.stateC, what='argmax')

        M = torch.zeros_like(self.stateC)
        # Populate the matrix
        for r in range(M.shape[1]):
            M[winners_idx[r], r] = 1
        return M, winners_idx

    def find_TPname(self, filleridx):
        """
        Find the state Number and name for a given list of
        winning fillers 
        """
        stateName = ""
        for filler in filleridx.long():
            fillerName = self.index2filler[int(filler)]
            if stateName == "":
                stateName += fillerName
            else:
                stateName += "-" + fillerName
        return stateName

    def find_TPnum(self, stateName):
        """Find the number corresponding to
            the generated stateName
        """
        try:
            stateNum = self.stateDict[stateName]
        except:
            raise KeyError("find_TPnum: StateName not found!!")
        return stateNum

    def find_symBinding(self, filleridx):
        """Find the bindings.
        Given an ordered list of fillers (the position == roles)
        returns the corresponding bindings

        Returns a dictionary:
            key : binding in the form filler/role 
            value : the binding index in the general binding dictionary
        """
        symBinding = dict()
        for i, filler in enumerate(filleridx):
            binding = self.index2filler[int(filler)] + "/" + self.index2role[i]
            symBinding[binding] = self.find_bindings(binding)
        return symBinding

    def L2norm(self, array):
        """Calculate the L2 Norm"""
        norm = torch.sqrt(torch.sum(array * array))
        return norm

    def add_noise(self):
        """Add noise to the system according
            to the T parameter"""
        self.noise = torch.rand(self.state.shape).double()
        self.noise *= torch.sqrt(2 *
                                 self.vars['T']*torch.tensor(self.vars['dt']))

    def check_overflow(self):
        """Check if the training converged or diverged"""
        self.stateC = self.toConceptual(self.state)

        check_inf = torch.any(torch.isinf(self.stateC)) or torch.any(
            torch.isinf(self.state))
        check_nan = torch.any(torch.isnan(self.stateC)) or torch.any(
            torch.isnan(self.state))

        if check_inf or check_nan:
            return True
        else:
            return False

    def check_convergency(self):
        """Check if the Net has converged"""
        if self.vars['ema_trace'][self.vars['step']] <= self.settings["emaSpeedTol"]:
            return True
        else:
            return False

    def update_lambda_T(self):
        """ After each step check the values of Lambda and T
        and update them"""

        # Set rate if we're at the beginning
        if self.vars['step'] <= 1:
            self.vars['lambdaDecayRate'] = self.settings['lambdaDecayRate']

        incremental_lambda = 1 - \
            (1-self.vars['lambdaDecayRate'])**self.vars['dt']
        incremental_temp = 1-(1-self.vars['T_decay_rate'])**self.vars['dt']

        # Update lambda
        self.vars['lambda'] -= incremental_lambda*(
            self.vars['lambda']-self.settings["lambdaMin"])

        # Update T
        self.vars['T'] -= incremental_temp * \
            (self.vars['T']-self.settings["TMin"])

    # -----------------------  VISUALIZATION -------------------------------

    def plot(self):  # TODO:
        pass

    # -----------------------  SAVE -----------------------------------------

    def logger(self, fp=None, **kwargs):
        if fp is None:
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

    def consoleLog_step(self, **kwargs):
        """Print infos on the console at each n step"""
        n = self.settings["printInterval"]
        if self.vars['step'] == 0 or self.vars['step'] % n == 0:
            print("\n" + "-"*80 + "\n")
            print(
                f"STEP : {self.vars['step']}\t Lambda : {self.vars['lambda']}\t Harmony: {self.vars['Harmony_trace'][self.vars['step']]}\n")
            if kwargs is not None:
                for k, v in kwargs.items():
                    print(f"{k} : {v}")
            C = self.toConceptual(self.state)
            C_as_df = self.matrix_to_df(C)
            print(f"\nConceptual Matrix:")
            print(C_as_df)
            print(f"Nearest TP: {self.vars['TP_trace'][-1]}")

    def matrix_to_df(self, matrix):
        """Transform C-matrix to pandas df for printing"""
        # Convert to np array
        matrix = matrix.numpy()
        df = pd.DataFrame(matrix, columns=list(
            self.role2index.keys()), index=list(self.filler2index))
        return df

    def save_final_traces(self):
        """Save the matrices produced as traces to disk
            both as .mat (MATLAB) and .pt (PyTorch) format
        """
        torch.save(self.full_traces, "data/full_traces.pt")
        torch.save({"W": self.W, "Wc": self.Wc,
                    "B": self.B, "Bc": self.Bc, "TP": self.TP}, "data/params.pt")
        scipy.io.savemat("data/full_traces.mat", mdict=self.full_traces)
        scipy.io.savemat("data/params.mat", mdict={"W": self.W.numpy(), "Wc": self.Wc.numpy(),
                                                   "B": self.B.numpy(), "Bc": self.Bc.numpy(), "TP": self.TP.numpy()})

    def __repr__(self):
        return "GSC Network"
