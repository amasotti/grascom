"""
The Gradient Symbolic Computation Network class

__date__ : Februar 201
__author__ : A. Masotti (on the model of LDNet 1.5)

"""
from src.classes.utilFunc import fortran_reshape
from src.classes.Bowl import Bowl
import torch
import numpy as np
import math
from tqdm import tqdm, trange
#import matplotlib.pyplot as plt
#import seaborn
import pandas as pd
# Set seed for reproducibility
torch.manual_seed(123)


class Net(object):
    """The GSC Network"""

    def __init__(self, grammar, extData_path="data/inp_pandas.csv"):

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
        self.extData_path = extData_path
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
        self.settings["epochs"] = 5  # Training epochs
        self.settings["timeStep"] = .0001
        self.settings["tgtStd"] = 0.00125
        self.settings["TInit"] = -1
        self.settings["TMin"] = 0
        self.settings["TdecayRate"] = 0.05
        self.settings["lambdaInit"] = 0.11
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
        """Variables to keep track in the learning phase

        It may seem a little bit redundant but to have a "settings" and a "vars"
        dictionary has its reasons.
        The settings are reference values, which cannot be updated or modified while training.
        Vars stores the traces, constantly changing in the training loop. These are reset to the
        reference values in settings after each epoch.

        """
        self.vars = dict()
        # TODO: somewhen toward ends I'll have to update this list
        self.vars["var_names"] = [
            'Activation', 'Harmony', 'H0', 'Q', 'q', 'Temp', 'time', 'ema_speed', 'speed']

        # Temperature params
        self.vars['T_init'] = -1
        self.vars['T_min'] = 0.
        self.vars['T_decay_rate'] = 1e-3
        # Bowl params
        self.vars['q_init'] = 2  # initial strength for the bowl
        self.vars['q_max'] = 200.
        self.vars['q_rate'] = 10.
        self.vars['bowl_center'] = 0.5
        self.vars['bowl_strength'] = None
        self.vars['beta_min_offset'] = 0.1
        # Time step params
        self.vars['max_dt'] = 0.01
        self.vars['min_dt'] = 0.0005
        self.vars['dt'] = 0.001
        # Training traces
        self.vars['s_trace'] = None
        self.vars['prev_s'] = None
        self.vars['Harmony_trace'] = None
        self.vars['speed_trace'] = None
        self.vars['ema_trace'] = None
        self.vars['lambda_trace'] = None
        self.vars['time_trace'] = None
        self.vars['TP_trace'] = None
        self.vars['TPnum_trace'] = None
        self.vars['TP_h_trace'] = None
        self.vars['TP_dist_trace'] = None

    def reset(self):
        pass  # TODO:

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
        # self.debug_copies()

        # Bowl
        self.bowl = Bowl(self)
        self.vars['bowl_center'] = self.bowl.center
        self.vars['bowl_strength'] = self.bowl.strength + \
            self.vars['beta_min_offset']
        self.vars['zeta_bowl'] = self.toNeural(self.vars['bowl_center'])

        # Calculate the recommended Lambda and Temp values:
        self.check_Q_T_lambda()

        # Calculate the weights and biases for the Bowl:
        self.Bowl_bC, self.Bowl_bS = self.bowl.set_biases()
        self.Bowl_WC, self.Bowl_WS = self.bowl.set_weights()

        # Add the bowl weights to the Harmoniy weights
        self.merge_bowl()

        # Initialize states
        self.initialize_state()
        self.create_result_states()

        # Log default parameters
        self.logger(default_settings="#"*25 + " DEFAULT SETTINGS" + "#"*25)
        self.logger(default_settings=self.__dict__)
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

    def merge_bowl(self):
        """Add bowl biases and weights to the Harmony biases and weights."""
        self.B += self.Bowl_bS
        self.Bc += self.Bowl_bC
        self.W += self.Bowl_WS
        self.Wc += self.Bowl_WS

    def debug_copies(self):
        """Create debug copies of the initialized matrices
        #TODO: Why? Can we delete this?
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
        #self.LV_W = self.LotkaVolterra_Weights2()

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

        #FIXME: Improved by delimiting the range where we now the floor division for i and j 
        will be equal. Still very slow. 

        #TODO: Check that the dimensions are right!
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
        #self.vars['q_rec'], self.vars['q_rec_nd'] = self.recommend_Q()

        self.vars['lambda_rec'] = self.recommend_L()
        # Check lambdas
        # 1e-2 tolerance #TODO: review tol
        if abs(self.settings['lambdaInit'] - self.vars['lambda_rec']) > 1e-3:
            print(
                f"LAMBDA RECOMMENDED: {self.vars['lambda_rec']}, ACTUAL LAMBDA = {self.settings['lambdaInit']}")
            choice = input(
                "If you want to change to the recommended value press 'y', else any other key:")
            if choice.lower() == 'y':
                # TODO: clean this chaos up.... What does belong to settings, what to vars, what to encodings?
                self.vars['lambdaInit'] = self.vars['lambda_rec']
            else:
                self.vars['lambdaInit'] = self.settings['lambdaInit']
            self.logger(lambda_value=self.vars['lambdaInit'])

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
    # ---------------------- STATES AND INPUTS --------------------

    def initialize_state(self):
        """Initialize network states and load external inputs"""
        self.state = self.initializer((self.nSym, 1))
        self.inp_s = self.initializer((self.nSym, 1))
        self.inp_c = self.initializer((self.nSym, 1))

        # Collect external inputs
        self.readInput()

        # Initialize Lotka Volterra
        self.LV_Matrices()

        # Initialize Temperature and Lambda
        # TODO: Does it makes sense to calculate, recommend and initialize
        # if we then just allocate zero values here?
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

        # Save the numbers of TP final states
        self.final_TPnum = torch.zeros(
            (self.nStimuli, self.settings['epochs']))

        # Reaction times
        self.reaction_times = torch.zeros(
            (self.nStimuli, self.settings['epochs']))

        # Divergence
        self.divergence = torch.zeros((self.nStimuli, self.settings['epochs']))

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
        """Transform Conceptual vectors into Neural vectors
        The dimension should be (nSym, 1)
        """
        if matrix is None:
            matrix = self.activationC

        matrix = fortran_reshape(matrix, (matrix.numel(), 1))
        return torch.matmul(self.TP, matrix)

    def toConceptual(self, matrix=None):
        """Transform Neural vectors into local vectors."""

        if matrix is None:
            matrix = self.activation
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
        for epoch in trange(self.settings['epochs'], desc="Epoch routine:"):
            for stimulus in trange(self.nStimuli, desc=f"Stimulus routine:"):
                stim_vec = self.stimuli[stimulus, :, :]
                diverge_prob, harmony = self.process_stimulus(stim_vec)
                # Update after stimulus processing
                self.update_res_stim()
                print(
                    f"\nLast best Harmony: {float(harmony)}\n")
            # Update values after each epoch
            self.update_res_epoch()
        self.final_update()

    # -----------------------  PROCESSING AND UPDATE ---------------------------------
    def process_stimulus(self, stimulus):
        """Process a single stimulus."""
        diverge_prob = False
        self.init_for_run(stimulus)
        harmony = self.calc_max_Harmony()

        # Update after each step
        self.updateAfterStep(harmony)

        return diverge_prob, harmony

    def updateAfterStep(self, harmony):
        """Record state after each step.

        This updates the traces initialized in init_for_run()

        """
        TP_state, winner, state_name, binding, Cdist, Sdist, state_num, TP_h = self.calc_nearest_state()

        self.vars['s_trace'] = self.state
        self.vars['Harmony_trace'] = harmony
        self.vars['speed_trace'] = self.calc_speed()
        self.vars['ema_trace'] = self.calc_ema()
        self.vars['lambda_trace'] = self.vars['lambda']
        self.vars['time_trace'] = self.vars['T']
        self.vars['TP_trace'] = state_name
        self.vars['TPnum_trace'] = state_num
        self.vars['TP_h_trace'] = TP_h
        self.vars['TP_dist_trace'] = Cdist
        self.vars['winners'] = winner

        # Log the updated values
        self.logger(step=self.vars['step'])
        self.logger(traces=self.vars)

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
        self.vars['s_trace'] = torch.zeros((max_steps, self.nSym))
        self.vars['prev_s'] = torch.tensor(
            [float('inf')]) * torch.ones((self.nSym, 1))
        self.vars['Harmony_trace'] = torch.zeros((max_steps, 1))
        self.vars['speed_trace'] = torch.zeros((max_steps, 1))
        self.vars['ema_trace'] = torch.zeros((max_steps, 1))
        self.vars['lambda_trace'] = torch.zeros((max_steps, 1))
        self.vars['time_trace'] = torch.zeros((max_steps, 1))
        self.vars['TP_trace'] = torch.zeros((max_steps, 1))
        self.vars['TPnum_trace'] = torch.zeros((max_steps, 1))
        self.vars['TP_h_trace'] = torch.zeros((max_steps, 1))
        self.vars['TP_dist_trace'] = torch.zeros((max_steps, 1))

        # Create the representations for the stimulus
        stimulus = fortran_reshape(
            stimulus, (torch.numel(stimulus), 1)).double()
        self.inp_s = self.TP.matmul(stimulus)
        self.inp_c = self.TP.T.matmul(self.inp_s)

        # Initialize state
        self.initial_state = self.settings['initStateMean'] * torch.rand(
            (self.grammar.nF, self.grammar.nR)) * self.settings["initStateStdev"]
        self.initial_state = self.initial_state.double()
        self.state = self.toNeural(self.initial_state)

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
        self.inp_s = self.toNeural(self.inp_c)
        self.state = torch.linalg.pinv(self.W).matmul(-self.B - self.inp_s)
        self.stateC = self.toConceptual(self.state)
        harmony = self.calc_harmony()
        return harmony

    def calc_harmony(self, state=None):
        """Calculate Harmony value"""

        if state is None:
            state = self.state

        harmony = (self.B + self.inp_s).T.matmul(self.state)
        harmony += .5 * self.state.T.matmul(self.W).matmul(self.state)
        return harmony

    def calc_speed(self):
        """Calc the speed at which the Learning is improving"""
        pass

    def calc_ema(self):
        """Calc the ema value"""
        pass

    def calc_nearest_state(self):
        """Calc the nearest TP state and the most probable winner"""
        self.stateC = self.toConceptual(self.state)
        CTP, winners = self.find_winner()
        self.vars['winners'] = winners

        state_name = self.find_TPname()
        binding = self.find_symBinding()
        state_num = self.find_TPnum()
        TP_state = self.TP.matmul(fortran_reshape(CTP, (torch.numel(CTP), 1)))
        Cdist = self.frobenius(CTP - self.stateC)
        Sdist = self.L2norm(TP_state - self.state)
        TP_h = self.calc_harmony(state=TP_state)

        return TP_state, winners, state_name, binding, Cdist, Sdist, state_num, TP_h

    def find_winner(self):
        """Return a binary Matrix that implements the winner-takes-all strategy"""
        winners_idx = []
        # Extract the max value of each col in the conceptual state matrix
        #state = fortran_reshape(self.stateC, self.stateC.shape)
        # TODO: check the usual issue with rows and cols
        for r in range(self.stateC.shape[1]):
            winners_idx.append(int(torch.argmax(self.stateC[:, r])))

        M = torch.zeros_like(self.stateC)
        # Populate the matrix
        for r in range(M.shape[1]):
            M[winners_idx[r], r] = 1
        return M, winners_idx

    def find_TPname(self):
        "Concatenate the winning fillers to give the name of the nearest TP state"
        TP_name = ""
        for winner in self.vars['winners']:
            filler = self.index2filler[winner]
            TP_name += filler
        return TP_name

    def find_TPnum(self):
        pass

    def find_symBinding(self):
        pass

    def frobenius(self, array):
        pass

    def L2norm(self, array):
        pass
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
