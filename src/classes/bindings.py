"""
Classes Roles, Fillers

"""
from src.classes.utilFunc import fixed_dotProduct_matrix, fortran_reshape
import torch
import itertools


class Roles(object):
    """Roles Class."""

    def __init__(self, roles):
        self.rolesNames = roles

        # Build dictionaries
        self.role2index, self.index2role = self.rolesDicts()
        self.nR = len(self.role2index)

        # Roles Matrix
        self.R = self.rolesMatrix()
        #self.R = self.positionalRoles(dotp=0)
        #self.R = torch.eye(self.nR)

    def rolesMatrix(self, dp=0):
        """Build the role Matrix, dp= 0, i.e. roles are maximally different"""
        #roleSimilarity = torch.tensor([[1, 0, -0.5, - 0.5], [0, 1, 0, 0], [-0.5, 0, 1, 0.1], [-0.5, 0, 0.1, 1]])
        print("Build role Matrix")
        # return fixed_dotProduct_matrix(self.nR, self.nR, z=None, target_matrix=roleSimilarity)
        return fixed_dotProduct_matrix(self.nR, self.nR, z=dp)

    def positionalRoles(self, dotp=0):
        """Build the matrix for positional roles

        Similar to the rolesMatrix() method, this matrix
        takes into account similarity between roles and similarity
        between positions.  

        In the concrete case implemented in this project, we have 4 roles
        that correspond to 4 positions.

        """
        nr = int(self.nR / 2)
        posM = fixed_dotProduct_matrix(nr, nr, z=.1, target_matrix=None)
        rolM = fixed_dotProduct_matrix(nr, nr, z=dotp, target_matrix=None)

        r = 0
        R = torch.zeros((self.nR, self.nR), dtype=torch.double)
        for i in range(nr):
            for j in range(nr):
                p = posM[:, i].unsqueeze(1)
                q = rolM[:, j].unsqueeze(1)
                row = p.mm(q.T)
                row = fortran_reshape(row, (row.numel(), 1))
                R[:, r] = row.squeeze()
                r += 1
        return R

    def rolesDicts(self):
        """Build dictionary role : idx and the reverse"""
        role2index = dict()
        for role in self.rolesNames:
            if role not in role2index:
                role2index[role] = len(role2index)
        index2role = {i: r for r, i in role2index.items()}
        return role2index, index2role

    def __len__(self):
        return self.nR

    def __repr__(self):
        return f"Roles Class with {self.nR} items"


class Fillers(object):
    """Fillers class.

    The filler Matrix and the fillers number are calculated starting from a 
    simple Python list.

    Optionally you can pass a similarity matrix (that should be symmetric and have
    ones on the main diagonal: A = A for each filler A)

    In case the filler Matrix is None, fillers are assumed to be maximally dissimilar
    and equal only to themselves (Identity Matrix).    

    """

    def __init__(self, fillers, fillerSimilarities=None, emptyFiller="#"):

        self.fillersNames = fillers
        # Add the padding element (useful to represent an empty onset or coda)
        # self.fillersNames.append(emptyFiller)

        # Build dictionaries
        self.filler2index, self.index2filler = self.fillersDicts()
        self.nF = len(self.filler2index)

        if fillerSimilarities is None:
            self.similarities = torch.eye(self.nF)
        else:
            self.similarities = fillerSimilarities

        # Filler Matrix
        self.F = self.fillersMatrix(self.similarities)
        #self.F = torch.eye(self.nF)

    def fillersMatrix(self, target_matrix):
        print(f"Buil Filler Matrix")
        return fixed_dotProduct_matrix(self.nF, self.nF, target_matrix=target_matrix)

    def fillersDicts(self):
        """Build dictionary role : idx and the reverse"""
        filler2index = dict()
        for filler in self.fillersNames:
            if filler not in filler2index:
                filler2index[filler] = len(filler2index)

        index2filler = {i: f for f, i in filler2index.items()}
        return filler2index, index2filler

    def __repr__(self):
        return f"Filler class with {self.nF} items"

    def __len__(self):
        return self.nF


class Bindings(object):
    """Binding harmony constraints."""

    def __init__(self, fillers, roles):
        self.fillers = fillers.filler2index
        self.roles = roles.role2index
        self.nF = fillers.nF
        self.nR = roles.nR
        self.F = fillers.F
        self.R = roles.R
        self.bind2index, self.index2bind = self.bindDicts()
        self.states, self.state_rev = self.combine_bindings()

        # Binding Matrix #TODO: Probably not needed
        self.BindM = torch.zeros((self.nF, self.nR))

    def bindDicts(self, sep="/"):
        """Build a dictionary for the bindings."""
        bind2index = dict()
        for role in self.roles:
            for filler in self.fillers:
                binding = filler + sep + role
                bind2index[binding] = len(bind2index)

        index2bind = {i: b for b, i in bind2index.items()}
        return bind2index, index2bind

    def combine_bindings(self):
        """Build a dictionary of all possible combinations
        of roles and fillers

        This dictionary will be later used to assign a name 
        to the produced state.

        """
        # Combinations
        fillers = list(self.fillers.keys())
        combi = itertools.combinations_with_replacement(
            fillers, len(self.roles))
        combi = list(combi)

        # Permutations
        perm = list()
        for state in combi:
            p = itertools.permutations(state)
            for per in p:
                perm.append(per)
        # Clean duplicates
        perm = list(set(perm))

        # simplify format
        states = dict()
        for p in perm:
            p_string = "-".join(p)
            states[p_string] = len(states)
        states_rev = {i: p for p, i in states.items()}
        return states, states_rev

    def __len__(self):
        return len(self.bind2index)

    def __repr__(self):
        return f"Binding class with {len(self.bind2index)} items"
