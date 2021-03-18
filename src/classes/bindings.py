"""
Classes Roles, Fillers

"""
from src.classes.utilFunc import fixed_dotProduct_matrix
import torch
# Set seed for reproducibility
torch.manual_seed(123)


class Roles(object):
    """Roles Class."""

    def __init__(self, roles):
        self.rolesNames = roles
        self.nR = len(self.rolesNames)
        self.R = self.rolesMatrix()
        self.role2index, self.index2role = self.rolesDicts()

    def rolesMatrix(self, dp=0):
        print("Build role Matrix")
        return fixed_dotProduct_matrix(self.nR, self.nR, z=dp)

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
        # Add the padding element
        self.fillersNames.append(emptyFiller)
        self.nF = len(self.fillersNames)

        if fillerSimilarities is None:
            self.similarities = torch.eye(self.nF)
        else:
            self.similarities = fillerSimilarities

        self.F = self.fillersMatrix(self.similarities)
        self.filler2index, self.index2filler = self.fillersDicts()

    def fillersMatrix(self, target_matrix):
        print(f"Buil Filler Matrix")
        return fixed_dotProduct_matrix(self.nF, self.nF, target_matrix=target_matrix)

    def fillersDicts(self):
        """Build dictionary role : idx and the reverse"""
        filler2index = dict()
        for filler in self.fillersNames:
            if filler not in filler2index:
                filler2index[filler] = len(filler2index)

        # Add padding element
        #filler2index[self.emptyFiller] = len(filler2index)

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

    def bindDicts(self, sep="/"):
        """Build a dictionary for the bindings."""
        bind2index = dict()
        for role in self.roles:
            for filler in self.fillers:
                binding = filler + sep + role
                bind2index[binding] = len(bind2index)

        index2bind = {i: b for b, i in bind2index.items()}
        return bind2index, index2bind

    def __len__(self):
        return len(self.bind2index)

    def __repr__(self):
        return f"Binding class with {len(self.bind2index)} items"
