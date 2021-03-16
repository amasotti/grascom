"""
Classes Roles, Fillers

"""
import torch
from src.classes.utilFunc import fixed_dotProduct_matrix

# Set seed for reproducibility
torch.seed(123)


class Roles(object):
    """Roles Class."""

    def __init__(self, roles):
        self.rolesNames = roles
        self.nR = len(self.rolesNames)
        self.R = self.rolesMatrix()

    def rolesMatrix(self, dp=0):
        print("Build role Matrix")
        return fixed_dotProduct_matrix(self.nR, self.nR, z=dp)


class Fillers(object):
    """Fillers class."""

    def __init__(self, fillers, fillerSimilarities=None):
        self.fillersNames = fillers
        self.nF = len(self.fillersNames)

        if fillerSimilarities is None:
            self.similarities = torch.eye(self.nF)
        self.F = self.fillersMatrix(self.similarities)

    def fillersMatrix(self, target_matrix):
        print(f"Buil Filler Matrix")
        return fixed_dotProduct_matrix(self.nF, self.nF, target_matrix=target_matrix)


class Bindings(object):
    """Binding harmony constraints."""

    def __init__(self, fillers, roles):
        self.fillers = fillers.fillersNames
        self.roles = roles.rolesNames
        self.nF = fillers.nF
        self.nR = roles.nR
        self.F = fillers.F
        self.R = roles.R

        # Single constituent Harmony - Initialize as zero (all const have the same Harmony)
        self.Hc = torch.zeros((self.nF, self.nR), dtype=torch.float)
        self.Hcc = torch.zeros(
            (self.nF, self.nR, self.nF, self.nR), dtype=torch.float)

    def singleHarmony_constraints(self, constraints):
        """Update single Harmony constraints given a list of tuples.

        Each tuple should be in the form:

        ("bh/r1", 3) = bh as filler for the role r1 improves the Harmony by 3 points
        ("bh/r1", -4) = bh as filler for the role r1 lowers the Harmony by 4 points

        """
        for constraint in constraints:
            binding = constraint[0]
            harmony = constraint[1]

            fillerName = binding.split("/")[0]
            roleName = binding.split("/")[1]

            fillerIdx = self.fillers.index(fillerName)
            roleIdx = self.roles.index(roleName)

            self.Hc[fillerIdx, roleIdx] = harmony
        return self.Hc

    def pairHarmony_constraints(self, constraints):
        """Update pairwise Harmony constraints given a list of tuples.

    Each tuple should be in the form:

    ("bh/r1", "b/r2", 3) = the co-occurence of bh in role1 and b in role 2 improves the Harmony by 3

    """
        for constraint in constraints:
            binding1 = constraint[0]
            binding2 = constraint[1]
            harmony = constraint[2]

            fillerName1 = binding1.split("/")[0]
            roleName1 = binding1.split("/")[1]
            fillerIdx1 = self.fillers.index(fillerName1)
            roleIdx1 = self.roles.index(roleName1)

            fillerName2 = binding2.split("/")[0]
            roleName2 = binding2.split("/")[1]
            fillerIdx2 = self.fillers.index(fillerName2)
            roleIdx2 = self.roles.index(roleName2)

            # Update the matrix (KEEP IT SYMMETRIC at top level!)
            self.Hcc[fillerIdx1, roleIdx1, fillerIdx2, roleIdx2] = harmony
            self.Hcc[fillerIdx2, roleIdx2, fillerIdx1, roleIdx1] = harmony

        return self.Hcc
