"""

The Grammar class joining Bindings of F/R and Harmony
constraints

"""
from pandas.core.base import DataError
from src.classes.bindings import Fillers, Roles, Bindings
from src.gsc.gsc_network import Net
from src.classes.utilFunc import is_symmetric
import torch


class Grammar(object):
    """The Harmonic Grammar."""

    def __init__(self, fillersList, rolesList, fillerSimilarities=None):

        # Roles, fillers and bindings
        self.fillers = Fillers(fillersList, fillerSimilarities)
        self.roles = Roles(rolesList)
        self.bind = Bindings(self.fillers, self.roles)

        # Dimensions
        self.nF = len(self.fillers)
        self.nR = len(self.roles)
        self.nS = self.nF * self.nR

        # Single constituent Harmony - Initialize as zero (all const have the same Harmony)
        self.Hc = torch.zeros((self.nF, self.nR), dtype=torch.float)
        self.Hcc = torch.zeros(
            (self.nF, self.nR, self.nF, self.nR), dtype=torch.float)
        self.check_symmetry_hcc()

    def update_Hc(self, constraints: list):
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
            try:
                fillerIdx = self.fillers.filler2index[fillerName]
                roleIdx = self.roles.role2index[roleName]
            except:
                raise KeyError(
                    "Hc matrix: Either a filler or a role was misspelled or not present in the dictionary")

            self.Hc[fillerIdx, roleIdx] = harmony

    def update_Hcc(self, constraints: list):
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
            try:
                fillerIdx1 = self.fillers.filler2index[fillerName1]
            except:
                raise KeyError(
                    f"The filler '{fillerName1}' in {constraint} is not in the dictionary. Check evt. misspellings!")
            try:
                roleIdx1 = self.roles.role2index[roleName1]
            except:
                raise KeyError(
                    f"The role '{roleName1}' in {constraint} is not in the dictionary. Check evt. misspellings!")

            fillerName2 = binding2.split("/")[0]
            roleName2 = binding2.split("/")[1]
            try:
                fillerIdx2 = self.fillers.filler2index[fillerName2]
            except:
                raise KeyError(
                    f"The filler '{fillerName2}' in {constraint} is not in the dictionary. Check evt. misspellings!")

            try:
                roleIdx2 = self.roles.role2index[roleName2]
            except:
                raise KeyError(
                    f"The role '{roleName2}' in {constraint} is not in the dictionary. Check evt. misspellings!")

            # Update the matrix (symmetric at top level!)
            self.Hcc[fillerIdx1, roleIdx1, fillerIdx2, roleIdx2] = harmony
            self.Hcc[fillerIdx2, roleIdx2, fillerIdx1, roleIdx1] = harmony

        self.check_symmetry_hcc()

    @classmethod
    def grammar_from_Input(cls, s):
        """Read input from string.

        Mainly for test purpose. Given a string, 
        it extracts the letters as fillers and the positions 
        as roles and builds the corresponding bindings.

        """
        # Extract segments
        s = list(s)

        fillers = s
        roles = ["r"+str(i) for i in range(len(s))]
        return cls(fillers, roles)

    def check_symmetry_hcc(self):
        """Checks the toplevel symmetry of the pairwise harmony matrix."""

        M = self.Hcc.reshape((self.nF * self.nR, -1))
        if not is_symmetric(M):
            raise DataError(
                "The Hcc Matrix should be symmetric at the top level!")

    def __repr__(self):
        return f"Harmonic Grammar with {len(self.fillers)} fillers and {len(self.roles)} roles"

    def __len__(self):
        """Grammar length as total symbols."""
        return len(self.fillers) * len(self.roles)


# ---------------- TESTING -----------------------------
if __name__ == '__main__':
    G = Grammar.grammar_from_Input("test-")
    sCon = [("t/r0", 7), ("-/r0", -10)]

    constr = []
    for i in G.fillers.fillersNames:
        for j in G.roles.rolesNames:
            for k in G.roles.rolesNames:
                if j != k:
                    c = (f"{i}/{j}", f"{i}/{k}", -10)
                    constr.append(c)
    extra = [("t/r1", "e/r0", -990)]

    fil = list("abrjot")
    roles = ["r1", "r2", "r4"]
    filSim = torch.eye(6)
    filSim[0, 1] = .75
    filSim[1, 0] = .75
    F = Grammar(fil, roles, filSim)

    with open('testing.txt', "w", encoding="utf-8") as test:
        for k, v in F.bind.__dict__.items():
            test.write(f"{str(k)}\n{str(v)}\n")
            test.write("#"*80 + "\n")
