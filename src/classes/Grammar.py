"""

The Grammar class joining Bindings of F/R and Harmony
constraints

"""
from src.classes.bindings import Fillers, Roles, Bindings


class Grammar(object):
    """The Harmonic Grammar."""

    def __init__(self, fillersList, rolesList, fillerSimilarities=None):
        self.fillers = Fillers(fillersList, fillerSimilarities)
        self.roles = Roles(rolesList)
        self.bind = Bindings(self.fillers, self.roles)
        self.Hc = self.bind.Hc
        self.Hcc = self.bind.Hcc

    def update_Hc(self, constraints):
        """Constraint form: 
          ("bh/r1", -4)
        """
        self.Hc = self.bind.singleHarmony_constraints(constraints=constraints)

    def update_Hcc(self, constraints):
        """Constraint form: 
          ("bh/r1", "b/r2", -4)
        """
        self.Hcc = self.bind.pairHarmony_constraints(constraints)

    @classmethod
    def grammar_from_Input(cls, s):
        """Read input from string."""
        # Extract segments
        s = list(s)

        fillers = s
        roles = ["r"+str(i) for i in range(len(s))]
        return cls(fillers, roles)


# ---------------- TESTING -----------------------------
if __name__ == '__main__':
    G = Grammar.read_Input("Antonio")
    sCon = [("a/r0", 7), ("n/r0", -10)]

    constr = []
    for i in G.fillers.fillersNames:
        for j in G.roles.rolesNames:
            for k in G.roles.rolesNames:
                if j != k:
                    c = (f"{i}/{j}", f"{i}/{k}", -10)
                    constr.append(c)
