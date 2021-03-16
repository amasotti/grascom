"""
Classes Roles, Fillers

"""
from src.utils.utilFunc import fixed_dotProduct_matrix
import torch


class Roles(object):
    """Roles Class."""

    def __init__(self, roles):
        self.rolesNames = roles
        self.nR = len(self.rolesNames)
        self.R = self.rolesMatrix()

    def rolesMatrix(self, dp=0):
        return fixed_dotProduct_matrix(self.nR, self.nR, z=dp)


class Fillers(object):
    """Fillers class."""

    def __init__(self, fillers, fillerSimilarities=None):
        self.fillersNames = fillers
        self.nF = len(self.fillersNames)

        if fillerSimilarities is None:
            self.similarities = torch.eye(self.nF)
        self.F = self.fillersMatrix(dp=self.similarities)

    def fillersMatrix(self, dp=0):
        return fixed_dotProduct_matrix(self.nF, self.nF, target_matrix=self.similarities)


if __name__ == '__main__':
    fillers = ['b' 'b^h' 'u' 'd' 'd^h' '#']
    F = Fillers(fillers)
