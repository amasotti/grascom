"""
Collections of auxiliary functions

@date : 07.03.2021
@author: A. Masotti (on the basis of the MATLAB code by Don Mathis)
"""
import torch
import numpy as np  # Just because Numpy has the "order='F'" option, then convert in tensor


def fortran_reshape(array, shape):
    """At moment very sloppy... 
    I've opened an issue on pytorch/pytorch, waiting for the devs
    to support the fortran order in Pytorch"""
    array = array.numpy()
    array = array.reshape(shape, order="F")
    return torch.tensor(array)


def is_symmetric(M, rtol=1e-06, atol=1e-08):
    """ Checks if a numpy tensor  is a symmetric matrix """
    return torch.allclose(M, M.T, rtol=rtol, atol=atol)

# ------------------------------------------------------------------------------


def fixed_dotProduct_matrix(n, d, z=0, target_matrix=None):
    """Generate a Matrix of random vectors (the representations of our fillers and roles ) 
        such that the pairwise similarity are close within a given tolerance 
        to the numbers specified in z or build the matrix specified as 'target_matrix'.

        The vectors desired are THE COLUMNS of the returned matrix.
    """

    if target_matrix is None:  # if a scalar is passed, build the corresponding symmetric matrix
        target_matrix = (z * torch.ones((n, n)) + (1 - z) * torch.eye(n))

    # Sanity check
    if not is_symmetric(target_matrix):
        raise 'The target matrix should be symmetric! If A == B, B == A'

    if torch.any(torch.diag(target_matrix) != 1):
        raise 'The target matrix main diagonal should have only 1s (A == A)'

    # generate d * n random numbers from the Uniform dist.
    M = torch.rand((d, n))
    M = fortran_reshape(M.flatten(), (d, n))

    step0 = .1
    tol = 1e-6
    for i in range(1000000):
        inc = torch.mm(M.T, M) - target_matrix
        inc = torch.mm(M, inc)
        step = min(step0, .01 / torch.abs(inc).max())
        M -= step * inc
        max_diff = torch.max(torch.abs(torch.mm(M.T, M) - target_matrix))
        if max_diff <= tol:
            print(f"Representations built after {i} attempts\n")
            return M

    print(
        f"Desidered matrix not found after {i} attempts. Rerun the script or use the last found matrix!")
    return M
