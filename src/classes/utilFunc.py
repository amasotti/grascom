"""
Collections of auxiliary functions

@date : 07.03.2021
@author: A. Masotti (on the basis of the MATLAB code by Don Mathis)
"""
import torch
import warnings
import numpy as np  # Just because Numpy has the "order='F'" option, then convert in tensor


def is_symmetric(M, rtol=1e-06, atol=1e-08):
    """ Checks if a numpy tensor  is a symmetric matrix """
    return np.allclose(M, M.T, rtol=rtol, atol=atol)

# ------------------------------------------------------------------------------


def fixed_dotProduct_matrix(n, d, z, target_matrix=None):
    """Generate a Matrix of random vectors (the representations of our fillers and roles ) 
        such that the pairwise similarity are close within a given tolerance 
        to the numbers specified in z or build the matrix specified as 'target_matrix'.

        The vectors desired are THE COLUMNS of the returned matrix.
    """

    if target_matrix is None:  # if a scalar is passed, build the corresponding symmetric matrix
        target_matrix = (z * np.ones((n, n)) + (1 - z) * np.eye(n))

    # Sanity check
    if not is_symmetric(target_matrix):
        raise 'The target matrix should be symmetric! If A == B, B == A'

    if np.any(np.diag(target_matrix) != 1):
        raise 'The target matrix main diagonal should have only 1s (A == A)'

    # generate d * n random numbers from the Uniform dist.
    M = np.random.uniform(size=d * n)
    # Reshape (d,n) -> the columns will be our vectors with nrows dimensions.
    M = M.reshape(d, n, order='F')
    print(f"First random Matrix: {M}")

    step0 = .1
    tol = 1e-6
    for i in range(1000000):
        inc = (M.T @ M - target_matrix)
        inc = np.dot(M, inc)
        step = min(step0, .01 / abs(inc).max())
        M -= step * inc
        max_diff = np.max(np.abs(M.T @ M - target_matrix))
        if max_diff <= tol:
            print("Representations built after {i} attempts\n")
            return torch.tensor(M)

    print(
        "Desidered matrix not found after {i} attempts. Rerun the script or use the last found matrix!")
    return torch.tensor(M)


# ------------------------------------------------------------------------------


def syllablePositionRoles(n_syl, n_pos, syl_dotP, pos_dotP):
    R = torch.eye(n_syl * n_pos) * 0
    s_roles = fixed_dotProduct_matrix(n_syl, n_syl, syl_dotP)
    p_roles = fixed_dotProduct_matrix(n_pos, n_pos, pos_dotP)
    """# Test with MATLAB
    s_roles = torch.tensor([0.7333, -0.6799, 0.6799, 0.7333]).reshape((2, 2))
    p_roles = torch.tensor([0.2840, -0.2963, 0.8388, 0.3577, 0.7889, 0.5296, -0.1866,
                            0.2497, -0.3240, 0.7715, 0.4880, -0.2481, -0.4381, 0.1909, -0.1530, 0.8650]).reshape((4, 4))
    """
    # Fill R left to right, s in outer loop
    Rcol = 0
    for syl in range(n_syl):
        for pos in range(n_pos):
            v1, v2 = fortran_reshape(
                s_roles[:, syl], (-1, 1)), fortran_reshape(p_roles[:, pos], (1, -1))
            value = torch.matmul(v1, v2)
            value = fortran_reshape(value, (value.numel()))
            R[:, Rcol] = value
            Rcol += 1
    return R
