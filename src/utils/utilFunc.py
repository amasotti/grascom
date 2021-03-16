"""
Collections of auxiliary functions

@date : 07.03.2021
@author: A. Masotti (on the basis of the MATLAB code by Don Mathis)
"""
import torch
import warnings


def fortran_reshape(v, shape):
    """Reshape in Fortran-like order"""
    if len(v.shape) > 1:
        v = v.transpose(1, 0).reshape(shape)
    else:
        v = v.reshape(shape)
    return v

# ------------------------------------------------------------------------------


def is_symmetric(M, rtol=1e-06, atol=1e-08):
    """ Checks if a numpy tensor  is a symmetric matrix """
    return torch.allclose(M, M.T, rtol=rtol, atol=atol)

# ------------------------------------------------------------------------------


def fixed_dotProduct_matrix(n, d, z=0, target_matrix=None):
    """ Creates a matrix of dim (n,d) with a common pairwise dotproduct equal to z

    Params:

      N (int) : determines the number of rows in the final tensor
      d (int) : determines the number of cols in the final tensor
      z (int) or (float) : the value of the dotproduct of each pair of cols

    Returns:

      M (tensor) : a tensor matrix with n col and d rows, each pair has dot product equal to z

    """
    if target_matrix is None:
        target_matrix = z * torch.ones((n, n)) + (1 - z) * torch.eye(n)

    # Sanity check
    if not is_symmetric(target_matrix):
        raise ValueError("The target matrix should be symmetric!")
    if (any(torch.diag(target_matrix) != 1)):
        raise Exception(
            "The elements on the main diagonal of DP_mat should all be equal to 1")

    # Initialize the matrix with the random uniform distribution
    M = torch.rand((n, d))
    M = fortran_reshape(M, (n, d))  # Bring into Fortran-like order

    step0 = .1
    tolerance = 1e-9
    for i in range(1000000):

        # QUESTION: should I use np.matmul() here instead of "*"? MATLAB used
        # matrix multiplication, but with elementwise multiplication, the target matrix is found much faster
        # and has the desiderd property
        inc = torch.matmul(M, torch.matmul(M.T, M) - target_matrix)
        actual_step = .01 / torch.max(torch.abs(inc))
        step = min(actual_step, step0)
        M = M - step * inc

        maxDiff = torch.max(torch.abs(M.conj().T @ M) - target_matrix)
        if maxDiff <= tolerance:
            print(f"dotProducts: Matrix found after {i} iterations")
            assert M.shape == (
                n, d), f"The found matrix has size {M.shape} but should have size {(d,n)}"
            return M
    warnings.warn(
        f"After {i} iteration no matrix was found. the last calculated Matrix will be returned \nConsider re-running this script!")
    return M

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


R = syllablePositionRoles(2, 4, 0, 0)
