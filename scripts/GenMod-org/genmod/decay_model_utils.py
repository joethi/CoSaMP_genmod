import numpy as np
import scipy.special as sps
import sklearn.linear_model as lm
import numpy.linalg as la

def decay_model(z, multiindex_constant, beta_parameter_matrix):
    """Evaluate exponential model."""
    G = multiindex_constant * np.exp(beta_parameter_matrix @ z)
    return G


def decay_model_jacobian(gen_model_eval, beta_parameter_matrix):
    """Evaluate generative model jacobian.
    :param gen_model_eval: evaluation of generative model (k np.array)
    :param beta_parameter_matrix: parameter values (P x k np.array)
    :return: jacobian (P x k np.array)
    """
    G_jac = np.diag(gen_model_eval) @ beta_parameter_matrix

    return G_jac


def make_multiindex_constant(multi_index_matrix):
    """Evaluate constant that depends on the multiindex a (i.e., |a|!/a!).
    :param multi_index_matrix: P multi-indices of length d (P x d np.array)
    :return: multi-index dependent constants (P np.array)
    """
    P = np.size(multi_index_matrix, 0)
    multiindex_constant = np.zeros(P)
    for i in range(P):
        current_multiindex = multi_index_matrix[i, :]
        multiindex_constant[i] = sps.factorial(
            np.sum(current_multiindex)
        ) / np.prod(sps.factorial(current_multiindex))
    return np.sqrt(multiindex_constant)
 

def make_beta_param_mat(d, multi_index_matrix, model='expalg'):
    """Make parameter matrix for exponential+algebraic decay model.
    :param d: Number of dimensions in stochastic spaces (int)
    :param multi_index matrix: P multi-indices of length d (P x d np.array)
    :return: parameter values (P x 2d+1 np.array)
    """
    P = np.size(multi_index_matrix, 0)
    if model == 'expalg':
        beta_parameter_matrix = np.ones((P, 1 + 2 * d))
        beta_parameter_matrix[:, 1:d + 1] = -multi_index_matrix
        beta_parameter_matrix[:, d + 1:2 * d + 1] = -np.log(
            multi_index_matrix + np.ones((P, d))
        )
    if model == 'exp':
        beta_parameter_matrix = np.ones((P, 1 + d))
        beta_parameter_matrix[:, 1:d + 1] = -multi_index_matrix
    return beta_parameter_matrix


def set_omp_signs(Psi, u_data, all=True):
    """Set coefficient signs using Orthogonal Matching Pusuit method."""
    P = np.size(Psi, 1)

    # Find initial signs with orthogonal matching pursuit
    omp = lm.OrthogonalMatchingPursuitCV(cv=5, fit_intercept=False)
    omp.fit(Psi, u_data)
    c = omp.coef_

    idx_nonzero = c.nonzero()
    if all:
        # Find the remaining signs using an additionally iteration of OMP
        error = u_data - Psi @ c
        for i in np.setdiff1d(np.array(range(P)), idx_nonzero):
            c[i] = Psi[:, i] @ error / (Psi[:, i] @ Psi[:, i])

    return(np.sign(c))
def set_lst_signs(Psi, u_data, all=True):
    """Set coefficient signs using Orthogonal Matching Pusuit method."""
    P = np.size(Psi, 1)

    # Find initial signs with orthogonal matching pursuit
    # omp = lm.OrthogonalMatchingPursuitCV(cv=5, fit_intercept=False)
    # omp.fit(Psi, u_data)
    # # Find least squares coefficients
    Psi_ls_T = np.transpose(Psi)
    c_ls = (la.inv(Psi_ls_T @ Psi) @ Psi_ls_T @ u_data).flatten()
    c = c_ls

    idx_nonzero = c.nonzero()
    if all:
        # Find the remaining signs using an additionally iteration of OMP
        error = u_data - Psi @ c
        for i in np.setdiff1d(np.array(range(P)), idx_nonzero):
            c[i] = Psi[:, i] @ error / (Psi[:, i] @ Psi[:, i])

    return(np.sign(c))