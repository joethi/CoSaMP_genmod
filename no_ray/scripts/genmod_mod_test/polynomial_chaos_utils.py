import numpy as np
import scipy.special as sps
import pandas as pd
"""
@author 1: Jacqui Wentz, Postdoctoral fellow, University of Colorado Boulder.
@author 2: Jothi Thondiraj, MS student, University of Colorado Boulder.
"""
def make_Psi(y, multi_index_matrix,chc_poly):
    """Evaluate Legendre polynomial for each multi-index.

    Args:
        y (n x d np.array): n datapoints of dimension
        multi_index_matrix (P x d np.array): P multi-indices of length d

    Returns:
        P x d np.array: matrix of Legendre polynomial evaluations

    """
    if chc_poly == 'Legendre':
        P = np.size(multi_index_matrix, 0)
        d = np.size(multi_index_matrix, 1)
        n = np.size(y, 0)
    
        Psi = np.ones((n, P))
        p = np.max(multi_index_matrix)  # largest polynomial order
    
        # Iterate through the datapoints
        for k in range(n):
            Leg = np.zeros((p + 1, d))
    
            # Evalulate 1D Legendre poly for each degree at datapoints.
            # Note that we do not include 1/sqrt(2) in these values since we are
            # othogonalizing using the 1/2 measure.
    
            for i in range(p + 1):
                Pn = sps.legendre(i)
                Leg[i, :] = Pn(y[k, :]) * np.sqrt((2 * i + 1))
    
            # Iterate through multi-indices and calculate multi-dim polynomial
            for i in range(P):
                alpha = multi_index_matrix[i, :]
                for j in range(np.size(alpha, 0)):
                    Psi[k, i] = Psi[k, i] * Leg[alpha[j], j]
    elif chc_poly=='Hermite':
        P = np.size(multi_index_matrix, 0)
        d = np.size(multi_index_matrix, 1)
        n = np.size(y, 0)

        Psi = np.ones((n, P))
        p = np.max(multi_index_matrix)  # largest polynomial order

        # Iterate through the datapoints
        for k in range(n):
            Hmt = np.zeros((p + 1, d))

            # Evalulate 1D Legendre poly for each degree at datapoints.
            # Note that we do not include 1/sqrt(2) in these values since we are
            # othogonalizing using the 1/2 measure.
            # Check orthogonality of physicist and probabilistic Hermite polynomials.
            for i in range(p + 1):
                Pn = sps.hermite(i)
                Hmt[i, :] = (Pn(y[k, :]/np.sqrt(2)) * 2**(-i/2))/np.sqrt(sps.factorial(i)) 
            if k ==0:
                df_Hmt = pd.DataFrame(Hmt)
                df_Hmt.to_csv('Hmt_samp.csv',index=False)
            # Iterate through multi-indices and calculate multi-dim polynomial
            for i in range(P):
                alpha = multi_index_matrix[i, :]
                for j in range(np.size(alpha, 0)):
                    Psi[k, i] = Psi[k, i] * Hmt[alpha[j], j]                    

    return Psi
def make_Psi_prll(y, multi_index_matrix,chc_poly,k):
    """Evaluate Legendre polynomial for each multi-index.

    Args:
        y (n x d np.array): n datapoints of dimension
        multi_index_matrix (P x d np.array): P multi-indices of length d

    Returns:
        P x d np.array: matrix of Legendre polynomial evaluations

    """
    if chc_poly == 'Legendre':
        P = np.size(multi_index_matrix, 0)
        d = np.size(multi_index_matrix, 1)
        n = np.size(y, 0)
    
        Psi = np.ones((n, P))
        p = np.max(multi_index_matrix)  # largest polynomial order
    
        # Iterate through the datapoints
        #for k in range(n):
        Leg = np.zeros((p + 1, d))

        # Evalulate 1D Legendre poly for each degree at datapoints.
        # Note that we do not include 1/sqrt(2) in these values since we are
        # othogonalizing using the 1/2 measure.

        for i in range(p + 1):
            Pn = sps.legendre(i)
            Leg[i, :] = Pn(y[k, :]) * np.sqrt((2 * i + 1))

        # Iterate through multi-indices and calculate multi-dim polynomial
        for i in range(P):
            alpha = multi_index_matrix[i, :]
            for j in range(np.size(alpha, 0)):
                Psi[k, i] = Psi[k, i] * Leg[alpha[j], j]
    elif chc_poly=='Hermite':
        P = np.size(multi_index_matrix, 0)
        d = np.size(multi_index_matrix, 1)
        n = np.size(y, 0)

        Psi = np.ones((n, P))
        p = np.max(multi_index_matrix)  # largest polynomial order
        # Iterate through the datapoints
        #for k in range(n):
        Hmt = np.zeros((p + 1, d))
        # Evalulate 1D Legendre poly for each degree at datapoints.
        # Note that we do not include 1/sqrt(2) in these values since we are
        # othogonalizing using the 1/2 measure.
        # Check orthogonality of physicist and probabilistic Hermite polynomials.
        for i in range(p + 1):
            Pn = sps.hermite(i)
            Hmt[i, :] = (Pn(y[k, :]/np.sqrt(2)) * 2**(-i/2))/np.sqrt(sps.factorial(i)) 
        # Iterate through multi-indices and calculate multi-dim polynomial
        for i in range(P):
            alpha = multi_index_matrix[i, :]
            for j in range(np.size(alpha, 0)):
                Psi[k, i] = Psi[k, i] * Hmt[alpha[j], j]                    

    return Psi[k,:]

def make_Psi_hermite(y, multi_index_matrix):
    """Evaluate Legendre polynomial for each multi-index.

    Args:
        y (n x d np.array): n datapoints of dimension
        multi_index_matrix (P x d np.array): P multi-indices of length d

    Returns:
        P x d np.array: matrix of Legendre polynomial evaluations

    """
    P = np.size(multi_index_matrix, 0)
    d = np.size(multi_index_matrix, 1)
    n = np.size(y, 0)

    Psi_hrmt = np.ones((n, P))
    p = np.max(multi_index_matrix)  # largest polynomial order

    # Iterate through the datapoints
    for k in range(n):
        Hmt = np.zeros((p + 1, d))

        # Evalulate 1D Legendre poly for each degree at datapoints.
        # Note that we do not include 1/sqrt(2) in these values since we are
        # othogonalizing using the 1/2 measure.
        # Check orthogonality of physicist and probabilistic Hermite polynomials.
        for i in range(p + 1):
            Pn = sps.hermite(i)
            Hmt[i, :] = (Pn(y[k, :]/np.sqrt(2)) * 2**(-i/2))/np.sqrt(sps.factorial(i)) 
        if k ==0:
            df_Hmt = pd.DataFrame(Hmt)
            df_Hmt.to_csv('Hmt_samp.csv',index=False)
        # Iterate through multi-indices and calculate multi-dim polynomial
        for i in range(P):
            alpha = multi_index_matrix[i, :]
            for j in range(np.size(alpha, 0)):
                Psi_hrmt[k, i] = Psi_hrmt[k, i] * Hmt[alpha[j], j]
                    
    return Psi_hrmt
def test_make_Psi_Hmt(i, j, y, multi_index_matrix,Psi):
    """Use to test make_Psi function."""
    # Psi = make_Psi_hermite(y, multi_index_matrix)
    d = np.size(y, 1)
    yi = y[i, :]

    # Note that we do not include 1/sqrt(2) in these values since we are
    # othogonalizing using the 1/2 measure.
    oned_leg = np.zeros((5, d))
    oned_leg[0, :] = 1.0
    oned_leg[1, :] = yi 
    oned_leg[2, :] = yi**2 - 1
    oned_leg[3, :] = yi**3 - 3*yi
    oned_leg[4, :] = yi**4 - 6*yi**2 + 3 

    mi_sample = multi_index_matrix[j, :]
    leg = 1
    for k in range(d):
        leg = leg * oned_leg[mi_sample[k], k]

    assert (np.abs(leg - Psi[i, j])<1e-12)
def make_Psi_drn(y, multi_index_matrix,lam,chc_poly):
    """Evaluate Legendre polynomial for each multi-index.

    Args:
        y (n x d np.array): n datapoints of dimension
        multi_index_matrix (P x d np.array): P multi-indices of length d

    Returns:
        P x d np.array: matrix of Legendre polynomial evaluations

    """
    if chc_poly=='Legendre':
        P = np.size(multi_index_matrix, 0)
        d = np.size(multi_index_matrix, 1)
        n = np.size(y, 0)
    
        Psi = np.ones((n, P))
        p = np.max(multi_index_matrix)  # largest polynomial order
    
        # Iterate through the datapoints
        for k in range(n):
            Leg = np.zeros((p + 1, d))
    
            # Evalulate 1D Legendre poly for each degree at datapoints.
            # Note that we do not include 1/sqrt(2) in these values since we are
            # othogonalizing using the 1/2 measure.
            for i in range(p + 1):
                Pn = sps.legendre(i)
                Leg[i, :] = Pn(y[k, :]) * np.sqrt((2 * i + 1))
    
            # Iterate through multi-indices and calculate multi-dim polynomial
            for i in lam:
                alpha = multi_index_matrix[i, :]
                for j in range(np.size(alpha, 0)):
                    Psi[k, i] = Psi[k, i] * Leg[alpha[j], j]
    elif chc_poly=='Hermite':
        P = np.size(multi_index_matrix, 0)
        d = np.size(multi_index_matrix, 1)
        n = np.size(y, 0)

        Psi = np.ones((n, P))
        p = np.max(multi_index_matrix)  # largest polynomial order

        # Iterate through the datapoints
        for k in range(n):
            Hmt = np.zeros((p + 1, d))

            # Evalulate 1D Legendre poly for each degree at datapoints.
            # Note that we do not include 1/sqrt(2) in these values since we are
            # othogonalizing using the 1/2 measure.
            # Check orthogonality of physicist and probabilistic Hermite polynomials.
            for i in range(p + 1):
                Pn = sps.hermite(i)
                Hmt[i, :] = (Pn(y[k, :]/np.sqrt(2)) * 2**(-i/2))/np.sqrt(sps.factorial(i)) 
            if k ==0:
                df_Hmt = pd.DataFrame(Hmt)
                df_Hmt.to_csv('Hmt_samp.csv',index=False)
            # Iterate through multi-indices and calculate multi-dim polynomial
            for i in lam:
                alpha = multi_index_matrix[i, :]
                for j in range(np.size(alpha, 0)):
                    Psi[k, i] = Psi[k, i] * Hmt[alpha[j], j]
    return Psi[:,lam]

def test_make_Psi(i, j, y, multi_index_matrix):
    """Use to test make_Psi function."""
    Psi = make_Psi(y, multi_index_matrix)
    d = np.size(y, 1)
    yi = y[i, :]

    # Note that we do not include 1/sqrt(2) in these values since we are
    # othogonalizing using the 1/2 measure.
    oned_leg = np.zeros((5, d))
    oned_leg[0, :] = 1 * np.sqrt(1)
    oned_leg[1, :] = yi * np.sqrt(3)
    oned_leg[2, :] = 1 / 2 * (3 * yi**2 - 1) * np.sqrt(5)
    oned_leg[3, :] = 1 / 2 * (5 * yi**3 - 3 * yi) * np.sqrt(7)
    oned_leg[4, :] = 1 / 8 * (35 * yi**4 - 30 * yi**2 + 3) * np.sqrt(9)

    mi_sample = multi_index_matrix[j, :]
    leg = 1
    for k in range(d):
        leg = leg * oned_leg[mi_sample[k], k]

    assert (leg == Psi[i, j])


def make_mi_mat(d, p):
    """Given a maximum order and dimension of space, find multi-index matrix.

    Args:
        d (int): dimension (# of parameters)
        p (int): maximum polynomial order
    Returns:
        P x d np.array: P possible multi-indices where P = (pd!)/(p!d!)

    """
    # Find number of multi-indices
    # P = int(sps.factorial(d+p)/(sps.factorial(d)*sps.factorial(p)))
    # More stable way to find P
    P = 1
    for i in range(p):
        P = P * (p - i + d)
    P = int(P / sps.factorial(p))

    # Initialize multi-index matrix
    mi_matrix = np.zeros((P, d), dtype=np.int16)

    # For each fixed order, calculate the multi-indices
    row = 0
    for current_order in range(p + 1):
        if current_order == 0:
            row += 1
        else:
            used_rows = int(sps.comb(current_order + d - 1, current_order))
            mi_matrix_p = make_mi_mat_p(d, current_order, used_rows)
            mi_matrix[row:row + used_rows, :] = mi_matrix_p
            row += used_rows
    return(mi_matrix)


def make_mi_mat_p(d, p, r):
    """Given a FIXED order and dimension of space, find multi-index matrix.

    Args:
        d (int): dimension (# of parameters)
        p (int): fixed polynomial order
        r (int): number of multi-index vectors to return

    Returns:
        r x d np.array: r possible multi-indices

    """
    mi_matrix_p = np.zeros((r, d), dtype=np.int16)

    # Put all orders in first element of first row
    mi_matrix_p[0, 0] = p

    # If there is a row left continue
    if r > 1:
        j = 1
        # If the order is greater than zero continue
        while mi_matrix_p[j - 1, 0] > 0:
            # Subtract one from row above
            mi_matrix_p[j, 0] = mi_matrix_p[j - 1, 0] - 1
            # Find the new order, number of elements, and r of next subsystem
            d_new = d - 1
            p_new = p - mi_matrix_p[j, 0]
            used_r = int(sps.comb(p_new + d_new - 1, p_new))
            # Fill in the first column of matrix with correct order
            mi_matrix_p[j + 1:j + used_r, 0] = mi_matrix_p[j, 0]
            # Recursively calculate submatrix next to the column filled above
            mi_submatrix_p = make_mi_mat_p(d_new, p_new, used_r)
            mi_matrix_p[j:j + used_r, 1:d] = mi_submatrix_p
            j += used_r
    return mi_matrix_p


def test_make_mi_mat(d, p):
    """Use to test make_mi_mat function."""
    mi_mat = make_mi_mat(d, p)
    P = np.size(mi_mat, 0)
    # Check that no two rows are the same
    P_check = np.size(np.unique(mi_mat, axis=0), 0)

    assert P_check == P
    # Check that only 0,1,...,p in array
    assert (np.unique(mi_mat) == np.linspace(0, p, p + 1)).all()

    # Check that no row sum is greater than p
    row_sum = np.sum(mi_mat, axis=1)
    assert (np.max(row_sum) == p)
