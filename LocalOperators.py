# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:37:11 2024

To generate general unitary matrix for n qubits such that we can optimise over
choice in parameters of this matrix for measures that involve extremisation.

@author: Herbie Warner
"""
import numpy as np

def create_givens_rotation(theta, phi, dim, i, j):
    """
    Create a Givens rotation matrix for indices (i, j) in a matrix of size dim x dim.
    
    Parameters:
    theta (float): The rotation angle
    phi (float): The phase angle
    dim (int): The dimension of the matrix
    i (int): The first index
    j (int): The second index
    
    Returns:
    np.ndarray: The Givens rotation matrix
    """
    R = np.eye(dim, dtype=np.complex128)
    c = np.cos(theta)
    s = np.sin(theta) * np.exp(1j * phi)
    
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -np.conj(s)
    R[j, i] = s
    
    return R

def construct_general_unitary(n, thetas, phis):
    """
    Constructs thegeneral unitary matrix for n qubits using given theta
    and phi parameters.
    
    Parameters:
    n (int): Number of qubits
    thetas (array-like): Array of theta parameters
    phis (array-like): Array of phi parameters
    
    Returns:
    np.ndarray: The unitary matrix of dimension 2^n x 2^n
    """
    dimension = 2**n
    num_params = 2**n - 1
    if len(thetas) != num_params or len(phis) != num_params:
        raise ValueError(f"For n={n} qubits, need {num_params} thetas and"
                         " {num_params} phis.")

    U = np.eye(dimension, dtype=np.complex128)

    k = 0
    for i in range(dimension):
        for j in range(i + 1, dimension):
            theta = thetas[k % num_params]
            phi = phis[k % num_params]
            P = create_givens_rotation(theta, phi, dimension, i, j)
            U = P @ U
            k += 1

    return U


def is_unitary(matrix):
    identity_matrix = np.eye(matrix.shape[0])
    return np.allclose(identity_matrix, np.dot(matrix.conj().T, matrix))

def generate_orthonormal_vectors(n,thetas,phis):
    U = construct_general_unitary(n, thetas, phis)
    basis = np.zeros(2**n,dtype=complex)
    
    orthonormals = []
    for i in range(2**n):
        vector = basis.copy()
        vector[i] = 1
        vector = U.dot(vector)
        orthonormals.append(vector)
    
    return np.array(orthonormals)     