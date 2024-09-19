# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:04:04 2024

All the entropic/correlation measures needed

@author: Herbie Warner
"""
from Utilities import (compute_entropy, partial_trace_multiple_qubits,
 validate_qubit_mapping, get_number_of_qubits, canonical_purify,
 translate_qubit_map)
from LocalOperators import generate_orthonormal_vectors
from scipy.optimize import differential_evolution
from scipy.linalg import eigh
import numpy as np

def compute_mutual_information(rho, qubit_mapping):
    validate_qubit_mapping(qubit_mapping, np.log2(rho.shape[0]))
    (_, qubitsA), (_, qubitsB) = qubit_mapping.items()


    rhoA = partial_trace_multiple_qubits(rho, qubitsB)
    rhoB = partial_trace_multiple_qubits(rho, qubitsA)
        
    S_A = compute_entropy(rhoA)
    S_B = compute_entropy(rhoB)
    S_AB = compute_entropy(rho)
    return S_A + S_B - S_AB

def partial_trace_vector_to_state_vector(state_vector, qubits):
    """
    Perform partial trace over specified subsystems for a given 
    state vector of an n-qubit system.
    
    Args:
    state_vector (ndarray): The state vector of the full system.
    num_qubits (int): Total number of qubits in the system.
    traced_out_subsys (list): List of indices of qubits to be traced out
    (0-based indexing).
    
    Returns:
    ndarray: The reduced density matrix after tracing out specified qubits.
    """
    num_qubits = int(np.log2(len(state_vector)))
    
    traced_out_subsys = [qb - 1 for qb in qubits]
      
    remaining_subsys = [i for i in range(num_qubits) if i not in traced_out_subsys]
    remaining_dims = 2 ** len(remaining_subsys)
    
    reshaped_state = state_vector.reshape([2] * num_qubits)
    

    rho_reduced = np.zeros((remaining_dims, remaining_dims), dtype=complex)
    
    for indices in np.ndindex((2,) * len(traced_out_subsys)):

        slices = [slice(None)] * num_qubits
        for j, index in enumerate(traced_out_subsys):
            slices[index] = indices[j]
        sliced_state = reshaped_state[tuple(slices)]
        remaining_state = sliced_state.flatten()
        rho_reduced += np.outer(remaining_state, np.conj(remaining_state))

    rho_reduced /= np.trace(rho_reduced)    
    return rho_reduced

def compute_reflected_entropy(rhoAB, qubit_mapping):
    tol = 1e-6
    num_of_qubits = int(np.log2(rhoAB.shape[0]))
    (_, _), (_, qubitsB) = qubit_mapping.items()
    qubitsB = np.concatenate((qubitsB, qubitsB + num_of_qubits))
    
    eigenvalues, eigenvectors = eigh(rhoAB)
    dim = rhoAB.shape[0]
    psi = np.zeros(dim * dim, dtype=complex)
    
    mask = eigenvalues > tol
    filtered_eigenvalues = eigenvalues[mask]
    filtered_eigenvectors = eigenvectors[:, mask]

    for i, ev in enumerate(filtered_eigenvalues):
        if ev.real > 0:
            psi += np.sqrt(ev) * np.kron(filtered_eigenvectors[:, i],
                                         filtered_eigenvectors[:, i])


    rho_A_Astar = partial_trace_vector_to_state_vector(psi, qubitsB)
  
    return compute_entropy(rho_A_Astar)


def compute_markov_gap(rho,qubit_map):
    SR = compute_reflected_entropy(rho, qubit_map)
    M = compute_mutual_information(rho, qubit_map)
    return SR-M


   
def compute_DW(rhoAB, qubit_mapping):
    num_of_qubits = get_number_of_qubits(rhoAB)
    validate_qubit_mapping(qubit_mapping, num_of_qubits)
    (_, qubitsA), (_, qubitsB) = qubit_mapping.items()
    
     
    rhoB = partial_trace_multiple_qubits(rhoAB, qubitsA)
    S_B = compute_entropy(rhoB)
    S_AB = compute_entropy(rhoAB)
    
  
    rhoABC =   canonical_purify(rhoAB)
    rhoAC = partial_trace_multiple_qubits(rhoABC, qubitsB)
    
    qubitsC = np.array(range(num_of_qubits+1, 2*num_of_qubits + 1))
    qubitsC = translate_qubit_map(qubitsB, qubitsC)
    
    qubitsAcopy = qubitsA.copy()
    qubitsAcopy = translate_qubit_map(qubitsB, qubitsAcopy)

    new_qb_mapping = {
        "System A": qubitsAcopy,
        "System C": qubitsC
        }
    EW = compute_reflected_entropy(rhoAC, new_qb_mapping)/2
    return S_B - S_AB + EW
    


def compute_minimised_SA(rhoAB, qubit_mapping):
    """
    Assumes qubits are continuous, qubits from A, and B are
    not disjoint within each other. Assumes qubits A comes first due
    to operator generation with tensor product

    Parameters
    ----------
    rhoAB : numpy array
        The density matrix of the combined system AB.
    qubit_mapping : dict
        A dictionary mapping system names to their respective qubits.

    Returns
    -------
    J : float
        The computed J value.
    best_thetas : numpy array
        The optimized theta values.
    best_phis : numpy array
        The optimized phi values.
    """
    
    
    (_, qubitsA), (_, qubitsB) = qubit_mapping.items()
    qubit_number = get_number_of_qubits(rhoAB)
    validate_qubit_mapping(qubit_mapping, qubit_number)

    expected_rangeA = list(range(1, int(qubitsA)+1))
    expected_rangeB = list(range(int(qubitsA)+1, int(qubit_number)+1))
    expected_tot = np.concatenate((expected_rangeA, expected_rangeB))
    assert np.array_equal(expected_tot,np.arange(1, qubit_number + 1)), ""
    "QubitsA must be less than QubitsB and must be continuous"
    
    
    B_dimension = len(qubitsB)
    num_params = 2 * (2**B_dimension - 1)
    
    I_A = np.eye(2**len(qubitsA))

    def objective(params):
        num_angles = 2**B_dimension - 1
        thetas = params[:num_angles]
        phis = params[num_angles:]
        base_operator_vectors = generate_orthonormal_vectors(B_dimension,
                                                             thetas, phis)
        value = 0
        
        for vector in base_operator_vectors:
            operator = np.outer(vector, np.conjugate(vector))
            kron_product = np.kron(I_A, operator)
            I_cross_Pi_x_rho = np.dot(kron_product, rhoAB)
            prob_x = np.trace(I_cross_Pi_x_rho).real
            if prob_x > 0:
                rho_x = partial_trace_multiple_qubits(I_cross_Pi_x_rho,
                                                      qubitsB) / prob_x
                value += prob_x * compute_entropy(rho_x)
                
        return value

    bounds = [(0, 2 * np.pi)] * num_params
    result = differential_evolution(objective, bounds)
    
    opt_term = result.fun
    #best_params = result.x
    #num_angles = 2**B_dimension - 1
    #best_thetas = best_params[:num_angles]
    #best_phis = best_params[num_angles:]
    return opt_term
    
    
    
def compute_J_second(rhoAB,qubit_mapping):
    """
    Assumes qubits are continuous, qubits from A, and B are
    not disjoint within each other. Assumes qubits A comes first due
    to operator generation with  tensor product

    Parameters
    ----------
    rhoAB : numpy array
        The density matrix of the combined system AB.
    qubit_mapping : dict
        A dictionary mapping system names to their respective qubits.

    Returns
    -------
    J : float
        The computed J value.
    best_thetas : numpy array
        The optimized theta values.
    best_phis : numpy array
        The optimized phi values.
    """
    
    
    (_, qubitsA), (_, qubitsB) = qubit_mapping.items()
    qubit_number = get_number_of_qubits(rhoAB)
    validate_qubit_mapping(qubit_mapping, qubit_number)

    
    B_dimension = len(qubitsB)
    num_params = 2 * (2**B_dimension - 1)
    I_A = np.eye(2**len(qubitsA))

    def objective(params):
        num_angles = 2**B_dimension - 1
        thetas = params[:num_angles]
        phis = params[num_angles:]
        base_operator_vectors = generate_orthonormal_vectors(B_dimension, thetas, phis)
        value = 0
        
        for vector in base_operator_vectors:
            operator = np.outer(vector, np.conjugate(vector))
            kron_product = np.kron(I_A, operator)
            I_cross_Pi_x_rho = np.dot(kron_product, rhoAB)
            prob_x = np.trace(I_cross_Pi_x_rho).real
            if prob_x > 0:
                rho_x = partial_trace_multiple_qubits(I_cross_Pi_x_rho, qubitsB) / prob_x
                value += prob_x * compute_entropy(rho_x)
                
        return value

    bounds = [(0, 2 * np.pi)] * num_params
    result = differential_evolution(objective, bounds,maxiter=100,disp=False,polish=True)
    opt_term = result.fun
    return opt_term

def compute_J(rhoAB, qubit_mapping):
    """
    Assumes qubits are continuous, qubits from A, and B are
    not disjoint within each other. Assumes qubits A comes first due
    to operator generation with tensor product

    Parameters
    ----------
    rhoAB : numpy array
        The density matrix of the combined system AB.
    qubit_mapping : dict
        A dictionary mapping system names to their respective qubits.

    Returns
    -------
    J : float
        The computed J value.
    best_thetas : numpy array
        The optimized theta values.
    best_phis : numpy array
        The optimized phi values.
    """
    
    
    (_, qubitsA), (_, qubitsB) = qubit_mapping.items()
    qubit_number = get_number_of_qubits(rhoAB)
    validate_qubit_mapping(qubit_mapping, qubit_number)
    S_A = compute_entropy(partial_trace_multiple_qubits(rhoAB, qubitsB))
    
    opt_term = compute_J_second(rhoAB, qubit_mapping)
    return S_A - opt_term


def compute_D(rhoAB, qubit_mapping):
    num_of_qubits = get_number_of_qubits(rhoAB)
    validate_qubit_mapping(qubit_mapping, num_of_qubits)
    mutual_info = compute_mutual_information(rhoAB, qubit_mapping)
    J = compute_J(rhoAB, qubit_mapping)
    return mutual_info - J

def compute_DW_MI(rhoAB, qubit_mapping):
    mut = compute_mutual_information(rhoAB, qubit_mapping)
    DW = compute_DW(rhoAB, qubit_mapping)
    return DW- mut/2

def compute_entanglement_of_purification(rhoAB, qubit_mapping):
    (_, qubitsA), (_, qubitsB) = qubit_mapping.items()
    qubit_number = int(np.log2(rhoAB.shape[0])) 
    eigenvalues, eigenvectors = np.linalg.eigh(rhoAB)
    num_params = 2**(qubit_number+1)-2  
    num_angles = num_params // 2
    
    def objective(params):
        thetas = params[:num_angles]
        phis = params[num_angles:]
        base_operator_vectors = generate_orthonormal_vectors(qubit_number,
                                                             thetas, phis)
        psi = np.zeros((2**(2*qubit_number)), dtype=complex)
        
        for i, ev in enumerate(eigenvalues):
            if ev.real > 0:
                psi += np.sqrt(ev) * np.kron(eigenvectors[:, i],
                                             base_operator_vectors[:,i])
        
        psi = psi.flatten()
        purified_state = np.outer(psi, np.conjugate(psi))
        
        rho_AA_star = partial_trace_multiple_qubits(purified_state,
                       np.concatenate((qubitsB, qubitsB + qubit_number)))
        
        entropy = compute_entropy(rho_AA_star)
        return entropy

    bounds = [(0, 2 * np.pi)] * (num_params) 
    result = differential_evolution(objective, bounds, maxiter=100,disp =False)
    
    minimum = result.fun
        
    return minimum




def compute_concurrence_exact(rho):
    assert rho.shape[0] == 4, "Rho must be two qubits"
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_y_tensor = np.kron(sigma_y, sigma_y)
    
    rho_tilde = sigma_y_tensor @ np.conjugate(rho) @ sigma_y_tensor
    R = rho @ rho_tilde
    
    eigenvalues = np.linalg.eigvals(R)
    eigenvalues = np.sqrt(np.maximum(eigenvalues, 0)).real
    eigenvalues = sorted(eigenvalues, reverse=True)
    
    C = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
    return C

def compute_binary_entropy(x):
    if x == 0 or x == 1:
        return 0
    return - x * np.log2(x) - (1 - x) * np.log2(1 - x)

def compute_entanglement_of_formation_2_qubits(rho):
    C = compute_concurrence_exact(rho)
    ef = compute_binary_entropy(0.5 * (1 + np.sqrt(1 - C**2)))
    return ef


def compute_tripartite_mutual_information(rhoABC,qubit_mapping):
    (_, qubitsA), (_, qubitsB),(_, qubitsC) = qubit_mapping.items()
    rhoAB = partial_trace_multiple_qubits(rhoABC, qubitsC)
    rhoAC = partial_trace_multiple_qubits(rhoABC, qubitsB)
    rhoBC = partial_trace_multiple_qubits(rhoABC, qubitsA)
    
    S_AB = compute_entropy(rhoAB)
    S_AC = compute_entropy(rhoAC)
    S_BC = compute_entropy(rhoBC)
    
    S_A = compute_entropy(partial_trace_multiple_qubits(rhoABC,
          np.concatenate((qubitsB,qubitsC))))
    S_B = compute_entropy(partial_trace_multiple_qubits(rhoABC, 
          np.concatenate((qubitsA,qubitsC))))
    S_C = compute_entropy(partial_trace_multiple_qubits(rhoABC, 
          np.concatenate((qubitsA,qubitsB))))
    
    S_ABC = compute_entropy(rhoABC)
    
    return S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
    

def validate_EP(rhoAB, qubit_mapping):
    #Unstable do not always trust!
      
    qubit_mapping = {
        "System A": np.array([1,2]),
        "System B": np.array([3])
    }
       
    
    (_, qubitsA), (_, qubitsB) = qubit_mapping.items()
    qubit_number = int(np.log2(rhoAB.shape[0])) 
    eigenvalues, eigenvectors = np.linalg.eigh(rhoAB)
    num_params = 2**(qubit_number+1)-2  
    num_angles = num_params // 2
    
    def objective(params):
        thetas = params[:num_angles]
        phis = params[num_angles:]
        base_operator_vectors = generate_orthonormal_vectors(qubit_number,
                                                             thetas, phis)
        psi = np.zeros((2**(2*qubit_number)), dtype=complex)
        
        for i, ev in enumerate(eigenvalues):
            if ev.real > 0:
                psi += np.sqrt(ev) * np.kron(eigenvectors[:, i],
                                             base_operator_vectors[:,i])
        
        psi = psi.flatten()
        purified_state = np.outer(psi, np.conjugate(psi))
        
        rho_AA_star = partial_trace_multiple_qubits(purified_state,
          np.concatenate((qubitsB, qubitsB + qubit_number)))
        
        entropy = compute_entropy(rho_AA_star)
        return entropy

    bounds = [(0, 2 * np.pi)] * (num_params) 
    result = differential_evolution(objective, bounds, maxiter=20)
    
    minimum = result.fun
        
    return minimum