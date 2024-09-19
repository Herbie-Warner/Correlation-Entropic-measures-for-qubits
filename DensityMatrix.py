# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:17:49 2024

Example stuff for graphs

@author: herbi
"""

import numpy as np
from qutip import rand_dm
from Utilities import canonical_purify, display_qubit_mapping, compute_entropy, partial_trace, translate_qubit_map, partial_trace_multiple_qubits
from EntropicQuantities import compute_mutual_information, compute_DW, compute_reflected_entropy
from EntropicQuantities import compute_J, compute_D, compute_DW_MI
from LocalOperators import generate_orthonormal_vectors
import matplotlib.pyplot as plt
from EntropicQuantities import compute_entanglement_of_purification



def generate_qubit_mappings(num_qubits):
    mappings = []
    for i in range(1, num_qubits):
        system_A = np.arange(1, i + 1)
        system_B = np.arange(i + 1, num_qubits + 1)
        mappings.append({"System A": system_A, "System B": system_B})
    return mappings

def plot_d_v_dw():
    num_experiments = 5
    max_num_qubits = 5
    
    dimensions = []
    diff = []
    labels = []
    
    for numQubits in range(2, max_num_qubits + 1):
        qubit_mappings = generate_qubit_mappings(numQubits)
        for qubit_mapping in qubit_mappings:
            dimensions.append(numQubits)
            label = f"Qubits: {numQubits}, A: {list(qubit_mapping['System A'])}, B: {list(qubit_mapping['System B'])}"
            labels.append(label)        
            Diff = []
            for i in range(num_experiments): 
                rho = rand_dm(2**numQubits).full()

                D = compute_D(rho, qubit_mapping)
                DW = compute_DW(rho, qubit_mapping)
                Diff.append(DW-D)
            
            diff.append(np.average(Diff))
    
    # Plotting the histogram
    figure = plt.figure()
    ax = figure.add_subplot()
    
    ax.bar(labels, diff)
    ax.set_xlabel('Qubit Mapping and Dimensions')
    ax.set_ylabel('Average Difference (DW - D)')
    ax.set_title('Histogram of Differences for Various Qubit Mappings and Dimensions')
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    plt.show()
    
    
def plot_some_mapping(ax,qubit_mapping,name,iterations,numQubits):
    D_current = []
    DW_current = []
    print(name)
    for i in range(iterations):   
        rho = rand_dm(2**numQubits).full()
        D = compute_D(rho, qubit_mapping)
        D_current.append(D)
        DW = compute_DW(rho, qubit_mapping)
        DW_current.append(DW)
        
    ax.scatter(D_current,DW_current,label=name)
        
        
    


   
def plotting_diff():
 

    
    
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": "DejaVu Sans",
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
    })
    
    figure = plt.figure()
    ax = figure.add_subplot()
    

    size = 15
    ax.set_xlabel(r'$D$',fontsize=size)
    ax.set_ylabel(r'$D_R$',fontsize=size)
    
    
    iterations = 5
    
    qubit_mapping = {
        "System A": np.array([1]),
        "System B": np.array([2])
    }
    plot_some_mapping(ax, qubit_mapping, "1:1",25, 2)
    
    """
    qubit_mapping = {
        "System A": np.array([1,2]),
        "System B": np.array([3])
    }
    plot_some_mapping(ax, qubit_mapping, "2:1",25, 3)
    
        
    qubit_mapping = {
        "System A": np.array([1,2]),
        "System B": np.array([3,4])
    }
    plot_some_mapping(ax, qubit_mapping, "2:2",20, 4)
    
    qubit_mapping = {
        "System A": np.array([1,2,3]),
        "System B": np.array([4])
    }
    plot_some_mapping(ax, qubit_mapping, "3:1",20, 4)
    
    qubit_mapping = {
        "System A": np.array([1,2,3,4]),
        "System B": np.array([5])
    }
    plot_some_mapping(ax, qubit_mapping, "4:1",20, 5)
    
    qubit_mapping = {
        "System A": np.array([1,2,3]),
        "System B": np.array([4,5])
    }
    plot_some_mapping(ax, qubit_mapping, "3:2",20, 5)
    
    qubit_mapping = {
        "System A": np.array([1,2]),
        "System B": np.array([3,4,5])
    }
    plot_some_mapping(ax, qubit_mapping, "2:3",20, 5)
    
    qubit_mapping = {
        "System A": np.array([1,2,3,4]),
        "System B": np.array([5,6])
    }
    plot_some_mapping(ax, qubit_mapping, "4:2",20, 6)
    
       
    qubit_mapping = {
        "System A": np.array([1,2,3]),
        "System B": np.array([4,5,6])
    }
    plot_some_mapping(ax, qubit_mapping, "3:3",20, 6)
    """
    
    
    
    
    
    max_D = 0.75
    x = np.linspace(0, max_D, 100)
    ax.plot(x, x, label=r'$D_R = D$', linestyle='--',color='k')
    #figure.patch.set_facecolor('none')


    #ax.set_facecolor('none')

        
    ax.legend()
    plt.savefig("monotonicity_upper_bound.png",dpi=1200)
    plt.show()
    

def create_w_state(n):

    dim = 2 ** n

    w_state = np.zeros(dim, dtype=complex)

    for i in range(n):
        basis_state = 1 << (n - i - 1) 
        w_state[basis_state] = 1.0 / np.sqrt(n) 
    
    return w_state
    
def D_monogamy():
    """
    σ(A : B) + σ(A : C) ≤ σ(A : BC)
    
    """
    
    
    w = create_w_state(3)
    rhoABC = np.outer(np.conjugate(w).T,w)
    
    
    qubit_mapping = {"A":np.array([1]),"B":np.array([2])}

    D_A_B = compute_D(partial_trace_multiple_qubits(rhoABC, [3]), qubit_mapping)
    D_A_C = compute_D(partial_trace_multiple_qubits(rhoABC, [2]), qubit_mapping)
    
    qubit_mapping = {"A":np.array([1]),"B":np.array([2,3])}
    
    D_A_BC = compute_D(rhoABC, qubit_mapping)
    
    print(D_A_BC - D_A_B-D_A_C)

if __name__ == "__main__":   
    D_monogamy()

    
    


    
   