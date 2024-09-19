# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:47:12 2024

Examples to show some of the code functionality and syntax

@author: herbi
"""

import numpy as np
from qutip import rand_dm
from EntropicQuantities import compute_DW

def demonstrate_compute_DW():
    rho = rand_dm(2**3).full() #create random 3 qubit density matrix
    
    #Meausres are commonly bipartite so instruct which qubits below to which
    #systems
    qubit_mapping = {"A": np.array([1]),"B":np.array([2,3])}
    
    DW = compute_DW(rho, qubit_mapping) #returns DW(A|B)
    print(DW)

demonstrate_compute_DW()