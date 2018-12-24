# -*- coding: utf-8 -*-
"""
Title:
    ratedist.py
Last update:
    2018-12-25
Author(s):
    Manuel Razo-Mejia
Purpose:
    This file compiles all of the functions necessary to compute
    the rate distortion function.
"""

import numpy as np

def rate_dist(Pc, rho_cp, beta, epsilon=1E-3):
    '''
    Performs the Blahut algorithm to compute the rate-distortion function
    R(D) given an input distribution Pc and a distortion matrix rho_cp.
    Parameters
    ----------
    Pc : array-like.
        Array containing the distribution of inputs
    rho_cp : 2D array.
        Matrix containing the distortion function for a given 
        input c and an output p. 
        NOTE: For the biological example this is defined as the absolute
        difference between the growth with optimal expression level and
        any other growth rate with optimal expression level and
        any other growth rate.
     beta : float. [-inf, 0]
        slope of the line with constant I(Q) - beta * sD. This parameter 
        emerges during the unconstraint optimization as a Lagrange 
        multiplier. It plays the analogous role of the inverse 
        temperature in the Boltzmann distribution.   
    epsilon : float.
        error tolerance for the algorithm to stop the iterations. The smaller
        epsilon is the more precise the rate-distortion function is, but also
        the larger the number of iterations the algorithm must perform
        
    Returns
    -------
    qp : array.
        marginal gene expression probability distribution.
    qp|C : 2D array.
        The input-output transition matrix for each of the inputs and outputs
        given in C and m respectively.
    r : float.
        Average performance.
    R : float.
        minimum amount of mutal information I(c;p) consistent 
        with performance.
    '''
    # Initialize the proposed output distribution as a uniform 
    # distribution
    qp0 = np.repeat(1 / rho_cp.shape[0], rho_cp.shape[0])
    
    # This will be the probabilities that will be updated on each cycle
    qp = qp0
    
    # Compute the cost matrix
    A_cp = np.exp(beta * rho_cp.T)

    # Initialize variable that will serve as termination criteria
    Tu_Tl = 1
    
    # Initialize loop counter
    loop_count = 0
    
    # Perform a while loop until the stopping criteria is reached
    while Tu_Tl > epsilon:
        # compute the relevant quantities. check the notes on the algorithm
        # for the interpretation of these quantities
        # ∑_p qp A_cp
        sum_p_qp_A_cp = np.sum(qp * A_cp, axis=1)

        # cp = ∑_C Pc A_cp / ∑_p qp A_cp
        cp = np.sum((Pc * A_cp.T / sum_p_qp_A_cp).T, axis=0) #+ 1E-10

        # qp = qp * cp
        qp = qp * cp

        # Tu = ∑_p qp log cp
        Tu = - np.sum(qp * np.log(cp))

        # Tl = max_p log cp
        Tl = - np.log(cp).max()
        
        # Tu - Tl
        Tu_Tl = Tu - Tl
        
        # increase the loop count
        loop_count += 1
    
    # Compute the outputs after the loop is finished.
    
    # ∑_p qp A_cp
    sum_p_qp_A_cp = np.sum(qp * A_cp, axis=1)
    
    # qp|C = A_cp qp / ∑_p A_cp qp
    qpC = ((qp * A_cp).T / sum_p_qp_A_cp).T
    
    # D = ∑_c Pc ∑_p qp|C rho_cp
    D = np.sum(Pc * np.sum(qpC * rho_cp.T, axis=1).T)
    
    # R(D) = beta D - ∑_C Pc log ∑_p A_cp qp - ∑_p qp log cp
    RD = beta * D \
    - np.sum(Pc * np.log(sum_p_qp_A_cp)) \
    - np.sum(qp * np.log(cp))

    # convert from nats to bits
    RD = RD / np.log(2)
    
    return qp, qpC, D, RD

