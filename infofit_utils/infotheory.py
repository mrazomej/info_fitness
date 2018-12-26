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


# BLAHUT-ARIMOTO ALGORITHM
def channel_capacity(QmC, epsilon=1E-3, info=1E4):
    '''
    Performs the Blahut-Arimoto algorithm to compute the channel capacity
    given a channel QmC.

    Parameters
    ----------
    QmC : array-like
        definition of the channel with C inputs and m outputs.
    epsilon : float.
        error tolerance for the algorithm to stop the iterations. The smaller
        epsilon is the more precise the rate-distortion function is, but also
        the larger the number of iterations the algorithm must perform
    info : int.
        Number indicating every how many cycles to print the cycle number as
        a visual output of the algorithm.
    Returns
    -------
    C : float.
        channel capacity, or the maximum information it can be transmitted
        given the input-output function.
    pc : array-like.
        array containing the discrete probability distribution for the input
        that maximizes the channel capacity
    '''
    # initialize the probability for the input.
    pC = np.repeat(1 / QmC.shape[0], QmC.shape[0])

    # Initialize variable that will serve as termination criteria
    Iu_Il = 1

    loop_count = 0
    # Perform a while loop until the stopping criteria is reached
    while Iu_Il > epsilon:
        if (loop_count % info == 0) & (loop_count != 0):
            print('loop : {0:d}, Iu - Il : {1:f}'.format(loop_count, Iu_Il))
        loop_count += 1
        # compute the relevant quantities. check the notes on the algorithm
        # for the interpretation of these quantities
        # cC = exp(∑_m Qm|C log(Qm|C / ∑_c pC Qm|C))
        sum_C_pC_QmC = np.sum((pC * QmC.T).T, axis=0)
        # Compute QmC * np.log(QmC / sum_C_pC_QmC) avoiding errors with 0 and
        # neg numbers
        with np.errstate(divide='ignore', invalid='ignore'):
            QmC_log_QmC_sum_C_pC_QmC = QmC * np.log(QmC / sum_C_pC_QmC)
        # check for values that go to -inf because of 0xlog0
        QmC_log_QmC_sum_C_pC_QmC[np.isnan(QmC_log_QmC_sum_C_pC_QmC)] = 0
        QmC_log_QmC_sum_C_pC_QmC[np.isneginf(QmC_log_QmC_sum_C_pC_QmC)] = 0
        cC = np.exp(np.sum(QmC_log_QmC_sum_C_pC_QmC, axis=1))

        # I_L log(∑_C pC cC)
        Il = np.log(np.sum(pC * cC))

        # I_U = log(max_C cC)
        Iu = np.log(cC.max())

        # pC = pC * cC / ∑_C pC * cC
        pC = pC * cC / np.sum(pC * cC)

        Iu_Il = Iu - Il

    # convert from nats to bits
    Il = Il / np.log(2)
    return Il, pC, loop_count
