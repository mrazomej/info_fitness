# -*- coding: utf-8 -*-
"""
Title:
    model.py
Last update:
    2018-12-21
Author(s):
    Manuel Razo-Mejia
Purpose:
    This file compiles all of the functions related to the
    theoretical model for transcriptional regulation and 
    its connection to fitness relevant for the information 
    and fitness project.
    
"""
# Our numerical workhorses
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.special
import scipy.integrate
import mpmath
import pandas as pd


# THERMODYNAMIC FUNCTIONS
def p_act(C, ka, ki, epsilon=4.5, logC=False):
    '''
    Returns the probability of a lac repressor being in the active state, i.e.
    able to bind the promoter as a function of the ligand concentration.

    Parameters
    ----------
    C : array-like.
        concentration(s) of ligand at which evaluate the function.
    ka, ki : float.
        dissociation constants for the active and inactive states respectively
        in the MWC model of the lac repressor.
    epsilon : float.
        energetic barrier between the inactive and the active state.
    logC : Bool.
        boolean indicating if the concentration is given in log scale

    Returns
    -------
    p_act : float.
        The probability of the repressor being in the active state.
    '''
    C = np.array(C)
    if logC:
        C = 10**C

    return (1 + C / ka)**2 / \
        ((1 + C / ka)**2 + np.exp(-epsilon) * (1 + C / ki)**2)


def fold_change_statmech(C, R, eRA, ka, ki, Nns=4.6E6, epsilon=4.5,
                         logC=False):
    '''
    Computes the gene expression fold-change as expressed in the simple
    repression thermodynamic model of gene expression as a function of
    repressor copy number, repressor-DNA binding energy, and MWC parameters.

    Parameters
    ----------
    C : array-like.
        concentration(s) of ligand at which evaluate the function.
    R : array-like.
        repressor copy number per cell
    eRA : array-like.
        repressor-DNA binding energy
    ka, ki : float.
        dissociation constants for the active and inactive states respectively
        in the MWC model of the lac repressor.
    Nns : float. Default = 4.6E6
        number of non-specific binding sites in the bacterial genome.
    epsilon : float.
        energetic barrier between the inactive and the active state.
    logC : Bool.
        boolean indicating if the concentration is given in log scale

    Returns
    -------
    p_act : float.
        The probability of the repressor being in the active state.
    '''
    return (1 + R / Nns * p_act(C, ka, ki, epsilon, logC) * np.exp(-eRA))**-1


# DISTRIBUTION MOMENT DYNAMICS
def dmomdt(A_mat, expo, t, mom_init, states=['E', 'P', 'R']):
    '''
    Function to integrate 
    dµ/dt = Aµ
    for any matrix A using the scipy.integrate.odeint
    function
    
    Parameters
    ----------
    A_mat : 2D-array
        Square matrix defining the moment dynamics
    expo : array-like
        List containing the moments involved in the 
        dynamics defined by A
    t : array-like
        Time array in seconds
    mom_init : array-like. lenth = A_mat.shape[1]
    states : list with strings. Default = ['E', 'P', 'R']
        List containing the name of the promoter states
    Returns
    -------
    Tidy dataframe containing the moment dynamics
    '''
    # Define a lambda function to feed to odeint that returns
    # the right-hand side of the moment dynamics
    def dt(mom, time):
        return np.dot(A_mat, mom)
    
    # Integrate dynamics
    mom_dynamics = sp.integrate.odeint(dt, mom_init, t)

    ## Save results in tidy dataframe  ##
    # Define names of columns
    names = ['m{0:d}p{1:d}'.format(*x) + s for x in expo 
             for s in states]

    # Save as data frame
    df = pd.DataFrame(mom_dynamics, columns=names)
    # Add time column
    df = df.assign(t_sec = t, t_min = t / 60)
    
    return df

# DEKEL & ALON 2005 FITNESS LANDSCAPE

def cost_func(p_rel, eta_o=0.02, M=1.8):
    '''
    Returns the relative growth rate cost of producing LacZ protein according to
    Dekel and Alon's model:
        eta_2 = eta_o * p_rel / (1 - p_rel / M)
    
    Parameter
    ---------
    p_rel : array-like.
        Relative expression with respect to the wild type expression when
        fully induced with IPTG
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function
    
    Returns
    -------
    eta_2 : array-like
        relative reduction in growth rate with respect to wild type when not
        expressing the enzyme.
    '''
    p_rel = np.array(p_rel)
    return eta_o * p_rel / (1 - p_rel / M)

def benefit_func(p_rel, C_array, delta=0.17, Ks=0.4):
    '''
    Returns the relative growth rate benefit of producing LacZ protein 
    according to Dekel and Alon's model:
        r = delta * p_rel * C / (Ks + C)
    
    Parameter
    ---------
    p_rel : array-like.
        Relative expression with respect to the wild type expression when
        fully induced with IPTG.
    C_array : array-like.
        Substrate concentration.
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    
    Returns
    -------
    r : array-like
        relative increase in growth rate with respect to wild type when not
        expressing the enzyme.
    '''
    p_rel = np.array(p_rel)
    return delta * p_rel * C_array / (Ks + C_array)

def fitness(p_rel, C_array, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8,
           logC=False):
    '''
    Returns the relative fitness according to Dekel and Alon's model.
    
    Parameter
    ---------
    p_rel : array-like.
        Relative expression with respect to the wild type expression when
        fully induced with IPTG.
    C_array : array-like.
        Substrate concentration. If logC==True this is defined as log10(C)
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function
    logC : Bool.
        boolean indicating if the concentration is given in log scale
    
    Returns
    -------
    fitness : array-like
        relative fitness with respect to wild type when not
        expressing the enzyme.
    '''
    p_rel = np.array(p_rel)
    C_array = np.array(C_array)
    if logC:
        C_array = 10**C_array
    # Compute benefit - cost
    return benefit_func(p_rel, C_array, delta, Ks) - cost_func(p_rel, eta_o, M)

def p_opt(C_array, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8, logC=False):
    '''
    Returns the optimal protein expression level p* as a function of
    substrate concentration.
    
    Parameters
    ----------
    C_array : array-like.
        Substrate concentration.
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function
    logC : Bool.
        boolean indicating if the concentration is given in log scale
        
    Returns
    -------
    p* the optimal expression level for a given concentration.
    '''
    C_array = np.array(C_array)
    if logC:
        C_array = 10**C_array
    
    # Dekel and Alon specify that concentrations lower than a lower 
    # threshold should be zero. Then let's build that array
    thresh = Ks * (delta / eta_o - 1)**-1
    
    popt = np.zeros_like(C_array)
    popt[C_array > thresh] = M * (1 - np.sqrt(eta_o / delta * \
        (C_array[C_array > thresh] + Ks) / C_array[C_array > thresh]))
    
    return popt

def abs_fitness(p_rel, C_array, delta=0.17, 
                Ks=0.4, eta_o=0.02, M=1.8,
                mu=1.15, logC=False):
    '''
    Returns the absolute fitness according to Dekel and Alon's model.
    
    Parameter
    ---------
    p_rel : array-like.
        Relative expression with respect to the wild type expression when
        fully induced with IPTG.
    C_array : array-like.
        Substrate concentration. If logC==True this is defined as log10(C)
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function
    mu : float.
        Absolute growth rate of reference strain
    logC : Bool.
        boolean indicating if the concentration is given in log scale
    
    Returns
    -------
    fitnes : array-like
        absolute fitness of strain.
    '''
    p_rel = np.array(p_rel)
    C_array = np.array(C_array)
    if logC:
        C_array = 10**C_array
    # Compute benefit - cost
    rel_fit = benefit_func(p_rel, C_array, delta, Ks) -\
              cost_func(p_rel, eta_o, M)
    
    return mu * (1 + rel_fit)

def abs_fitness_opt(c, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8, mu=1.15):
    '''
    Computes the absolute fitness when evaluated at the optimal relative 
    expression level p* for a given input concetration c, according to
    Dekel and Alon's 2005 fitness landscape.
    Parameters
    ----------
    c : array-like.
        Substrate concentration. If logC==True this is defined as log10(C)
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function
    mu : float.
        Absolute growth rate of reference strain
    
    Returns
    -------
    popt : array-like.
        The optimal relative expression level for a given input c.
    '''
    # Compute optimal expression level
    popt = p_opt(c, delta, Ks, eta_o, M)
    
    # Compute optimal growth rate
    return abs_fitness(popt, c, delta, Ks, eta_o, M, mu)

def growth_diff_c(p_rel, c, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8,
                  mu=1.15):
    '''
    Returns the difference between the growth rate r(C, m) and the optimal
    growth rate r_max(C).

    Parameter
    ---------
    Parameter
    ---------
    p_rel : array-like.
        Relative expression with respect to the wild type expression when
        fully induced with IPTG.
    C_array : array-like.
        Substrate concentration. If logC==True this is defined as log10(C)
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function
    mu : float.
        Absolute growth rate of reference strain

    Returns
    -------
    growth_diff : array-like.
        difference between relative growth rate and the maximum 
        relative growth rate.
    ''' 
    # Compute the optimal expression level for each input
    rc_max = abs_fitness_opt(c, delta, Ks, eta_o, M)
    
    # Compute the non-optimal expression levels
    rcp = abs_fitness(p_rel, c, delta, Ks, eta_o, M, mu)
    
    # Return the difference between these two growth rates
    return rc_max - rcp
