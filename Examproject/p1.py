# importing relevant packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import math
from tabulate import tabulate
import sympy as sm
from types import SimpleNamespace


""" Defining the model (question 1-4)'"""
class TaxGov:

    """ The initial parameters"""
    def __init__(self):

        par = self.par = SimpleNamespace()

        # (i) Defining the given parameters
        par.alpha = 0.5                                     # share that goes to capital
        par.kappa = 1                                       # free private consumption
        par.nu = 1/(2*16**2)                                # disutility of labor
        par.w = 1                                           # real wage
        par.tau = 0.3                                       # labor-income tax

        # (ii) Defining the equations
        par.C = lambda L: par.kappa+(1-par.tau)*par.w*L     # Private consumption (a function of L)
        par.G3 = lambda tau, Ls: tau * par.w * Ls           # Government consumption (used in question 3)
        par.G = 1 #np.linspace(1,2, num=10)                 # Government consumption (question 1-2)



    """ Defining L* to question 1 for verification purposes """
    def L_star(self,wtilde):

        par = self.par

        # (i) Returning the equation the L* with the given parameters
        return (-par.kappa+np.sqrt(par.kappa**2+4*(par.alpha/par.nu)*wtilde**2))/(2*wtilde)
    


    """ Defining w_tilde """
    def wtilde(self, w = None, tau = None): # for G in 1 to 2

        par = self.par

        # (i) If no argument is given then w = 1 is used (default value) 
        if w == None:
            w = par.w

        # (ii) If no argument is given then tau = 0.3 is used (default value) 
        if tau == None:
            tau = par.tau

        # (iii) Return
        return (1-tau)*w

    # def V(self, L):
    #     par = self.par
    #     return -(np.log(par.C(L) ** par.alpha * par.G ** (1 - par.alpha)) - par.nu * L ** 2 / 2)



    """ Optimized labour supply"""
    def optimize1(self):

        par = self.par

        # (i) Guess on L
        L_guess = 0.0

        # (ii) Defining the maximization function:
        V = lambda L: -(np.log(par.C(L) ** par.alpha * par.G ** (1 - par.alpha)) - par.nu * L ** 2 / 2)

        # (iii) Running an optimize code to find the optimal L*
        sol = optimize.minimize(V, L_guess, bounds = ((0,25),), method = 'L-BFGS-B')
        
        # (iv) Return
        return sol
    

    """ Optimized labour supply based on the tax rate (question 4)"""
    def optimize2(self):

        par = self.par

        # (i) Guess on tau
        tau_guess = 0.1

        # (ii) Defining consumption when tau varies
        C_tau = lambda tau, L: par.kappa + (1- tau)*par.w*L

        # (iii) Defining V as a function of C_tau
        V = lambda tau: -(np.log(C_tau(tau, self.L_star(self.wtilde(tau=tau))) ** par.alpha * par.G3(tau, self.L_star(self.wtilde(tau=tau))) ** (1 - par.alpha)) - par.nu * self.L_star(self.wtilde(tau=tau)) ** 2 / 2)
        
        # (iv) Running an optimization code
        sol2 = optimize.minimize(V, tau_guess, bounds = ((0,1),), method = 'Nelder-Mead')
        
        # (i) Return
        return sol2




    """ Defining the socially optimal function """
    def V_tau(self, tau):

        par = self.par
        
        C_tau = lambda tau, L: par.kappa + (1- tau)*par.w*L

        V = lambda tau: -(np.log(C_tau(tau, self.L_star(self.wtilde(tau=tau))) ** par.alpha * par.G3(tau, self.L_star(self.wtilde(tau=tau))) ** (1 - par.alpha)) - par.nu * self.L_star(self.wtilde(tau=tau)) ** 2 / 2)

        # (i) Returning the equation tau
        return V(tau)





""" Defining a model with symbols (question 5-6) """
class TaxGov_sym:


    """ The initial parameters"""
    def __init__(self):

        # (i) create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # (ii) defining the parameters
        par.alpha = sm.symbols('alpha')             
        par.kappa = sm.symbols('kappa')             # free private consumption component
        par.v = sm.symbols('v')                     # disutility of labor scaling
        par.w = sm.symbols('w')                     # labor-income
        par.tau = sm.symbols('tau')                 # labor-income tax
        par.C = sm.symbols('C')                     # consumption
        par.L = sm.symbols('L')                     # Labor
        par.L_star = sm.symbols('L^*')              # Optimal labor
        par.sigma = sm.symbols('sigma')
        par.rho = sm.symbols('rho')
        par.epsilon = sm.symbols('epsilon')



    """ Defining consumption """
    def consumption(self, L):
        
        par = self.par

        # (i) Defining consumption
        Con = sm.Eq(par.C, par.kappa + (1 - par.tau) * par.w * par.L)

        # (ii) Defining government consumption
        Gov_con = sm.Eq(par.G, par.tau * par.w * par.L_star * (1 - par.tau) * par.w)

        # (iii) Defining max prob
        V = sm.Eq(((par.alpha * par.C ** ((par.sigma -1)/ par.sigma) + (1 - par.alpha) * par.G ** (par.sigma / (par.sigma - 1)) ** (1 - par.rho) - 1) / (1 - par.rho)) - par.v * par.L ** (1 + par.epsilon) / (1 + par.epsilon))

        # (iv) Substitution Consumption into V 
        V_c = V.subs(par.C, Con[0])
        
        # (v) Substitution Government consumption into V
        V_c_g = V_c.subs(par.G, Gov_con[0])

        # (vi) FOC w.r.t. L
        FOC_V = sm.diff(V_c_g, L)

        # (vii) Take the derivative
        sol3 = sm.solve(sm.Eq(FOC_V, 0), L)

        # (iix) return
        return sol3
    


    """ Defining socially optimal tax rate (question 6) """
    def optimize3(self, tau):
        
        par = self.par

        # (iii) Defining V as a function of tau
        V = lambda tau: (((par.alpha * par.C ** ((par.sigma -1)/ par.sigma) + (1 - par.alpha) * par.G ** (par.sigma / (par.sigma - 1)) ** (1 - par.rho) - 1) / (1 - par.rho)) - par.v * par.L ** (1 + par.epsilon) / (1 + par.epsilon))

        # (iv) Running an optimization code
        sol3 = optimize.minimize(V, tau_guess, bounds = ((0,1),), method = 'Nelder-Mead')
        
        # (i) Return
        return sol3

