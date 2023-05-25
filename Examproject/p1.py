# importing relevant packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import math
from tabulate import tabulate
import sympy as sm
from types import SimpleNamespace

class TaxGov:

    def __init__(self):

        par = self.par = SimpleNamespace()

        par.alpha = 0.5
        par.kappa = 1
        par.nu = 1/(2*16**2)
        par.w = 1
        par.tau = 0.3
        par.C = lambda L: par.kappa+(1-par.tau)*par.w*L # Function of L
        par.G3 = lambda tau, Ls: tau * par.w * Ls


        # grid
        par.G = 1 #np.linspace(1,2, num=10)

    def L_star(self,wtilde): # Mest til verificering
        par = self.par
        return (-par.kappa+np.sqrt(par.kappa**2+4*(par.alpha/par.nu)*wtilde**2))/(2*wtilde)
    
    def wtilde(self, w = None, tau = None): # for G in 1 to 2
        par = self.par
        # if no argument is given then default value of w is used.
        if w == None:
            w = par.w
        if tau == None:
            tau = par.tau
        return (1-tau)*w

    # def V(self, L):
    #     par = self.par
    #     return -(np.log(par.C(L) ** par.alpha * par.G ** (1 - par.alpha)) - par.nu * L ** 2 / 2)

    def optimize1(self):

        par = self.par
        L_guess = 0.0
        V = lambda L: -(np.log(par.C(L) ** par.alpha * par.G ** (1 - par.alpha)) - par.nu * L ** 2 / 2)
        sol = optimize.minimize(V, L_guess, bounds = ((0,25),), method = 'L-BFGS-B')
        return sol
    

    def optimize2(self):

        par = self.par
        tau_guess = 0.1
        C_tau = lambda tau, L: par.kappa + (1- tau)*par.w*L
        V = lambda tau: -(np.log(C_tau(tau, self.L_star(self.wtilde(tau=tau))) ** par.alpha * par.G3(tau, self.L_star(self.wtilde(tau=tau))) ** (1 - par.alpha)) - par.nu * self.L_star(self.wtilde(tau=tau)) ** 2 / 2)
        sol2 = optimize.minimize(V, tau_guess, bounds = ((0.1,0.99999999999),), method = 'Nelder-Mead')
        return sol2





class TaxGov_sym:

    def __init__(self, sigma, rho, epsilon):
        """ setup model """

        # (i) create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # (ii) preferences
        par.alpha = sm.symbols('alpha')             # 
        par.kappa = sm.symbols('kappa')               # free private consumption component
        par.v = sm.symbols('v')   # disutility of labor scaling
        par.w = sm.symbols('w')                 # labor-income
        par.tau = sm.symbols('tau')               # labor-income tax
        par.C = sm.symbols('C')
        par.L = sm.symbols('L')


    def V(self):
        
        par = self.par

        consumption = sm.Eq(par.C, par.kappa +(1-par.tau)*par.w*par.L)

