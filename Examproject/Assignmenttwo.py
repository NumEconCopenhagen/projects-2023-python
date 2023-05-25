# (i) Importing packages
import numpy as np
from scipy import optimize
import sympy as sm
import matplotlib.pyplot as plt
from types import SimpleNamespace




""" Defining the model """
class hairdresser:
        
    def __init__(self, **kwargs):

        # (i) create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # (ii) defining parameters
        par.eta = 0.5                                  # elasticity of demand
        par.w = 1.0                                    # wage
        par.kappa_t = np.linspace(1.0, 2.0)            # demand shock
    
        # (iii) solution
        sol.l_t = np.nan



    """ Profit """
    def negativeprofit(self, l_t, kappa_t):
        par = self.par
        sol = self.par

        # (ii) production
        y_t = l_t

        # (iii) price per haircut
        p_t = kappa_t * y_t ** (- par.eta)

        # (iv) profit
        pi_t = p_t * y_t - par.w * l_t 

        return -pi_t


    def optimal_l_t(self):
        par =self.par
        optimal_l_t_values = []
        for kappa_t_value in par.kappa_t:
            l_t = ((1 - par.eta) * kappa_t_value / par.w) ** (1 / par.eta)
            optimal_l_t_values.append(l_t)
        return optimal_l_t_values
    




    """ Defining the maximization problem """
    def opti_employees(self):

        par = self.par
        sol = self.sol

        sols = dict()
        for kappa_t in par.kappa_t:
            l_guess = 0
            tmp_util = lambda l_t: self.negativeprofit(l_t,kappa_t=kappa_t)
            sols[str(kappa_t)] = optimize.minimize(tmp_util, l_guess, bounds =((0,None),), method='Nelder-Mead')['x'][0]

        return sols
    


class dynamic_hairdresser:

        
    def __init__(self, **kwargs):

        # (i) create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # (ii) defining parameters
        par.eta = 0.5                                  # elasticity of demand
        par.w = 1.0                                    # wage
        #par.kappa_t = np.linspace(1.0, 2.0)            # demand shock
        par.rho = 0.9
        par.iota = 0.01
        par.sigma_e = 0.1
        par.R = (1+0.01)**(1/12)

        self.T = 120

    def kappa_t(self):
        # evt sæt random seed
        
        kappas = np.array([])
        par = self.par
        k_t = 1
        eps = 0
        for t in range(self.T):
            k_t = np.exp(par.rho * np.log(k_t) + eps)
            #eps = random generated from normal distrubtion
            kappas = np.append(kappas, k_t) 
        return kappas

    def h(self):
        par = self.par
        kappas = self.kappa_t()
        sum_ = 0
        l_t = 0
        for i in range(self.T):
            new_l_t = self.l_t() # definer denne funktion
            if new_l_t != l_t:
                sum_ = sum_ + par.R ** (-t) * (kappas[i]*l_t**(1-par.eta) - par.w*l_t - par.iota)
            else:
                sum_ = sum_ + par.R ** (-t) * (kappas[i]*l_t**(1-par.eta) - par.w*l_t)