# (i) Importing packages
from types import SimpleNamespace
from scipy import optimize
import pandas as pd 
import numpy as np
import sympy as sm
from scipy.optimize import minimize_scalar



""" Defining the model """
class optimaltaxation():

    def __init__(self):
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

        # (iii) Government spendings within the interval (1.0, 2.0)
        par.G = np.linspace(1.0, 2.0, num=100)

        # (iv) labor
        par.L = sm.symbols('L')
        par.L_ss = sm.symbols('L^*')

        # Supporting symbols
        par.V = sm.symbols('V')

    """ calculate utility """
    def utility(self):

        par = self.par
        sol = self.sol

        # (i) defining wbar
        w_bar = (1 - par.tau) * par.w

        # (ii) private consumption of market goods
        C = par.kappa + w_bar * par.L

        # (iii) utility 
        lign1 = sm.Eq(sm.log(C ** par.alpha * par.G ** (1 - par.alpha)) - par.v * par.L ** 2 / 2,par.V)

        # (iv) Isolation L qua 'solver
        u_L = sm.solve(lign1, par.L_ss)

        # (iv) return
        return u_L



    """ optimal labor """
    def opti_labor(self, do_print=False):

        par = self.par
        #sol = self.sol
        opt = SimpleNamespace()

        # (i) Defining objective function
        def objective(L):
            return -self.utility()
        obj = objective(par.L)

        # (ii) Defning the guess
        guess = [7]

        # (iii) Defining bounds
        bounds = (0, 25)

        # (iv) Applying the minimize function      
        res = optimize.minimize(obj, guess, method='Nelder-Mead', bounds=bounds)

        # (v) sating the result
        opt.Labor = res.x

        # (iv) return
        return opt













