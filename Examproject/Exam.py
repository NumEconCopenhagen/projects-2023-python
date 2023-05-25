# Importing packages
from scipy import optimize
import sympy as sm
import numpy as np
from types import SimpleNamespace

class assignmen_one():

    """Creating the model """
    def __init__(self):
        
        self.par = SimpleNamespace()
        self.setup()


    """ Defining the parameters and variables """
    def setup(self):

        par = self.par

        # (i) worker's utility parameters
        par.C = sm.symbols('C')          # Private consumption 
        par.kappa = sm.symbols('kappa')       # free private consumption conponent
        par.w = sm.symbols('w')          # real wage
        par.tau = sm.symbols('tau')         # labor-income tax rate
        par.G = sm.symbols('G')         # Government consumption
        par.v = sm.symbols('ny')        # disutility of labor scaling factor
        par.L = sm.symbols('L')         # Labor

    
    
    """ Defining the utility function for the worker """
    def utility(self):
        par = self.par

        # (i) Consumer's utility function
        return sm.log(par.C ** par.alpha * par.G ** (1 - par.alpha)) - par.v * (par.L ** 2 / 2)



    """ Defining the private consumption """
    def consumption(self):
        par = self.par

        # (i) private consumption
        consumption = sm.Eq(C, par.kappa + (1 - par.tau) * par.w * par.L)
