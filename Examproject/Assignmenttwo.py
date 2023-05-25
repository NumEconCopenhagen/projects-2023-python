# (i) Importing packages
from types import SimpleNamespace
from scipy import optimize
import pandas as pd 
import numpy as np
import sympy as sm
from scipy.optimize import minimize_scalar



""" Defining the model """
class hairdresser():
        
    def __init__(self, **kwargs):

        self.setup_initial()
        self.setup_updated(kwargs)
        self.hairdresserfunc()


    """ Defining the parameters and variables """
    def setup_initial(self):

        # (i) Model
        self.eta = 0.5                                  # elasticity of demand
        self.w = 1.0
        self.kappa_t = np.linspace(1.0, 2.0, num=100)   # demand shock

        # (ii) Transition
        self.kt_min = 1
        self.kt_max = 120       # 120 months = 10 years
        self.numberdots = 1000
        


    """ Opdating the parameters and variables """
    def setup_updated(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)



    """ Defining the function """
    def hairdresserfunc(self):

        # (i) price per haircut
        price = self.kappa_t * y_t ** (- self.eta)

        # (ii) hairdressers employed
        y_t = l_t ** (1 - self.eta)

        # (iii) profit
        pi_t = p_t * y_t - self.w * l_t 
