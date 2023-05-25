# Importing packages
from scipy import optimize
import sympy as sm
import numpy as np
from types import SimpleNamespace

class OLGmodelanalytical():

    """Creating the model """
    def __init__(self):
        
        self.par = SimpleNamespace()
        self.setup()


    """ Defining the parameters and variables """
    def setup(self):

        par = self.par

        # (i) Consumer's utility function
        par.c_1t = sm.symbols('c_{1t}')          # Young 
        par.c_2t1 = sm.symbols('c_{2t+1}')       # Old
        par.rho = sm.symbols('rho')             # preference
        par.U_t = sm.symbols('U_t')              # utility

        # (ii) Young generation's budget contraints
        par.w_t = sm.symbols('w_t')              # wage in t
        par.tau = sm.symbols('tau')             # tax
        par.s_t = sm.symbols('s_t')              # saving rate

        # (iii) Old generation's budget contraints
        par.w_t1 = sm.symbols('w_{t+1}')         # wage in t+1
        par.d_t1 = sm.symbols('d_{t+1}')         # retirement in t+1
        par.n = sm.symbols('n')                 # population growth
        par.r_t = sm.symbols('r_t')              # rent of capital in t
        par.r_t1 = sm.symbols('r_{t+1}')         # rent of capital in t+1

        # (iv) Lagrange multiplier
        par.lambdaa = sm.symbols('lambda')      # lamdba

        # (v) Firm's production function
        par.K_t = sm.symbols('K_t')              # capital in t
        par.K_t1 = sm.symbols('K_{t+1}')         # capital in t+1
        par.k_t = sm.symbols('k_t')              # capital per worker in t
        par.k_t1 = sm.symbols('k_{t+1}')         # kapital per worker in t+1
        par.k_ss = sm.symbols('k^*')             # ss for capital
        par.L_t = sm.symbols('L_t')              # Labour in t
        par.L_t1 = sm.symbols('L_{t+1}')         # Labour in t+1
        par.alpha = sm.symbols('alpha')         # share of capital
        par.A = sm.symbols('A')                 # TFP

        # (vi) helping parameters
        par.a = sm.symbols('a')
        par.b = sm.symbols('b')
        par.c = sm.symbols('c')


    """ Defining the utility function for consumers """
    def utility(self):
        par = self.par

        # (i) Consumer's utility function
        return sm.log(par.c_1t) + 1 / (1 + par.rho) * sm.log(par.c_2t1)



    """ Defining the budget constraint for consumers """
    def consumerBC(self):
        par = self.par

        # (i) Public retirement
        d_t1 = par.w_t1 + par.tau

        # (ii) Young generation's BC
        BC_Y = sm.Eq(par.c_1t + par.s_t, (1 - par.tau) * par.w_t)

        # (iii) Old generation's BC
        BC_O = sm.Eq(par.c_2t1, par.s_t * (1 + par.r_t1) + (1 + par.n) * d_t1)

        # (iv) Isolating saving qua 'solver'
        BC_O_saving = sm.solve(BC_O, par.s_t) 

        # (v) Inserting savings into BC_Y
        BC_Y_s =  BC_Y.subs(par.s_t, BC_O_saving[0])

        # (vi) Solving for the optimal savingrate
        RHS = sm.solve(BC_Y_s, par.w_t * (1 - par.tau))[0]
        LHS = par.w_t * (1 - par.tau)

        # (vii) Return
        return RHS - LHS
        


    """ Defining Eulers Equation """
    def eulerequation(self):
        par = self.par

        # (i) Lagrange
        Lagrange = self.utility() + par.lambdaa * self.consumerBC()

        # (ii) FOC
        FOC_Y = sm.Eq(0, sm.diff(Lagrange, par.c_1t))
        FOC_O = sm.Eq(0, sm.diff(Lagrange, par.c_2t1))

        # (iii) Lambda
        Lambda1 = sm.solve(FOC_Y, par.lambdaa)[0]
        Lambda2 = sm.solve(FOC_O, par.lambdaa)[0]

        # (iv) Eulers equation
        E_Y = sm.solve(sm.Eq(Lambda1, Lambda2), par.c_1t)[0]

        # (v) Return 
        return sm.Eq(E_Y, par.c_1t)



    """ Defining the optimal saving for the consumers """
    def Optimalsaving(self):
        par = self.par

        # (i) Redefining the public retirement
        d_t1 = par.w_t1 * par.tau

        # (ii) BC for periods
        BC_Y = par.w_t * (1 - par.tau) - par.s_t 
        BC_O = par.s_t * (1 + par.r_t1) + d_t1 * (1 + par.n)

        # (iii) BC into Euler
        Eul = self.eulerequation()
        sav = (Eul.subs(par.c_1t, BC_Y)).subs(par.c_2t1, BC_O)

        # (iv) Simplify equations
        saving1 = sm.solve(sav, par.s_t)[0]
        saving11 = sm.collect(saving1, [par.tau])
        saving = sm.collect(saving1, [par.w_t, par.w_t1])

        # (v) Optimal saving
        return saving
    


    """ Defining capital accumulation """
    def capitalacc(self):
        par = self.par

        # (i) defining parameters to help
        a = (1 / (1 + (1 + par.rho)/(2 + par.rho) * ((1 - par.alpha) / par.alpha) * par.tau))
        b = ((1 - par.alpha) * (1 - par.tau)) / ((1 + par.n) * (2 + par.rho))
        c = par.A * par.k_t**par.alpha
         
        # (ii) Capital accumulation
        kt_00 = par.a * (par.b * par.c)
        kt_01 = ((kt_00.subs(par.a, a)).subs(par.b ,b)).subs(par.c, c)
        k_t = sm.Eq(par.k_t1, kt_01)

        # (iii) Return
        return k_t
        
        
        
    """ Defining steady state """
    def steadystate(self):
        par = self.par

        # (i) Helping parameters
        a = (1 / (1 + (1 + par.rho) / (2 + par.rho) * ((1 - par.alpha) / par.alpha) * par.tau))
        b = ((1 - par.alpha) * (1 - par.tau)) / ((1 + par.n) * (2 + par.rho))
        c = 1 / (1 - par.alpha)
        
        # (ii) Steady state 
        k_ss_0 = (a * b * par.a) ** par.c
        k_ss_1 = ((k_ss_0.subs(par.a, a)).subs(par.b, b)).subs(par.c, c) 
        k_star = sm.Eq(par.k_ss, k_ss_1)
        
        # (iii) Return 
        return k_ss_1