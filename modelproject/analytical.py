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
        self.utility()
        self.consumerBC
        self.eulerequation
        self.Prices
        self.Optimalsaving


    """ Defining the parameters and variables """
    def setup(self):

        par = self.par

        # (i) Consumer's utility function
        par.c1t = sm.symbols('c_{1t}')          # Young 
        par.c2t1 = sm.symbols('c_{2t+1}')       # Old
        par.beta = sm.symbols('beta')           # preference
        par.Ut = sm.symbols('U_t')              # utility

        # (ii) Young generation's budget contraints
        par.wt = sm.symbols('w_t')              # wage in t
        par.tau = sm.symbols('tau')             # tax
        par.st = sm.symbols('s_t')              # saving rate

        # (iii) Old generation's budget contraints
        par.wt1 = sm.symbols('w_{t+1}')         # wage in t+1
        par.dt1 = sm.symbols('d_{t+1}')         # retirement in t+1
        par.n = sm.symbols('n')                 # population growth
        par.rt = sm.symbols('r_t')              # rent of capital in t
        par.rt1 = sm.symbols('r_{t+1}')         # rent of capital in t+1

        # (iv) Lagrange multiplier
        par.lambdaa = sm.symbols('lambda')      # lamdba

        # (v) Firm's production function
        par.Kt = sm.symbols('K_t')              # Kapital in t
        par.Kt1 = sm.symbols('K_{t+1}')         # Kapital in t+1
        par.kt = sm.symbols('k_t')              # kapital per worker in t
        par.kt1 = sm.symbols('k_{t+1}')         # kapital per worker in t+1
        par.kss = sm.symbols('k^*')             # ss for capital
        par.Lt = sm.symbols('L_t')              # Labour in t
        par.Lt1 = sm.symbols('L_{t+1}')         # Labour in t+1
        par.alpha = sm.symbols('alpha')         # share of capital
        par.A = sm.symbols('A')                 # TFP



    """ Defining the utility function for consumers """
    def utility(self):
        par = self.par

        # (i) Consumer's utility function
        return sm.log(par.c1t) + par.beta * sm.log(par.c2t1)



    """ Defining the budget constraint for consumers """
    def consumerBC(self):
        par = self.par



        # (i) Young generation's BC
        BC_Y = sm.Eq(par.c1t, (1 - par.tau) * par.wt - par.st)

        # (i.a) Isolating for disposal income
        BC_Y_dis_income = sm.Eq((1 - par.tau) * par.wt, par.c1t + par.st)


        # (ii) Old generation's BC
        d_t1 = par.wt1 + par.tau
        BC_O = sm.Eq(par.c2t1, par.st * (1 + par.rt1) + (1 + par.n) * d_t1)

        # (ii.a) Isolating saving qua 'solver'
        BC_O_saving = sm.solve(BC_O, par.st) 
        

        # (iii) Solving for the optimal savingrate
        BC_Y = BC_Y_dis_income.subs(par.st, BC_O_saving)
        L = sm.solve(BC_Y, par.wt * (1 - par.tau))[0]
        R = par.wt * (1-par.tau)


        return L-R
        


    """ Defining Eulers Equation """
    def eulerequation(self):
        par = self.par

        # (a) Lagrange
        Lagrangee = self.utility() + par.lambdaa * self.consumerBC()


        # (b) FOC
        FOC_Y = sm.Eq(0, sm.diff(Lagrangee, par.c1t))
        FOC_O = sm.Eq(0, sm.diff(Lagrangee, par.c2t1))


        # (c) Lambda
        Lambda1 = sm.solve(FOC_Y, par.lambdaa)[0]
        Lambda2 = sm.solve(FOC_O, par.lambdaa)


        # (d) Eulers equation
        E_Y = sm.solve(sm.Eq(Lambda1, Lambda2), par.c1t)[0]


        # (e) Return 
        return sm.Eq(E_Y, par.c1t)



    """ Defining prices for the firms """
    def Prices(self):
        par = self.par

        # (i) Firms profit funciton
        Profit_t = par.A * par.Kt ** par.alpha * par.Lt ** (1-par.alpha) - (1 + par.rt) * par.Kt - par.wt * par.Lt
        Profit_t1 = par.A * par.Kt1 ** par.alpha * par.Lt1 ** (1-par.alpha) - (1 + par.rt1) * par.Kt1 - par.wt1 * par.Lt1


        # (ii) Rent of capital (FOC)
        r_k_t = sm.Eq(0, sm.diff(Profit_t, par.Kt))
        r_k_t1 = sm.Eq(0, sm.diff(Profit_t1, par.Kt1))        
        

        # (iii) Solve for rent of capital
        r_k__t = sm.solve(r_k_t, par.rt)[0]
        r_k__t1 = sm.solve(r_k_t1, par.rt1)[0]


        # (iv) Defining capital 
        K_t_ = par.Lt * par.kt
        K_t1_ = par.Lt1 * par.kt1


        # (v) Inserting
        r_k___t = r_k_t.subs(par.Kt, K_t_)[0]
        r_k___t1 = r_k_t1.subs(par.Kt1, K_t1_)[0]
        

        # (vi) Wage (FOC)
        w__t = sm.Eq(0, sm.diff(Profit_t, par.Lt))
        w__t1 = sm.Eq(0, sm.diff(Profit_t1, par.Lt1)) 
        
        
        # (vii) Solve for wage
        w___t = sm.solve(w__t, par.wt)[0]
        w___t1 = sm.solve(w__t1, par.wt1)[0]


        # (iix) Inserting
        w____t = w__t.subs(par.Kt, K_t_)[0]
        w____t1 = w__t1.subs(par.Kt1, K_t1_)[0]


        return r_k___t, r_k___t1, w____t, w____t1



    """ Defining the optimal saving for the consumers """
    def Optimalsaving(self):
        par = self.par


        # (i) Redefining the public retirement
        d_t1_ = par.wt1 * par.tau


        # (ii) BC for periods
        BC_Y_ = par.wt * (1 - par.tau) - par.st 
        BC_O_ = par.st * (1 + par.rt1) + d_t1_ * (1 + par.n)


        # (iii) BC into Euler
        Eulers = self.eulersequation()
        Saving_Y = Eulers.subs(par.c1t, BC_Y_)
        Saving_O = Eulers.subs(par.c2t1, BC_O_)


        # (iv) Optimal saving
        return sm.solve(Saving_O, par.st)[0]
    


    """ Defining capital accumulation """
    def capitalacc(self):
        par = self.par

        # (i) Transition
        k_t1_ = 1 / (1+par.n) * par.st


        # (ii) Market clearing in t=1
        r_1 = self.Prices()[0]
        w_1 = self.Prices()[1]


        # (iii) Market clearing in t=2
        r_2 = self.Prices()[2]
        w_2 = self.Prices()[3]


        # (iv) savings
        k_t1__ = k_t1_.subs(par.st, self.Optimalsaving())


        # (v) Substituting parameters 

        # (v.a) rent of capital
        k_t1___ = k_t1__.subs(par.rt, r_1)
        k_t1____ = k_t1___.subs(par.rt1, r_2)
        
        # (v.b) wage
        k_t1_____ = k_t1____.subs(par.wt, w_1)
        k_t1______ = k_t1_____.subs(par.wt1, w_2)


        # (vi) Equation for capital
        k_t1_eq = sm.Eq(0, (par.kt1 - k_t1______))


        # (vii) Solving for kapital
        k_t1_eq_ = sm.solve(k_t1_eq, par.kt1)[0]


        # (iix) Steady state
        k_t1_eq__ = k_t1_eq_.subs(par.kt, par.kss)
        k_t1_eq___ = sm.Eq(par.kss, k_t1_eq__)
        kapital_ss = sm.solve(k_t1_eq___, par.kss)[0]


        # (ix) Return kapital (k_(t+1)) and steady state for capital
        return sm.Eq(par.kt1, k_t1_eq_), sm.Eq(par.kss, kapital_ss), sm.solve(k_t1_eq___, par.kss)[0] 
        