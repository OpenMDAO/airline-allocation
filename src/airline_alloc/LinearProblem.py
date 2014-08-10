
from openmdao.main.api import Component, Array, Bool

import numpy as np

try:
    from scipy.optimize import linprog
except ImportError, e:
    print "SciPy version >= 0.15.0 is required for linprog support!!"
    pass


class LinearProblem(Component):

    # inputs
    f     = Array(iotype='in', desc='objective function')

    A     = Array(iotype='in', desc='inequality constraints, coefficients matrix')
    b     = Array(iotype='in', desc='inequality constraints')

    Aeq   = Array(iotype='in', desc='equality constraints, coefficients matrix')
    beq   = Array(iotype='in', desc='equality constraints')

    lb    = Array(iotype='in', desc='lower bounds')
    ub    = Array(iotype='in', desc='upper bounds')

    # outputs
    x_F   = Array(iotype='out', desc='')
    b_F   = Array(iotype='out', desc='')

    eflag = Bool(iotype='out', desc='flag indicating success or failure in finding a solution')

    def execute(self):
        """ solve subproblem using linprog
        """
        bounds = zip(self.lb, self.ub)

        results = linprog(self.f,
                          A_eq=None,   b_eq=None,
                          A_ub=self.A, b_ub=self.b,
                          bounds=bounds,
                          options={ 'maxiter': 100, 'disp': True })

        print 'results:\n---------------\n', results, '\n---------------'
        self.x_F = results.x
        self.b_F = results.fun
        self.eflag = 1 if results.success else 0
