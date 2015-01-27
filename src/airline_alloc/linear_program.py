"""
This file demonstrates the wrapping of a linear program solver as an OpenMDAO component.

Two different solvers are implemented: numpy.optimize.linprog and lpsolve

As of this writing, there is a bug in linprog that results in an incorrect solution
for some cases. This has been reported and should be fixed in due course.

In the meantime it is recommended to download and install linprog from:

    http://lpsolve.sourceforge.net

    You will need the libraries from the appropriate 'lp_solve_*_dev' archive
    and the python setup from the 'lp_solve_*_Python_source' archive
"""

from openmdao.main.api import Component
from openmdao.main.datatypes.api import Array, Float, Bool, Int

from zope.interface import Interface, Attribute, implements

import numpy as np

try:
    from scipy.optimize import linprog
except ImportError, e:
    print "SciPy version >= 0.15.0 is required for linprog support!!"
    pass

try:
    from lpsolve55 import *
except ImportError:
    print 'lpsolve is not available'
    pass


class ILinearProgram(Interface):
    # inputs
    f       = Attribute('coefficients of the linear objective function to be maximized')

    A       = Attribute('2-D array which, when matrix-multiplied by x, gives the values of the upper-bound inequality constraints at x')

    b       = Attribute('1-D array of values representing the upper-bound of each inequality constraint (row) in A_ub')

    A_eq    = Attribute('2-D array which, when matrix-multiplied by x, gives the values of the equality constraints at x')

    b_eq    = Attribute('1-D array of values representing the RHS of each equality constraint (row) in A_eq')

    lb      = Attribute('lower bounds for each independent variable in the solution')

    ub      = Attribute('upper bounds for each independent variable in the solution')

    # outputs
    x       = Attribute('independent variable vector which optimizes the linear programming problem')

    fun     = Attribute('function value')

    success = Attribute('flag indicating success or failure in finding an optimal solution')

    status  = Attribute('exit status of the optimization: 0=optimized, 1=max iterations, 2=infeasible, 3=unbounded')

    def execute(self):
        """ solve the linear program """


class LinProg(Component):
    """ A simple component wrapper for scipy.optimize.linprog
    """
    implements(ILinearProgram)

    # inputs
    f     = Array(iotype='in',
            desc='coefficients of the linear objective function to be maximized')

    A     = Array(iotype='in',
            desc='2-D array which, when matrix-multiplied by x, gives the values of the upper-bound inequality constraints at x')

    b     = Array(iotype='in',
            desc='1-D array of values representing the upper-bound of each inequality constraint (row) in A_ub')

    A_eq  = Array(iotype='in',
            desc='2-D array which, when matrix-multiplied by x, gives the values of the equality constraints at x')

    b_eq  = Array(iotype='in',
            desc='1-D array of values representing the RHS of each equality constraint (row) in A_eq')

    lb    = Array(iotype='in',
            desc='lower bounds for each independent variable in the solution')

    ub    = Array(iotype='in',
            desc='upper bounds for each independent variable in the solution')

    # outputs
    x     = Array(iotype='out',
            desc='independent variable vector which optimizes the linear programming problem')

    fun   = Float(iotype='out',
            desc='function value')

    success = Bool(iotype='out',
              desc='flag indicating success or failure in finding an optimal solution')

    status  = Int(iotype='out',
              desc='exit status of the optimization: 0=optimized, 1=max iterations, 2=infeasible, 3=unbounded')

    def execute(self):
        """ solve the linear program """

        results = linprog(self.f,
                          A_eq=self.A_eq,   b_eq=self.b_eq,
                          A_ub=self.A,      b_ub=self.b,
                          bounds=zip(self.lb, self.ub),
                          options={ 'maxiter': 100, 'disp': True })

        print self.get_pathname(), 'results:\n---------------\n', results, '\n---------------'
        self.x   = results.x
        self.fun = results.fun
        self.success = results.success
        self.status  = results.status


class LPSolve(Component):
    """ A simple component wrapper for lpsolve
    """
    implements(ILinearProgram)

    # inputs
    f     = Array(iotype='in',
            desc='coefficients of the linear objective function to be maximized')

    A     = Array(iotype='in',
            desc='2-D array which, when matrix-multiplied by x, gives the values of the upper-bound inequality constraints at x')

    b     = Array(iotype='in',
            desc='1-D array of values representing the upper-bound of each inequality constraint (row) in A_ub')

    A_eq  = Array(iotype='in',
            desc='2-D array which, when matrix-multiplied by x, gives the values of the equality constraints at x')

    b_eq  = Array(iotype='in',
            desc='1-D array of values representing the RHS of each equality constraint (row) in A_eq')

    lb    = Array(iotype='in',
            desc='lower bounds for each independent variable in the solution')

    ub    = Array(iotype='in',
            desc='upper bounds for each independent variable in the solution')

    # outputs
    x     = Array(iotype='out',
            desc='independent variable vector which optimizes the linear programming problem')

    fun   = Float(iotype='out',
            desc='function value')

    success = Bool(iotype='out',
              desc='flag indicating success or failure in finding an optimal solution')

    status  = Int(iotype='out',
              desc='exit status of the optimization: 0=optimized, 1=max iterations, 2=infeasible, 3=unbounded')

    def execute(self):
        """ solve the linear program """

        obj = self.f.tolist()
        lp = lpsolve('make_lp', 0, len(obj))
        lpsolve('set_verbose', lp, 'IMPORTANT')
        lpsolve('set_obj_fn', lp, obj)

        i = 0
        for con in self.A:
            lpsolve('add_constraint', lp, con.tolist(), 'LE', self.b[i])
            i = i+1

        for i in range (len(self.lb)):
            lpsolve('set_lowbo', lp, i+1,  self.lb[i])
            lpsolve('set_upbo',  lp, i+1, self.ub[i])

        results = lpsolve('solve', lp)

        print self.get_pathname(), 'results:\n---------------\n', results, '\n---------------'
        self.x   = np.array(lpsolve('get_variables', lp)[0])
        print 'fun:', lpsolve('get_objective', lp)
        self.fun = lpsolve('get_objective', lp)
        self.success = True if results == 0 else False
        if results == 0:            # optimized
            self.status = 1
        elif results == 2:          # infeasible
            self.status = -2
        elif results == 3:          # unbounded
            self.status = -3
        else:
            self.status = -1
        lpsolve('delete_lp', lp)
