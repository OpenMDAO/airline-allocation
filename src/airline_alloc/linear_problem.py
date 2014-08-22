"""
This file demonstrates the use of a linear program solver as an OpenMDAO component.

Two different solvers are implemented: numpy.optimize.linprog and lpsolve

As of this writing, there is a bug in linprog that results in an incorrect solution
for some cases. This has been reported and should be fixed in due course.

In the meantime it is recommended to download and install lpsolve from:

    http://lpsolve.sourceforge.net

    You will need the libraries from the appropriate 'lp_solve_*_dev' archive
    and the python setup from the 'lp_solve_*_Python_source' archive

An Airline Allocation sub-problem is used to demonstrate usage of the component.
"""

from openmdao.main.api import Component, Assembly, set_as_top
from openmdao.main.datatypes.api import Array, Float, Bool, Int, Str

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

from dataset import *
from optimization import get_objective, get_constraints


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


class AirlineAllocationProblem(Component):
    """ Selects a subset of the large network dataset (2134 routes and 18 types of aircraft)
        specified by the inputs (ac_ind, ac_num, distance, dvector) and formulates a linear
        programming problem defined by the outputs
    """

    # inputs
    filename = Str('Dataset.mat', iotype='in',
               desc='the filename of the MATLAB data file containing the aircraft/network data')

    ac_ind   = Array(iotype='in', desc='the indices of the aircraft to select')

    ac_num   = Array(iotype='in', desc='the number of each aircraft')

    distance = Array(iotype='in', desc='the route distances to select')

    dvector  = Array(iotype='in', desc='the route demand')

    # outputs
    f   = Array(iotype='out',
          desc='coefficients for the linear objective function to be maximized')

    A   = Array(iotype='out',
          desc='2-D array which, when matrix-multiplied by x, gives the values of the upper-bound inequality constraints at x')

    b   = Array(iotype='out',
          desc='1-D array of values representing the upper-bound of each inequality constraint (row) in A')

    Aeq = Array(iotype='out',
          desc='2-D array which, when matrix-multiplied by x, gives the values of the equality constraints at x')

    beq = Array(iotype='out',
          desc='1-D array of values representing the RHS of each equality constraint (row) in Aeq')

    lb  = Array(iotype='out',
          desc='lower bounds for each independent variable in the solution')

    ub  = Array(iotype='out',
          desc='upper bounds for each independent variable in the solution')

    ind_conCon = Array(iotype='out',
            desc='indices in the A matrix correspoding to the constraints containing only continuous type design variables')

    ind_intCon = Array(iotype='out',
            desc='indices in the A matrix correspoding to the constraints containing integer and continuous (if any) type design variables')

    def execute(self):
        # read and select the data
        data = Dataset(self.filename)
        data.filter(self.ac_ind, self.ac_num, self.distance, self.dvector)

        # linear objective coefficients
        objective = self.get_objective(data)
        f_int = objective[0]                # integer type design variables
        f_con = objective[1]                # continuous type design variables
        self.f = np.concatenate((f_int, f_con))

        # coefficient matrix for linear inequality constraints, Ax <= b
        constraints = self.get_constraints(data)
        self.A = constraints[0]
        self.b = constraints[1]

        # coefficient matrix for linear equality constraints, Aeqx <= beq (N/A)
        self.Aeq = np.ndarray(shape=(0, 0))
        self.beq = np.ndarray(shape=(0, 0))

        J = data.inputs.DVector.shape[0]    # number of routes
        K = len(data.inputs.AvailPax)       # number of aircraft types

        # lower and upper bounds
        self.lb = np.zeros((2*K*J, 1))
        self.ub = np.concatenate((
            np.ones((K*J, 1)) * data.inputs.MaxTrip.reshape(-1, 1),
            np.ones((K*J, 1)) * np.inf
        ))

        # indices into A matrix for continuous & integer/continuous variables
        self.ind_conCon = range(2*J)
        self.ind_intCon = range(2*J, len(constraints[0])+1)

    def get_objective(self, data):
        """ generate the objective matrix for linprog
            returns the coefficients for the integer and continuous design variables
        """
        # sharing function definition with non-openmdao code
        return get_objective(data)

    def get_constraints(self, data):
        """ generate the constraint matrix/vector for linprog
        """
        # sharing function definition with non-openmdao code
        return get_constraints(data)


if __name__ == '__main__':
    top = set_as_top(Assembly())
    top.add('problem', AirlineAllocationProblem())
    top.add('program', LPSolve())  # LinProg())

    top.connect('problem.f',   'program.f')
    top.connect('problem.A',   'program.A')
    top.connect('problem.b',   'program.b')
    top.connect('problem.Aeq', 'program.A_eq')
    top.connect('problem.beq', 'program.b_eq')
    top.connect('problem.lb',  'program.lb')
    top.connect('problem.ub',  'program.ub')

    top.driver.workflow.add(['problem', 'program'])

    top.problem.ac_ind    = np.array([9, 10]) - 1
    top.problem.ac_num    = np.array([6,  4])
    top.problem.distance  = np.array([2000, 1500, 1000])
    top.problem.dvector   = np.array([
        [1, 300],
        [2, 700],
        [3, 220]
    ])

    top.run()

    print 'success: \t', top.program.success
    print 'status:  \t', top.program.status
    print 'solution:\t', top.program.x
    print 'value:   \t', top.program.fun
