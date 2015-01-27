"""
An Airline Allocation sub-problem is used to demonstrate usage of
a linear program solver as an OpenMDAO component.
"""

from openmdao.main.api import Component, Assembly, set_as_top
from openmdao.main.datatypes.api import Array, Str

import numpy as np

from dataset import *
from optimization import get_objective, get_constraints

from linear_program import LinProg, LPSolve


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

        # indices into A matrix for continuous & mixed integer/continuous variables
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
