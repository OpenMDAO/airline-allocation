
from openmdao.main.api import Component, Array, Bool, Int

try:
    from scipy.optimize import linprog
except ImportError, e:
    print "SciPy version >= 0.15.0 is required for linprog support!!"
    pass


class LinearProblem(Component):

    # inputs
    c     = Array(iotype='in',
            desc='coefficients of the linear objective function to be maximized')

    A_ub  = Array(iotype='in',
            desc='2-D array which, when matrix-multiplied by x, gives the values of the upper-bound inequality constraints at x')

    b_ub  = Array(iotype='in',
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

    fun   = Array(iotype='out',
            desc='function value')

    success = Bool(iotype='out',
              desc='flag indicating success or failure in finding an optimal solution')

    status  = Int(iotype='out',
              desc='exit status of the optimization: 0=optimized, 1=max iterations, 2=infeasible, 3=unbounded')

    def execute(self):
        """ solve problem
        """

        results = linprog(self.c,
                          A_eq=self.A_eq,   b_eq=self.b_eq,
                          A_ub=self.A_ub,   b_ub=self.b_ub,
                          bounds=zip(self.lb, self.ub),
                          options={ 'maxiter': 100, 'disp': True })

        print self.get_pathname(), 'results:\n---------------\n', results, '\n---------------'
        self.x   = results.x
        self.fun = results.fun
        self.success = results.success
        self.status  = results.status
