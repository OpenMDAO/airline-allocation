
import unittest
from unittest import SkipTest

import numpy as np
from numpy import inf

np.set_printoptions(linewidth=240, suppress=True)


class LinearProgramTestCase(unittest.TestCase):
    """ test linear program solvers

        NOTE: these tests are meant to evaluate the available linear program
              solvers for python. 'lpsolve' is currently the only one that
              can be shown to get the same answer as MATLAB, which is an
              objective value of 6544.9346 for the given problem.
    """

    def setUp(self):
        """ define the problem
        """

        self.f = np.array(
            [ 30078.18010747,  23390.74548188,  16779.05660923,  35794.3199911,   28282.05913705,  20794.61312494,   -295.86821908,   -235.33496386,   -176.7478392,    -308.77010256,   -248.29363982,   -188.59604081])

        self.A = np.array([
            [   0.,            0.,            0.,            0.,            0.,            0.,            1.,            0.,            0.,            1.,            0.,            0.        ],
            [   0.,            0.,            0.,            0.,            0.,            0.,            0.,            1.,            0.,            0.,            1.,            0.        ],
            [   0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,            1.,            0.,            0.,            1.        ],
            [   0.,            0.,            0.,            0.,            0.,            0.,           -1.,            0.,            0.,           -1.,            0.,            0.        ],
            [   0.,            0.,            0.,            0.,            0.,            0.,            0.,           -1.,            0.,            0.,           -1.,            0.        ],
            [   0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,           -1.,            0.,            0.,           -1.        ],
            [  10.46525114,    8.31288378,    6.15314413,    0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.        ],
            [   0.,            0.,            0.,           10.44600005,    8.29977157,    6.14597706,    0.,            0.,            0.,            0.,            0.,            0.        ],
            [-107.,            0.,            0.,            0.,            0.,            0.,            1.,            0.,            0.,            0.,            0.,            0.        ],
            [   0.,         -107.,            0.,            0.,            0.,            0.,            0.,            1.,            0.,            0.,            0.,            0.        ],
            [   0.,            0.,         -107.,            0.,            0.,            0.,            0.,            0.,            1.,            0.,            0.,            0.        ],
            [   0.,            0.,            0.,         -122.,            0.,            0.,            0.,            0.,            0.,            1.,            0.,            0.        ],
            [   0.,            0.,            0.,            0.,         -122.,            0.,            0.,            0.,            0.,            0.,            1.,            0.        ],
            [   0.,            0.,            0.,            0.,            0.,         -122.,            0.,            0.,            0.,            0.,            0.,            1.        ],
            [   0.,           -1.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.        ],
            [   0.,            0.,            0.,            1.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.        ],
            [   0.,            0.,            0.,            0.,            0.,           -1.,            0.,            0.,            0.,            0.,            0.,            0.        ],
            [   0.,            0.,            1.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.        ],
            [   0.,            0.,            0.,            0.,            0.,            1.,            0.,            0.,            0.,            0.,            0.,            0.        ],
            [  -1.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.        ],
            [   0.,            0.,            0.,           -1.,            0.,            0.,            0.,            0.,            0.,            0.,            0.,            0.        ]
        ])

        self.b = np.array([
            [  300.],
            [  700.],
            [  220.],
            [  -60.],
            [ -140.],
            [  -44.],
            [   72.],
            [   48.],
            [    0.],
            [    0.],
            [    0.],
            [    0.],
            [    0.],
            [    0.],
            [   -7.],
            [    2.],
            [   -1.],
            [    0.],
            [    1.],
            [   -1.],
            [   -2.]
        ])

        self.lb = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        self.ub = np.array([ 12.,  12.,  12.,   8.,   8.,   8.,  inf,  inf,  inf,  inf,  inf,  inf])

        self.Aeq = np.ndarray(shape=(0, 0))
        self.beq = np.ndarray(shape=(0, 0))

    def test_linprog(self):
        """ test the scipy linprog function

            (https://github.com/scipy/scipy)
        """

        try:
            from scipy.optimize import linprog
        except ImportError:
            raise SkipTest('SciPy version >= 0.15.0 is required for linprog support!!')

        print '\n------------------------------ SciPy linprog ------------------------------'
        results = linprog(self.f,
                          A_eq=self.Aeq, b_eq=self.beq,
                          A_ub=self.A,   b_ub=self.b,
                          bounds=zip(self.lb, self.ub),
                          options={ 'maxiter': 100, 'disp': True })
        print results
        print 'x:', np.round(results.x, decimals=2)

    def test_cvxopt(self):
        """ test the cvxopt lp solver

            (https://github.com/cvxopt/cvxopt/)
        """

        try:
            from cvxopt import matrix, solvers
        except ImportError:
            raise SkipTest('cvxopt is not available')

        print '\n------------------------------ cvxopt lp ------------------------------'
        results = solvers.lp(matrix(self.f), matrix(self.A), matrix(self.b))
        print results
        print 'x:', np.round(results['x'], decimals=2).reshape(1, -1)

    def test_cvxopt_glpk(self):
        """ test the cvxopt lp solver with glpk

            (https://github.com/cvxopt/cvxopt/)
            (http://cvxopt.org/userguide/coneprog.html#optional-solvers)
            (https://www.gnu.org/software/glpk/glpk.html)
        """

        try:
            from cvxopt import matrix, solvers
        except ImportError:
            raise SkipTest('cvxopt is not available')

        print '\n------------------------------ cvxopt lp glpk ------------------------------'
        try:
            results = solvers.lp(matrix(self.f), matrix(self.A), matrix(self.b), solver='glpk')
            print results
            print 'x:', np.round(results['x'], decimals=2).reshape(1, -1)
        except ValueError, err:
            raise SkipTest(err)

    def test_lpsolve(self):
        """ test lpsolve

            (http://lpsolve.sourceforge.net)
        """

        try:
            from lpsolve55 import *
        except ImportError:
            raise SkipTest('lpsolve is not available')

        print '\n------------------------------ lpsolve ------------------------------'
        obj = self.f.tolist()
        lp = lpsolve('make_lp', 0, len(obj))
        lpsolve('set_verbose', lp, 'IMPORTANT')
        lpsolve('set_obj_fn', lp, obj)

        i = 0
        for con in self.A:
            lpsolve('add_constraint', lp, con.tolist(), 'LE', self.b[i])
            i = i+1

        for i in range (len(self.lb)):
            lpsolve('set_lowbo', lp, i+1, self.lb[i])
            lpsolve('set_upbo',  lp, i+1, self.ub[i])

        results = lpsolve('solve', lp)

        result_text = [
            'OPTIMAL      An optimal solution was obtained',
            'SUBOPTIMAL   The model is sub-optimal. Only happens if there are integer variables and there is already an integer solution found. The solution is not guaranteed the most optimal one.',
            'INFEASIBLE   The model is infeasible',
            'UNBOUNDED    The model is unbounded',
            'DEGENERATE   The model is degenerative',
            'NUMFAILURE   Numerical failure encountered',
            'USERABORT    The abort routine returned TRUE. See put_abortfunc',
            'TIMEOUT      A timeout occurred. A timeout was set via set_timeout',
            'N/A'
            'PRESOLVED    The model could be solved by presolve. This can only happen if presolve is active via set_presolve',
            'PROCFAIL     The B&B routine failed',
            'PROCBREAK    The B&B was stopped because of a break-at-first (see set_break_at_first) or a break-at-value (see set_break_at_value)',
            'FEASFOUND    A feasible B&B solution was found',
            'NOFEASFOUND  No feasible B&B solution found'
        ]

        print 'results: (%d)' % results, result_text[results]
        print 'f:', lpsolve('get_objective', lp)
        print 'x:', lpsolve('get_variables', lp)[0]

        lpsolve('delete_lp', lp)


if __name__ == "__main__":
    unittest.main()
