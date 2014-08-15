
import unittest
from unittest import SkipTest

import numpy as np
from numpy import inf

np.set_printoptions(linewidth=240, suppress=True)


class LinearProgramTestCase(unittest.TestCase):
    """ test linear program solvers
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


if __name__ == "__main__":
    unittest.main()
