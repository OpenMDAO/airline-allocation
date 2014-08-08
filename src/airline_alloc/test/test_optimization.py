
import unittest

import numpy as np

from airline_alloc.optimization import *


def load_data(file_name):
    """ utility function to load MATLAB data
    """
    from os.path import dirname, pardir, join
    from scipy.io import loadmat

    import airline_alloc
    data_path = join(dirname(airline_alloc.__file__),pardir,pardir,'MATLAB','Data')

    return loadmat(join(data_path,file_name),
                   squeeze_me=True, struct_as_record=False)


class RangeExtractTestCase(unittest.TestCase):

    def test_3routes(self):
        inputs = load_data('inputs_before_3routes.mat')['Inputs']

        distance = [2000, 1500, 1000]

        indices = range_extract(inputs.RVector, distance)

        expected = np.array([80, 1782, 674])

        self.assertTrue(np.allclose(indices, expected-1))  # zero indexing

    def test_11routes(self):
        inputs = load_data('inputs_before_11routes.mat')['Inputs']

        distance = [162, 753, 974, 1094, 1357, 1455, 2169, 2249, 2269, 2337, 2350]

        indices = range_extract(inputs.RVector, distance)

        expected = np.array([
            394, 1598, 410, 2042, 615, 742, 427, 1501, 1308, 414, 1317
        ])

        self.assertTrue(np.allclose(indices, expected-1))  # zero indexing

    def test_31routes(self):
        inputs = load_data('inputs_before_31routes.mat')['Inputs']

        distance = [
            113, 174, 289, 303, 324, 331,  342,  375,  407,  427,
            484, 486, 531, 543, 550, 570,  594,  609,  622,  680,
            747, 758, 760, 823, 837, 991, 1098, 1231, 1407, 1570, 1626
        ]

        indices = range_extract(inputs.RVector, distance)

        expected = np.array([
            1483, 1045, 1944, 1856,  754, 1463, 1718,  948,  416, 1801,
            1795,  225,  845,  919,  840, 1746, 1739, 1797, 1947, 1987,
            1429, 1802, 2060, 1897, 1410, 1241, 971,  1399, 1597, 2028, 1433
        ])

        self.assertTrue(np.allclose(indices, expected-1))  # zero indexing


class ObjectiveTestCase(unittest.TestCase):

    def test_3routes(self):
        inputs       = load_data('inputs_after_3routes.mat')['Inputs']
        outputs      = load_data('outputs_after_3routes.mat')['Outputs']
        constants    = load_data('constants_after_3routes.mat')['Constants']
        coefficients = load_data('coefficients_after_3routes.mat')['Coefficients']

        obj_int, obj_con = get_objective(inputs, outputs, constants, coefficients)

        expected_int = np.array([
            30078.1801074742,
            23390.7454818768,
            16779.0566092325,
            35794.3199911030,
            28282.0591370451,
            20794.6131249422
        ])

        expected_con = np.array([
            -295.868219080555,
            -235.334963855215,
            -176.747839203397,
            -308.770102562314,
            -248.293639817771,
            -188.596040811846,
        ])

        self.assertTrue(np.allclose(obj_int, expected_int))
        self.assertTrue(np.allclose(obj_con, expected_con))


class ConstraintsTestCase(unittest.TestCase):

    def test_3routes(self):
        inputs       = load_data('inputs_after_3routes.mat')['Inputs']
        constants    = load_data('constants_after_3routes.mat')['Constants']
        coefficients = load_data('coefficients_after_3routes.mat')['Coefficients']

        A, b = get_constraints(inputs, constants, coefficients)

        expected_A = np.array([
            [0,       0,    0,   0,     0,    0,   1,   0,   0,   1,   0,   0],
            [0,       0,    0,   0,     0,    0,   0,   1,   0,   0,   1,   0],
            [0,       0,    0,   0,     0,    0,   0,   0,   1,   0,   0,   1],
            [0,       0,    0,   0,     0,    0,  -1,   0,   0,  -1,   0,   0],
            [0,       0,    0,   0,     0,    0,   0,  -1,   0,   0,  -1,   0],
            [0,       0,    0,   0,     0,    0,   0,   0,  -1,   0,   0,  -1],
            [10.4652511360000,    8.31288377600000,    6.15314412800000,    0,   0,   0,   0,   0,   0,   0,   0,   0],
            [0,       0,    0,   10.4460000480000,    8.29977156800000,    6.14597705600000,    0,   0,   0,   0,   0,   0],
            [-107,    0,    0,    0,    0,    0,   1,   0,   0,   0,   0,   0],
            [0,    -107,    0,    0,    0,    0,   0,   1,   0,   0,   0,   0],
            [0,       0, -107,    0,    0,    0,   0,   0,   1,   0,   0,   0],
            [0,       0,    0, -122,    0,    0,   0,   0,   0,   1,   0,   0],
            [0,       0,    0,    0, -122,    0,   0,   0,   0,   0,   1,   0],
            [0,       0,    0,    0,    0, -122,   0,   0,   0,   0,   0,   1]
        ])

        expected_b = np.array([
            300, 700, 220, -60, -140, -44, 72, 48, 0, 0, 0, 0, 0, 0
        ]).reshape(-1, 1)

        self.assertTrue(np.allclose(A, expected_A))
        self.assertTrue(np.allclose(b, expected_b))


class GomoryCutTestCase(unittest.TestCase):

    def test_problem(self):
        """ test problem from GomoryCut.m
        """
        # input arguments
        x = np.array([
            [55./14.],
            [10./7.]
        ])
        A = np.array([
            [2./5., 1.],
            [2./5., -2./5.]
        ])
        b = np.array([
            [3.],
            [1.]
        ])
        Aeq = np.array([])
        beq = np.array([])

        # expected results (from MATLAB)
        expected = {
            'A_up': np.array([
                [0.4000,    1.0000],
                [0.4000,   -0.4000],
                [0.6000,    0.4000]
            ]),
            'b_up': np.array([
                [3.0000],
                [1.0000],
                [2.0000]
            ]),
            'eflag': 1
        }

        # call the function
        A_up, b_up, eflag = gomory_cut(x, A, b, Aeq, beq)

        # check answer against expected results
        self.assertTrue(np.allclose(A_up, expected['A_up']))
        self.assertTrue(np.allclose(b_up, expected['b_up']))
        self.assertTrue(eflag == expected['eflag'])


class CutPlaneTestCase(unittest.TestCase):

    def test_problem(self):
        """ test problem from call_Cutplane.m
        """
        # input arguments
        x = np.array([
            [9./4.],
            [15./4.],
            [1200.],
            [500.]
        ])
        A = np.array([
            [1., 1., 3., 3.],
            [4., 1., 6., 5.],
            [1., 1., 0., 0.],
            [5., 9., 0., 0.]
        ])
        b = np.array([
            [12.],
            [1.],
            [6.],
            [45.]
        ])
        Aeq = np.array([])
        beq = np.array([])

        ind_con = np.array([0, 1])  # indices of con rows
        ind_int = np.array([2, 3])  # indices of int rows

        indeq_con = np.array([])
        indeq_int = np.array([])

        num_int = 2

        # expected results (from MATLAB)
        expected = {
            'A_up': np.array([
                [1., 1., 3., 3.],
                [4., 1., 6., 5.],
                [1., 1., 0., 0.],
                [5., 9., 0., 0.],
                [2., 3., 1., 1.],
            ]),
            'b_up': np.array([
                [12.],
                [1.],
                [6.],
                [45.],
                [1715.]
            ]),
            'eflag': 1
        }

        # call the function
        A_up, b_up  = cut_plane(x, A, b, Aeq, beq, ind_con, ind_int, indeq_con, indeq_int, num_int)

        # check answer against expected results
        self.assertTrue(np.allclose(A_up, expected['A_up']))
        self.assertTrue(np.allclose(b_up, expected['b_up']))


class BranchCutTestCase(unittest.TestCase):

    def test_branch_cut(self):
        # smaller network with 3 routes
        inputs       = load_data('inputs_after_3routes.mat')['Inputs']
        outputs      = load_data('outputs_after_3routes.mat')['Outputs']
        constants    = load_data('constants_after_3routes.mat')['Constants']
        coefficients = load_data('coefficients_after_3routes.mat')['Coefficients']

        # linear objective coefficients
        objective   = get_objective(inputs, outputs, constants, coefficients)
        f_int = objective[0]    # integer type design variables
        f_con = objective[1]    # continuous type design variables

        # coefficient matrix for linear inequality constraints, Ax <= b
        constraints = get_constraints(inputs, constants, coefficients)
        A = constraints[0]
        b = constraints[1]

        J = inputs.DVector.shape[0]  # number of routes
        K = len(inputs.AvailPax)     # number of aircraft types

        # lower and upper bounds
        lb = np.zeros((2*K*J, 1))
        ub = np.concatenate((
            np.ones((K*J, 1)) * inputs.MaxTrip.reshape(-1, 1),
            np.ones((K*J, 1)) * np.inf
        ))

        # initial x
        x0 = []

        # indices into A matrix for continuous & integer/continuous variables
        ind_conCon = range(2*J)
        ind_intCon = range(2*J, len(constraints[0])+1)

        # call the branch and cut algorithm to solve the MILP problem
        branch_cut(f_int, f_con, A, b, [], [], lb, ub, x0,
                   ind_conCon, ind_intCon, [], [])


if __name__ == "__main__":
    unittest.main()
