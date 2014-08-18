
import unittest

import numpy as np

from airline_alloc.dataset import Dataset
from airline_alloc.optimization import *


class ObjectiveTestCase(unittest.TestCase):
    """ test the get_objective function
    """

    def test_3routes(self):

        data = Dataset(suffix='after_3routes')

        obj_int, obj_con = get_objective(data)

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
    """ test the get_constraints function
    """

    def test_3routes(self):

        data = Dataset(suffix='after_3routes')

        A, b = get_constraints(data)

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
    """ test the gomotry_cut function
    """

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
        self.assertTrue(np.allclose(A_up, expected['A_up']),
            msg='\n' + str(A_up) + '\n' + str(expected['A_up']))
        self.assertTrue(np.allclose(b_up, expected['b_up']),
            msg='\n' + str(b_up) + '\n' + str(expected['b_up']))
        self.assertTrue(eflag == expected['eflag'])


class CutPlaneTestCase(unittest.TestCase):
    """ test the cut_plane function
    """

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

        ind_con = np.array([1, 2]) - 1  # indices of con rows
        ind_int = np.array([3, 4]) - 1  # indices of int rows

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
        }

        # call the function
        A_up, b_up  = cut_plane(x, A, b, Aeq, beq, ind_con, ind_int, indeq_con, indeq_int, num_int)

        # check answer against expected results
        self.assertTrue(np.allclose(A_up, expected['A_up']),
            msg='\n' + str(A_up) + '\n' + str(expected['A_up']))
        self.assertTrue(np.allclose(b_up, expected['b_up']),
            msg='\n' + str(b_up) + '\n' + str(expected['b_up']))

    def test_problem2(self):
        """ another test problem
        """

        # input arguments
        x = np.array([
            [0.],
            [7.],
            [2.0561],
            [2.4590],
            [0.],
            [0.],
            [0.],
            [700.],
            [220.],
            [300.],
            [0.],
            [0.],

        ])
        A = np.array([
            [         0,         0,         0,         0,         0,         0,    1.0000,         0,         0,    1.0000,         0,         0],
            [         0,         0,         0,         0,         0,         0,         0,    1.0000,         0,         0,    1.0000,         0],
            [         0,         0,         0,         0,         0,         0,         0,         0,    1.0000,         0,         0,    1.0000],
            [         0,         0,         0,         0,         0,         0,   -1.0000,         0,         0,   -1.0000,         0,         0],
            [         0,         0,         0,         0,         0,         0,         0,   -1.0000,         0,         0,   -1.0000,         0],
            [         0,         0,         0,         0,         0,         0,         0,         0,   -1.0000,         0,         0,   -1.0000],
            [   10.4653,    8.3129,    6.1531,         0,         0,         0,         0,         0,         0,         0,         0,         0],
            [         0,         0,         0,   10.4460,    8.2998,    6.1460,         0,         0,         0,         0,         0,         0],
            [ -107.0000,         0,         0,         0,         0,         0,    1.0000,         0,         0,         0,         0,         0],
            [         0, -107.0000,         0,         0,         0,         0,         0,    1.0000,         0,         0,         0,         0],
            [         0,         0, -107.0000,         0,         0,         0,         0,         0,    1.0000,         0,         0,         0],
            [         0,         0,         0, -122.0000,         0,         0,         0,         0,         0,    1.0000,         0,         0],
            [         0,         0,         0,         0, -122.0000,         0,         0,         0,         0,         0,    1.0000,         0],
            [         0,         0,         0,         0,         0, -122.0000,         0,         0,         0,         0,         0,    1.0000],
            [         0,   -1.0000,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0]
        ])
        b = np.array([
            [   300.],
            [   700.],
            [   220.],
            [   -60.],
            [  -140.],
            [   -44.],
            [    72.],
            [    48.],
            [     0.],
            [     0.],
            [     0.],
            [     0.],
            [     0.],
            [     0.],
            [    -7.]
        ])
        Aeq = np.array([])
        beq = np.array([])

        ind_con = np.array([1,     2,     3,     4,     5,     6]) - 1  # indices of con rows
        ind_int = np.array([7,     8,     9,    10,    11,    12,    13,    14]) - 1  # indices of int rows

        indeq_con = np.array([])
        indeq_int = np.array([])

        num_int = 6

        # expected results (from MATLAB)
        expected = {
            'A_up': np.array([
                [         0,         0,         0,         0,         0,         0,    1.0000,         0,        0,    1.0000,         0,         0],
                [         0,         0,         0,         0,         0,         0,         0,    1.0000,        0,         0,    1.0000,         0],
                [         0,         0,         0,         0,         0,         0,         0,         0,   1.0000,         0,         0,    1.0000],
                [         0,         0,         0,         0,         0,         0,   -1.0000,         0,        0,   -1.0000,         0,         0],
                [         0,         0,         0,         0,         0,         0,         0,   -1.0000,        0,         0,   -1.0000,         0],
                [         0,         0,         0,         0,         0,         0,         0,         0,  -1.0000,         0,         0,   -1.0000],
                [   10.4653,    8.3129,    6.1531,         0,         0,         0,         0,         0,        0,         0,         0,         0],
                [         0,         0,         0,   10.4460,    8.2998,    6.1460,         0,         0,        0,         0,         0,         0],
                [ -107.0000,         0,         0,         0,         0,         0,    1.0000,         0,        0,         0,         0,         0],
                [         0, -107.0000,         0,         0,         0,         0,         0,    1.0000,        0,         0,         0,         0],
                [         0,         0, -107.0000,         0,         0,         0,         0,         0,   1.0000,         0,         0,         0],
                [         0,         0,         0, -122.0000,         0,         0,         0,         0,        0,    1.0000,         0,         0],
                [         0,         0,         0,         0, -122.0000,         0,         0,         0,        0,         0,    1.0000,         0],
                [         0,         0,         0,         0,         0, -122.0000,         0,         0,        0,         0,         0,    1.0000],
                [         0,   -1.0000,         0,         0,         0,         0,         0,         0,        0,         0,         0,         0],
                [  -97.0000,   -1.0000,         0, -122.0000,         0,         0,    1.0000,    1.0000,   1.0000,    1.0000,    1.0000,    1.0000]
            ]),
            'b_up': np.array([
                [300],
                [700],
                [220],
                [-60],
                [-140],
                [-44],
                [72],
                [48],
                [ 0],
                [ 0],
                [ 0],
                [ 0],
                [ 0],
                [ 0],
                [-7],
                [924]
            ]),
        }

        # call the function
        A_up, b_up  = cut_plane(x, A, b, Aeq, beq, ind_con, ind_int, indeq_con, indeq_int, num_int)

        # check answer against expected results
        self.assertTrue(np.allclose(A_up, expected['A_up']),
            msg='\n' + str(A_up) + '\n' + str(expected['A_up']))
        self.assertTrue(np.allclose(b_up, expected['b_up']),
            msg='\n' + str(b_up) + '\n' + str(expected['b_up']))


class BranchCutTestCase(unittest.TestCase):
    """ test the branch_cut function
    """

    def test_3routes(self):
        data = Dataset(suffix='after_3routes')

        # linear objective coefficients
        objective = get_objective(data)
        f_int = objective[0]    # integer type design variables
        f_con = objective[1]    # continuous type design variables

        # coefficient matrix for linear inequality constraints, Ax <= b
        constraints = get_constraints(data)
        A = constraints[0]
        b = constraints[1]

        Aeq = np.ndarray(shape=(0, 0))
        beq = np.ndarray(shape=(0, 0))

        J = data.inputs.DVector.shape[0]  # number of routes
        K = len(data.inputs.AvailPax)     # number of aircraft types

        # lower and upper bounds
        lb = np.zeros((2*K*J, 1))
        ub = np.concatenate((
            np.ones((K*J, 1)) * data.inputs.MaxTrip.reshape(-1, 1),
            np.ones((K*J, 1)) * np.inf
        ))

        # indices into A matrix for continuous & integer/continuous variables
        ind_conCon = range(2*J)
        ind_intCon = range(2*J, len(constraints[0]))

        # call the branch and cut algorithm to solve the MILP problem
        xopt, fopt, can_x, can_F, x_best_relax, f_best_relax, funCall, eflag = \
            branch_cut(f_int, f_con, A, b, Aeq, beq, lb, ub,
                       ind_conCon, ind_intCon, [], [])

        # TODO: check return values against MATLAB results


class OutputTestCase(unittest.TestCase):
    """ test the output function
    """
    def compare(self, dataset1, dataset2):
        # check that the dataset matches the output dataset

        self.assertTrue(np.allclose(dataset1.outputs.Cost, dataset2.outputs.Cost),
            msg='\n' + str(dataset1.outputs.Cost) + '\n' + str(dataset2.outputs.Cost))

        self.assertTrue(np.allclose(dataset1.coefficients.PPNM, dataset2.coefficients.PPNM),
            msg='\n' + str(dataset1.coefficients.PPNM) + '\n' + str(dataset2.coefficients.PPNM))

        self.assertTrue(np.allclose(dataset1.coefficients.Profit, dataset2.coefficients.Profit),
            msg='\n' + str(dataset1.coefficients.Profit) + '\n' + str(dataset2.coefficients.Profit))

        self.assertTrue(np.allclose(dataset1.coefficients.Info, dataset2.coefficients.Info),
            msg='\n' + str(dataset1.coefficients.Info) + '\n' + str(dataset2.coefficients.Info))

        # TODO: compare Trips, FleetUsed, Fuel, Doc, BlockTime, Nox, Maxpax, Pax, Miles
        #               CostDetail, RevDetail, PaxDetail, RevArray, CostArray, PaxArray, ProfitArray, Revenue

    def test_3routes(self):
        dataset = Dataset(suffix='before_3routes')
        dataout = Dataset(suffix='after_3routes')   # TODO: this dataset is probably not post-solution, check it

        xopt = np.array([0, 3, 2, 2, 3, 0, 0, 321, 214, 244, 366, 0])
        fopt = -1.9417e+04

        generate_outputs(xopt, fopt, dataset)

        self.compare(dataset, dataout)


if __name__ == "__main__":
    unittest.main()
