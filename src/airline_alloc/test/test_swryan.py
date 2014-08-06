
import os
import unittest

import numpy as np
from scipy.io import loadmat


def load_data_file(file_name):
    #../../../MATLAB/Data/<data_file>
    file_path = os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        os.path.pardir,
        os.path.pardir,
        'MATLAB',
        'Data',
        file_name)

    return loadmat(file_path, squeeze_me=True, struct_as_record=False)


def range_extract(RVector, distance):
    """ find the closest match in RVector for each value in distance
        returns an index into RVector for each value in distance
    """
    indices = np.zeros(len(distance))

    for cc in xrange(len(distance)):
        dist = distance[cc]
        diff_min = np.inf

        for r_id, r_dist in np.ndenumerate(RVector):
            diff = np.abs(r_dist - dist)
            if diff < diff_min:
                indices[cc] = r_id[0]
                diff_min = diff

    return indices


def get_objective(inputs, outputs, constants, coefficients):
    """ generate the objective matrix for linprog
        returns the coefficients for the integer and continuous design variables
    """
    J = inputs.DVector.shape[0]  # number of routes
    K = len(inputs.AvailPax)     # number of aircraft types
    KJ = K*J

    fuelburn  = coefficients.Fuelburn
    docnofuel = coefficients.Doc
    price     = outputs.TicketPrice
    fuelcost  = constants.FuelCost

    obj_int = np.zeros((KJ, 1))
    obj_con = np.zeros((KJ, 1))

    for kk in xrange(K):
        for jj in xrange(J):
            col = kk*J + jj
            obj_int[col] = docnofuel[kk, jj] + fuelcost * fuelburn[kk, jj]
            obj_con[col] = -price[kk, jj]

    return obj_int.flatten(), obj_con.flatten()


def get_constraints(inputs, constants, coefficients):
    """ generate the constraint matrix/vector for linprog
    """

    J = inputs.DVector.shape[0]  # number of routes
    K = len(inputs.AvailPax)     # number of aircraft types
    KJ  = K*J
    KJ2 = KJ*2

    dem   = inputs.DVector[:, 1].reshape(-1, 1)
    BH    = coefficients.BlockTime
    MH    = constants.MH.reshape(-1, 1)
    cap   = inputs.AvailPax.flatten()
    fleet = inputs.ACNum.reshape(-1, 1)
    t     = inputs.TurnAround

    # Upper demand constraint
    A1 = np.zeros((J, KJ2))
    b1 = dem.copy()
    for jj in xrange(J):
        for kk in xrange(K):
            col = K*J + kk*J + jj
            A1[jj, col] = 1

    # Lower demand constraint
    A2 = np.zeros((J, KJ2))
    b2 = -0.2 * dem
    for jj in xrange(J):
        for kk in xrange(K):
            col = K*J + kk*J + jj
            A2[jj, col] = -1

    # Aircraft utilization constraint
    A3 = np.zeros((K, KJ2))
    b3 = np.zeros((K, 1))
    for kk in xrange(K):
        for jj in xrange(J):
            col = kk*J + jj
            A3[kk, col] = BH[kk, jj]*(1 + MH[kk, 0]) + t
        b3[kk, 0] = 12*fleet[kk]

    # Aircraft capacity constraint
    A4 = np.zeros((KJ, KJ2))
    b4 = np.zeros((KJ, 1))
    rw = 0
    for kk in xrange(K):
        for jj in xrange(J):
            col1 = kk*J + jj
            A4[rw, col1] = 0.-cap[kk]
            col2 = K*J + kk*J + jj
            A4[rw, col2] = 1
            rw = rw + 1

    A = np.concatenate((A1, A2, A3, A4))
    b = np.concatenate((b1, b2, b3, b4))

    return A, b


class RangeExtractTestCase(unittest.TestCase):

    def test_3routes(self):
        inputs = load_data_file('inputs_before_3routes.mat')['Inputs']

        distance = [2000, 1500, 1000]

        indices = range_extract(inputs.RVector, distance)

        expected = np.array([80, 1782, 674])

        self.assertTrue(np.allclose(indices, expected-1))  # zero indexing

    def test_11routes(self):
        inputs = load_data_file('inputs_before_11routes.mat')['Inputs']

        distance = [162, 753, 974, 1094, 1357, 1455, 2169, 2249, 2269, 2337, 2350]

        indices = range_extract(inputs.RVector, distance)

        expected = np.array([
            394, 1598, 410, 2042, 615, 742, 427, 1501, 1308, 414, 1317
        ])

        self.assertTrue(np.allclose(indices, expected-1))  # zero indexing

    def test_31routes(self):
        inputs = load_data_file('inputs_before_31routes.mat')['Inputs']

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
        inputs       = load_data_file('inputs_after_3routes.mat')['Inputs']
        outputs      = load_data_file('outputs_after_3routes.mat')['Outputs']
        constants    = load_data_file('constants_after_3routes.mat')['Constants']
        coefficients = load_data_file('coefficients_after_3routes.mat')['Coefficients']

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
        inputs       = load_data_file('inputs_after_3routes.mat')['Inputs']
        constants    = load_data_file('constants_after_3routes.mat')['Constants']
        coefficients = load_data_file('coefficients_after_3routes.mat')['Coefficients']

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


if __name__ == "__main__":
    unittest.main()
