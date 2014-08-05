
import os
import unittest

import numpy as np
from scipy.io import loadmat

from runtime import *


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
    ind = np.zeros(len(distance))

    for cc in xrange(len(distance)):
        _range = distance[cc]
        diff_min = np.inf

        for ii, ii_value in np.ndenumerate(RVector):
            diff = np.abs(ii_value - _range)
            if diff < diff_min:
                ind[cc] = ii[0]
                diff_min = diff

    return ind


def get_objective(Inputs, Outputs, Constants, Coefficients):
    """ generate the objective matrix for linprog
        returns the coefficients for the integer and continuous design variables
    """
    J = length_(Inputs.DVector(arange_(), 2))
    K = length_(Inputs.AvailPax)

    fuelburn  = Coefficients.Fuelburn
    docnofuel = Coefficients.Doc
    price     = Outputs.TicketPrice
    fuelcost  = Constants.FuelCost

    obj_int = np.zeros(K*J, 1)
    obj_con = np.zeros(K*J, 1)

    for kk in arange_(1, K):
        for jj in arange_(1, J):
            col = (kk-1) * J + jj
            obj_int[col] = docnofuel[kk, jj] + fuelcost * fuelburn[kk, jj]
            obj_con[col] = -price[kk, jj]

    return [obj_int, obj_con]


def get_constraints(Inputs, Constants, Coefficients):
    """ generate the constraint matrix/vector for linprog
    """
    J = length_(Inputs.DVector(arange_(), 2))
    K = length_(Inputs.AvailPax)

    dem   = Inputs.DVector(arange_(), 2)
    BH    = Coefficients.BlockTime
    MH    = Constants.MH
    cap   = Inputs.AvailPax
    fleet = Inputs.ACNum
    t     = Inputs.TurnAround

    # Upper demand constraint
    A1 = np.zeros(J, 2*K*J)
    b1 = copy_(dem)
    for jj in arange_(1, J):
        for kk in arange_(1, K):
            col = K*J + (kk-1)*J + jj
            A1[jj, col] = 1

    # Lower demand constraint
    A2 = np.zeros(J, 2*K*J)
    b2 = -0.2 * dem
    for jj in arange_(1, J):
        for kk in arange_(1, K):
            col = K*J + (kk-1)*J + jj
            A2[jj, col] = -1

    # Aircraft utilization constraint
    A3 = np.zeros(K, K*J*2)
    b3 = np.zeros(K, 1)
    for kk in arange_(1, K):
        for jj in arange_(1, J):
            col = (kk-1)*J + jj
            A3[kk, col] = BH[kk, jj]*(1 + MH[kk, 1]) + t
        b3[kk, 1] = 12*fleet[kk]

    # Aircraft capacity constraint
    A4 = np.zeros(K*J, K*J*2)
    b4 = np.zeros(K*J, 1)
    rw = 1
    for kk in arange_(1, K):
        for jj in arange_(1, J):
            col1 = (kk-1)*J + jj
            A4[rw, col1] = -cap[kk]
            col2 = K*J + (kk-1)*J + jj
            A4[rw, col2] = 1
            rw = rw + 1

    A = matlabarray([[A1], [A2], [A3], [A4]])
    b = matlabarray([[b1], [b2], [b3], [b4]])

    return A, b


class MyTestCase(unittest.TestCase):

    def test_range_extract(self):
        pass

    def test_objective(self):
        pass

    def test_constraints(self):
        pass

if __name__ == "__main__":
    unittest.main()
