"""
    dataset.py

    a wrapper class and functions for extracting data from the MATLAB dataset:
    NASA_LEARN_AirlineAllocation_Branch_Cut/Data/Dataset.mat

    per Satadru Roy:
    This dataset has all 2134 US network related data. It also has the
    performance and cost data for 18 different types of aircraft we
    studied in one of our early NASA effort.
"""

from os.path import dirname, pardir, join
from scipy.io import loadmat

import numpy as np

data_path = join(dirname(__file__), pardir, pardir, 'MATLAB', 'Data')


class Dataset(object):
    """ convenience wrapper for managing a dataset

        a dataset consists of four separate data structures:
            Inputs, Outputs, Constants, Constraints

        the dataset can be filtered to include selected aircraft and routes
    """
    def __init__(self, file_name=None, suffix=None):
        if file_name is not None:
            mat = load_data(file_name)
            self.inputs       = mat['Inputs']
            self.outputs      = mat['Outputs']
            self.constants    = mat['Constants']
            self.coefficients = mat['Coefficients']
        elif suffix is not None:
            # loads data from four files with the naming convention:
            # "key_suffix.mat", where key is the lower case name of
            # one of the four structs and suffix is arbitrary
            for key in ['Inputs', 'Outputs', 'Constants', 'Coefficients']:
                file_name = key.lower() + '_'+suffix+'.mat'
                mat = load_data(file_name)
                setattr(self, key.lower(), mat[key])
        else:
            self.inputs = None
            self.outputs = None
            self.constants = None
            self.coefficients = None

    def filter(self, ac_ind=[], ac_num=[], distance=[], dvector=[], add_trip=0):
        """ filters the dataset to include only the specified aircraft and routes

            arguments:
                ac_ind          the indices of the aircraft to select
                ac_num          the number of each aircraft
                distance        the route distances to select
                dvector         the route demand
        """
        route_ind = range_extract(self.inputs.RVector, distance)

        self.inputs.RVector    = self.inputs.RVector[route_ind]
        self.inputs.DVector    = dvector
        self.inputs.AvailPax   = self.inputs.AvailPax[ac_ind]
        self.inputs.ACNum      = np.array(ac_num)
        self.inputs.TurnAround = 1

        K = len(self.inputs.AvailPax)        # number of aircraft types
        J = len(self.inputs.DVector[:, 1])   # number of routes
        self.inputs.Lim = np.ones((K, J))

        self.constants.Runway   = 1e4 * len(self.inputs.RVector)
        self.constants.MH       = self.constants.MH[ac_ind]
        self.constants.FuelCost = 0.2431
        self.constants.demfac   = 1

        self.outputs.TicketPrice    = filter_data(self.outputs.TicketPrice,    ac_ind, route_ind)

        self.coefficients.Fuelburn  = filter_data(self.coefficients.Fuelburn,  ac_ind, route_ind)
        self.coefficients.Doc       = filter_data(self.coefficients.Doc,       ac_ind, route_ind)
        self.coefficients.Nox       = filter_data(self.coefficients.Nox,       ac_ind, route_ind)
        self.coefficients.BlockTime = filter_data(self.coefficients.BlockTime, ac_ind, route_ind)

        rw = 0
        max_trip = np.zeros(K*J)
        for kk in range(K):
            for jj in range(J):
                max_trip[rw] = self.inputs.ACNum[kk] \
                             * np.ceil(12./(self.coefficients.BlockTime[kk, jj] * (1 + self.constants.MH[kk]) + self.inputs.TurnAround)) \
                             + add_trip
                rw = rw + 1

        self.inputs.MaxTrip = max_trip


def load_data(file_name):
    """ load MATLAB dataset
    """
    return loadmat(join(data_path, file_name),
                   squeeze_me=True, struct_as_record=False)


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

    return indices.astype(int)


def filter_data(data, ac_ind, route_ind):
    """ filter a 2-dimensional array of aircraft/route data to include
        only the specified aircraft and routes
    """
    filtered_data = np.array([])

    for kk in xrange(len(ac_ind)):
        tmp = np.array([])

        for jj in xrange(len(route_ind)):
            tmp = np.append(tmp, data[ac_ind[kk], route_ind[jj]])

        filtered_data = np.append(filtered_data, tmp)

    return filtered_data.reshape(len(ac_ind), -1)


if __name__ == "__main__":
    # check against "OverrideFunction_11routes.m"
    dataset  = Dataset(suffix='before_11routes')
    filtered = Dataset(suffix='after_11routes')

    ac_ind    = np.array([ 9, 10]) - 1
    ac_num    = np.array([12,  8])
    distance  = [162, 753, 974, 1094, 1357, 1455, 2169, 2249, 2269, 2337, 2350]
    dvector   = np.array([
        [1,   41],
        [2, 1009],
        [3,   89],
        [4,  661],
        [5, 1041],
        [6,  358],
        [7,  146],
        [8,   97],
        [9,  447],
        [10, 194],
        [11, 263]
    ])

    dataset.filter(ac_ind, ac_num, distance, dvector, add_trip=1)
    # self.compare(dataset, filtered)
