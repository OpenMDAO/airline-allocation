
import unittest

import numpy as np

from airline_alloc.dataset import *


class RangeExtractTestCase(unittest.TestCase):
    """ test the range_extract function
    """

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


class FilterDataTestCase(unittest.TestCase):
    """ test the filter_data function
    """

    def check_filtered(self, ac_ind, route_ind):
        # TicketPrice
        filtered = filter_data(self.outputs.TicketPrice, ac_ind, route_ind)
        self.assertTrue(np.allclose(filtered, self.outputs_filtered.TicketPrice))

        # Fuelburn
        filtered = filter_data(self.coefficients.Fuelburn, ac_ind, route_ind)
        self.assertTrue(np.allclose(filtered, self.coefficients_filtered.Fuelburn))

        # Doc
        filtered = filter_data(self.coefficients.Doc, ac_ind, route_ind)
        self.assertTrue(np.allclose(filtered, self.coefficients_filtered.Doc))

        # Nox
        filtered = filter_data(self.coefficients.Nox, ac_ind, route_ind)
        self.assertTrue(np.allclose(filtered, self.coefficients_filtered.Nox))

        # BlockTime
        filtered = filter_data(self.coefficients.BlockTime, ac_ind, route_ind)
        self.assertTrue(np.allclose(filtered, self.coefficients_filtered.BlockTime))

    def test_3routes(self):
        self.outputs      = load_data('outputs_before_3routes.mat')['Outputs']
        self.coefficients = load_data('coefficients_before_3routes.mat')['Coefficients']

        self.outputs_filtered      = load_data('outputs_after_3routes.mat')['Outputs']
        self.coefficients_filtered = load_data('coefficients_after_3routes.mat')['Coefficients']

        route_ind = np.array([80, 1782, 674]) - 1
        ac_ind    = np.array([9, 10]) - 1

        self.check_filtered(ac_ind, route_ind)

    def test_11routes(self):
        self.outputs      = load_data('outputs_before_11routes.mat')['Outputs']
        self.coefficients = load_data('coefficients_before_11routes.mat')['Coefficients']

        self.outputs_filtered      = load_data('outputs_after_11routes.mat')['Outputs']
        self.coefficients_filtered = load_data('coefficients_after_11routes.mat')['Coefficients']

        route_ind = np.array([394, 1598, 410, 2042, 615, 742, 427, 1501, 1308, 414, 1317]) - 1
        ac_ind = np.array([9, 10]) - 1

        self.check_filtered(ac_ind, route_ind)

    def test_31routes(self):
        self.outputs      = load_data('outputs_before_31routes.mat')['Outputs']
        self.coefficients = load_data('coefficients_before_31routes.mat')['Coefficients']

        self.outputs_filtered      = load_data('outputs_after_31routes.mat')['Outputs']
        self.coefficients_filtered = load_data('coefficients_after_31routes.mat')['Coefficients']

        route_ind = np.array([
            1483, 1045, 1944, 1856,  754, 1463, 1718,  948,  416, 1801,
            1795, 225,   845,  919,  840, 1746, 1739, 1797, 1947, 1987,
            1429, 1802, 2060, 1897, 1410, 1241, 971,  1399, 1597, 2028,
            1433
        ]) - 1
        ac_ind = np.array([6, 10, 4, 9, 3, 8]) - 1

        self.check_filtered(ac_ind, route_ind)


class DatasetFilterTestCase(unittest.TestCase):
    """ test the dataset filter function
    """

    def compare(self, dataset1, dataset2):
        # check that the dataset matches the filtered dataset

        self.assertTrue(np.allclose(dataset1.outputs.TicketPrice,    dataset2.outputs.TicketPrice),
            msg='\n' + str(dataset1.outputs.TicketPrice) + '\n' + str(dataset2.outputs.TicketPrice))

        self.assertTrue(np.allclose(dataset1.coefficients.Fuelburn,  dataset2.coefficients.Fuelburn),
            msg='\n' + str(dataset1.coefficients.Fuelburn) + '\n' + str(dataset2.coefficients.Fuelburn))

        self.assertTrue(np.allclose(dataset1.coefficients.Doc,       dataset2.coefficients.Doc),
            msg='\n' + str(dataset1.coefficients.Doc) + '\n' + str(dataset2.coefficients.Doc))

        self.assertTrue(np.allclose(dataset1.coefficients.Nox,       dataset2.coefficients.Nox),
            msg='\n' + str(dataset1.coefficients.Nox) + '\n' + str(dataset2.coefficients.Nox))

        self.assertTrue(np.allclose(dataset1.coefficients.BlockTime, dataset2.coefficients.BlockTime),
            msg='\n' + str(dataset1.coefficients.BlockTime) + '\n' + str(dataset2.coefficients.BlockTime))

        self.assertTrue(np.allclose(dataset1.inputs.MaxTrip, dataset2.inputs.MaxTrip),
            msg='\n' + str(dataset1.inputs.MaxTrip) + '\n' + str(dataset2.inputs.MaxTrip))

    def test_3routes(self):
        # check against "OverrideFunction_3routes.m"
        dataset  = Dataset(suffix='before_3routes')
        filtered = Dataset(suffix='after_3routes')

        ac_ind    = np.array([9, 10]) - 1
        ac_num    = np.array([6,  4])
        distance  = [2000, 1500, 1000]
        dvector   = np.array([
            [1, 300],
            [2, 700],
            [3, 220]
        ])

        dataset.filter(ac_ind, ac_num, distance, dvector)
        self.compare(dataset, filtered)

    def test_11routes(self):
        # check against "OverrideFunction_11routes.m"
        dataset  = Dataset(suffix='before_11routes')
        filtered = Dataset(suffix='after_11routes')

        ac_ind    = np.array([ 9, 10]) - 1
        ac_num    = np.array([12,  8])
        distance  = [163, 753, 974, 1094, 1357, 1455, 2169, 2249, 2269, 2337, 2350]
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
        self.compare(dataset, filtered)

    def test_31routes(self):
        # check against "OverrideFunction_31routes.m"
        dataset  = Dataset(suffix='before_31routes')
        filtered = Dataset(suffix='after_31routes')

        ac_ind    = np.array([6, 10, 4, 9,  3, 8]) - 1
        ac_num    = np.array([1,  7, 2, 8, 19, 1])
        distance  = [
            113, 174, 289, 303, 324, 331,  342,  375,  407, 427,
            484, 486, 531, 543, 550, 570,  594,  609,  622, 680,
            747, 758, 760, 823, 837, 991, 1098, 1231, 1407, 1570, 1626
        ]
        dvector   = np.array([
            [ 1,   99],
            [ 2,   80],
            [ 3,   51],
            [ 4,  184],
            [ 5,  263],
            [ 6,  169],
            [ 7,  158],
            [ 8,  135],
            [ 9,  284],
            [10,  184],
            [11,   92],
            [12,  132],
            [13,  754],
            [14,  150],
            [15,  238],
            [16,  264],
            [17,  365],
            [18,  749],
            [19,  234],
            [20,   50],
            [21,  101],
            [22,  124],
            [23,  175],
            [24,  221],
            [25, 1258],
            [26,  105],
            [27,  112],
            [28,  129],
            [29,  506],
            [30,  134],
            [31,  266],
        ])

        dataset.filter(ac_ind, ac_num, distance, dvector)
        self.compare(dataset, filtered)


if __name__ == "__main__":
    unittest.main()
