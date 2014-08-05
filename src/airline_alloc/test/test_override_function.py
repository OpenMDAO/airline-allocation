
import os
import unittest

import numpy as np
from scipy.io import loadmat

from airline_alloc.override_functions import *

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
            
class RangeExtractTestCase(unittest.TestCase):
    def test_3routes(self):
        mat_file = load_data_file('inputs_before_3routes.mat')
        mat_file = mat_file['Inputs']
        
        comp = RangeExtract()
        comp.RVector = mat_file.RVector
        comp.distance = np.array([2000, 1500, 1000])
        
        comp.execute()
        print comp.ind
        
        self.assertTrue(np.allclose(comp.ind, np.array([80, 1782, 674]) - 1))
        
    def test_11routes(self):
        mat_file = load_data_file('inputs_before_11routes.mat')
        mat_file = mat_file['Inputs']
        
        comp = RangeExtract()
        comp.RVector = mat_file.RVector
        comp.distance = np.array([
            162,753,974,1094,
            1357,1455,2169,2249,
            2269,2337,2350])

        comp.execute()
        self.assertTrue(np.allclose(comp.ind, np.array([
            394,1598,410,2042,
            615,742,427,1501,
            1308,414,1317]) - 1))
        
    def test_31routes(self):
        mat_file = load_data_file('inputs_before_31routes.mat')
        mat_file = mat_file['Inputs']
        
        comp = RangeExtract()
        comp.RVector = mat_file.RVector
        comp.distance = np.array([
            113,174,289,303,
            324,331,342,375,
            407,427,484,486,
            531,543,550,570,
            594,609,622,680,
            747,758,760,823,
            837,991,1098,1231,
            1407,1570,1626])

        comp.execute()
        self.assertTrue(np.allclose(comp.ind, np.array([
            1483,1045,1944,1856,
            754,1463,1718,948,
            416,1801,1795,225,
            845,919,840,1746,
            1739,1797,1947,1987,
            1429,1802,2060,1897,
            1410,1241,971,1399,
            1597,2028,1433]) - 1))
        
class InitializationTestCase(unittest.TestCase):
        
    def test_init_3routes(self):
        inputs = load_data_file('inputs_before_3routes.mat')
        inputs = inputs['Inputs']
        
        constants = load_data_file('constants_before_3routes.mat')
        constants = constants['Constants']
        
        comp = Initialization()
        comp.ac_ind = np.array([9, 10]) - 1
        comp.distance = np.array([2000, 1500, 1000])
        comp.DVector = np.array([[1,2,3],[300, 700, 200]]).T
        comp.ACNum = np.array([6,4])
        comp.RVector_in = inputs.RVector
        comp.AvailPax_in = inputs.AvailPax
        comp.route_ind = np.array([80,1782,674]) - 1
        comp.MH_in = constants.MH
        
        comp.execute()
        
        inputs = load_data_file('inputs_after_3routes.mat')
        inputs = inputs['Inputs']
        
        constants = load_data_file('constants_after_3routes.mat')
        constants = constants['Constants']
        
        self.assertTrue(np.allclose(comp.RVector_out, inputs.RVector))
        self.assertTrue(np.allclose(comp.AvailPax_out, inputs.AvailPax))

        self.assertTrue(np.allclose(comp.TurnAround, inputs.TurnAround))

        self.assertTrue(np.allclose(comp.J, 3))
        self.assertTrue(np.allclose(comp.K, 2))
        self.assertTrue(np.allclose(comp.Lim, inputs.Lim))
        
    def test_init_11routes(self):
        inputs = load_data_file('inputs_before_3routes.mat')
        inputs = inputs['Inputs']
        
        constants = load_data_file('constants_before_3routes.mat')
        constants = constants['Constants']
        
        comp = Initialization()
        comp.ac_ind = np.array([9, 10]) - 1
        comp.distance = np.array([
            162,753,974,1094,
            1357,1455,2169,2249,
            2269,2337,2350])
        comp.DVector = np.array([
            np.arange(1,12),[
                41,1009,89,661,
                1041,358,146,97,
                447,194,263]]).T
        comp.ACNum = np.array([12,8])
        comp.RVector_in = inputs.RVector
        comp.AvailPax_in = inputs.AvailPax
        comp.route_ind = np.array([
            394,1598,410,2042,
            615,742,427,1501,
            1308,414,1317]) - 1
        comp.MH_in = constants.MH
        
        print repr(comp.DVector)
        comp.execute()
        
        inputs = load_data_file('inputs_after_11routes.mat')
        inputs = inputs['Inputs']
        
        constants = load_data_file('constants_after_11routes.mat')
        constants = constants['Constants']
        
        self.assertTrue(np.allclose(comp.RVector_out, inputs.RVector))
        self.assertTrue(np.allclose(comp.AvailPax_out, inputs.AvailPax))
        self.assertTrue(np.allclose(comp.TurnAround, inputs.TurnAround))
        self.assertTrue(np.allclose(comp.J, 11))
        self.assertTrue(np.allclose(comp.K, 2))
        self.assertTrue(np.allclose(comp.Lim, inputs.Lim))
        
    def test_init_31routes(self):
        inputs = load_data_file('inputs_before_31routes.mat')
        inputs = inputs['Inputs']
        
        constants = load_data_file('constants_before_31routes.mat')
        constants = constants['Constants']
        
        comp = Initialization()
        comp.ac_ind = np.array([6,10,4,9,3,8]) - 1
        comp.distance = np.array([
            113,174,289,303,
            324,331,342,375,
            407,427,484,486,
            531,543,550,570,
            594,609,622,680,
            747,758,760,823,
            837,991,1098,1231,
            1407,1570,1626])
        comp.DVector = np.array([
            np.arange(1,32),[
                99,80,51,184,
                263,169,158,135,
                284,184,92,132,
                754,150,238,264,
                365,749,234,50,
                101,124,175,221,
                258,105,112,129,
                506,134,266]]).T
        comp.ACNum = np.array([1,7,2,8,19,1])
        comp.RVector_in = inputs.RVector
        comp.AvailPax_in = inputs.AvailPax
        comp.route_ind = np.array([
            1483,1045,1944,1856,
            754,1463,1718,948,
            416,1801,1795,225,
            845,919,840,1746,
            1739,1797,1947,1987,
            1429,1802,2060,1897,
            1410,1241,971,1399,
            1597,2028,1433]) - 1
            
        comp.MH_in = constants.MH
        
        comp.execute()
        
        inputs = load_data_file('inputs_after_31routes.mat')
        inputs = inputs['Inputs']
        
        constants = load_data_file('constants_after_31routes.mat')
        constants = constants['Constants']
        
        self.assertTrue(np.allclose(comp.RVector_out, inputs.RVector))
        self.assertTrue(np.allclose(comp.AvailPax_out, inputs.AvailPax))
        self.assertTrue(np.allclose(comp.TurnAround, inputs.TurnAround))
        self.assertTrue(np.allclose(comp.J, 31))
        self.assertTrue(np.allclose(comp.K, 6))
        self.assertTrue(np.allclose(comp.Lim, inputs.Lim))
        
class ArrayFilterTestCase(unittest.TestCase):
    def test_3routes(self):
        outputs_before = load_data_file('outputs_before_3routes.mat')
        outputs_before = outputs_before['Outputs']
        
        coefficients_before = load_data_file('coefficients_before_3routes.mat')
        coefficients_before = coefficients_before['Coefficients'] 
        
        outputs_after = load_data_file('outputs_after_3routes.mat')
        outputs_after = outputs_after['Outputs']
        
        coefficients_after = load_data_file('coefficients_after_3routes.mat')
        coefficients_after = coefficients_after['Coefficients'] 
        
        comp = ArrayFilter()
        comp.route_ind = np.array([80,1782,674]) - 1
        comp.ac_ind = np.array([9,10]) - 1
        
        #TicketPrice
        comp.original = outputs_before.TicketPrice
        comp.execute()
        
        print comp.filtered
        print outputs_after.TicketPrice
        self.assertTrue(np.allclose(comp.filtered, outputs_after.TicketPrice))
        
        #Fuelburn
        comp.original = coefficients_before.Fuelburn
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, coefficients_after.Fuelburn))
        
        #Doc
        comp.original = coefficients_before.Doc
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, coefficients_after.Doc))
        
        #Nox
        comp.original = coefficients_before.Nox
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, coefficients_after.Nox))
        
        #BlockTime
        comp.original = coefficients_before.BlockTime
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, coefficients_after.BlockTime))
        
    def test_11routes(self):
        outputs_before = load_data_file('outputs_before_11routes.mat')
        outputs_before = outputs_before['Outputs']
        
        coefficients_before = load_data_file('coefficients_before_11routes.mat')
        coefficients_before = coefficients_before['Coefficients'] 
        
        outputs_after = load_data_file('outputs_after_11routes.mat')
        outputs_after = outputs_after['Outputs']
        
        coefficients_after = load_data_file('coefficients_after_11routes.mat')
        coefficients_after = coefficients_after['Coefficients'] 
        
        comp = ArrayFilter()
        comp.route_ind = np.array([
            394,1598,410,2042,
            615,742,427,1501,
            1308,414,1317]) - 1
        comp.ac_ind = np.array([9,10]) - 1
        
        #TicketPrice
        comp.original = outputs_before.TicketPrice
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, outputs_after.TicketPrice))
        
        #Fuelburn
        comp.original = coefficients_before.Fuelburn
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, coefficients_after.Fuelburn))
        
        #Doc
        comp.original = coefficients_before.Doc
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, coefficients_after.Doc))
        
        #Nox
        comp.original = coefficients_before.Nox
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, coefficients_after.Nox))
        
        #BlockTime
        comp.original = coefficients_before.BlockTime
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, coefficients_after.BlockTime))
        
    def test_31routes(self):
        outputs_before = load_data_file('outputs_before_31routes.mat')
        outputs_before = outputs_before['Outputs']
        
        coefficients_before = load_data_file('coefficients_before_31routes.mat')
        coefficients_before = coefficients_before['Coefficients'] 
        
        outputs_after = load_data_file('outputs_after_31routes.mat')
        outputs_after = outputs_after['Outputs']
        
        coefficients_after = load_data_file('coefficients_after_31routes.mat')
        coefficients_after = coefficients_after['Coefficients'] 
        
        comp = ArrayFilter()
        comp.route_ind = np.array([
            1483,1045,1944,1856,
            754,1463,1718,948,
            416,1801,1795,225,
            845,919,840,1746,
            1739,1797,1947,1987,
            1429,1802,2060,1897,
            1410,1241,971,1399,
            1597,2028,1433]) - 1
        comp.ac_ind = np.array([6,10,4,9,3,8]) - 1
        
        #TicketPrice
        comp.original = outputs_before.TicketPrice
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, outputs_after.TicketPrice))
        
        #Fuelburn
        comp.original = coefficients_before.Fuelburn
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, coefficients_after.Fuelburn))
        
        #Doc
        comp.original = coefficients_before.Doc
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, coefficients_after.Doc))
        
        #Nox
        comp.original = coefficients_before.Nox
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, coefficients_after.Nox))
        
        #BlockTime
        comp.original = coefficients_before.BlockTime
        comp.execute()
        self.assertTrue(np.allclose(comp.filtered, coefficients_after.BlockTime))
    
class MaxTripTestCase(unittest.TestCase):
    
    def test_3routes(self):
        inputs_after = load_data_file('inputs_after_3routes.mat')
        inputs_after = inputs_after['Inputs']
        
        coefficients_after = load_data_file('coefficients_after_3routes.mat')
        coefficients_after = coefficients_after['Coefficients']
        
        constants_after = load_data_file('constants_after_3routes.mat')
        constants_after = constants_after['Constants']
        
        comp = MaxTrip_3Routes()
        comp.J = 3
        comp.K = 2
        comp.ACNum = inputs_after.ACNum
        comp.BlockTime = coefficients_after.BlockTime
        comp.MH = constants_after.MH
        comp.TurnAround = inputs_after.TurnAround
        
        comp.execute()
        
        self.assertTrue(np.allclose(comp.MaxTrip, inputs_after.MaxTrip))
        
    def test_11routes(self):
        inputs_after = load_data_file('inputs_after_11routes.mat')
        inputs_after = inputs_after['Inputs']
        
        coefficients_after = load_data_file('coefficients_after_11routes.mat')
        coefficients_after = coefficients_after['Coefficients']
        
        constants_after = load_data_file('constants_after_11routes.mat')
        constants_after = constants_after['Constants']
        
        comp = MaxTrip_11Routes()
        comp.J = 11
        comp.K = 2
        comp.ACNum = inputs_after.ACNum
        comp.BlockTime = coefficients_after.BlockTime
        comp.MH = constants_after.MH
        comp.TurnAround = inputs_after.TurnAround
        
        comp.execute()
        
        print comp.MaxTrip
        print inputs_after.MaxTrip
        
        self.assertTrue(np.allclose(comp.MaxTrip, inputs_after.MaxTrip))
        
    def test_31routes(self):
        inputs_after = load_data_file('inputs_after_31routes.mat')
        inputs_after = inputs_after['Inputs']
        
        coefficients_after = load_data_file('coefficients_after_31routes.mat')
        coefficients_after = coefficients_after['Coefficients']
        
        constants_after = load_data_file('constants_after_31routes.mat')
        constants_after = constants_after['Constants']
        
        comp = MaxTrip_31Routes()
        comp.J = 31
        comp.K = 6
        comp.ACNum = inputs_after.ACNum
        comp.BlockTime = coefficients_after.BlockTime
        comp.MH = constants_after.MH
        comp.TurnAround = inputs_after.TurnAround
        
        comp.execute()
        
        self.assertTrue(np.allclose(comp.MaxTrip, inputs_after.MaxTrip))
    

if __name__ == "__main__":

    unittest.main()

    
    