

import unittest
from scipy.io import loadmat

from airline_alloc import OverrideFunction_Init

def load_data_file(data_dir, file_name):
			file_name = os.path.join(data_dir, file_name)
			return loadmat(file_name, squeeze_me=True, struct_as_record=False)
			

class OverrideFunction_Init_TestCase(unittest.testcase):
	def setUp(self):
		self.data_dir = os.path.join(
			os.path.pardir,
			os.path.pardir,
			os.path.pardir,
			'Data'
		)
		
	def test_range_extract(self):
		mat_file = load_data_file(self.data_dir, 'inputs_before_3routes.mat')
		mat_file = mat_file['Inputs']
		
		comp = RangeExtract()
		comp.RVector = mat_file.RVector
		comp.distance = np.array([2000, 1500, 1000])
		
		comp.execute()
		
		self.assertTrue(np.close(comp.ind, np.array([80, 1782, 674])))
		
	def test_init_3route(self):
		inputs = load_data_file(self.data_dir, 'inputs_before_3routes.mat')
		inputs = inputs['Inputs']
		
		constants = load_data_file(self.data_dir, 'constants_before_3routes.mat')
		constants = constants['constants']
		
		range_extract = RangeExtract()
		range_extract.RVector = inputs.RVector
		range_extract.distance = np.array([2000, 1500, 1000])
		range_extract.execute()
		
		comp = OverrideFunction_Init()
		comp.ac_ind = np.array([9, 10])
		comp.distance = np.array([2000, 1500, 1000])
		comp.DVector = np.array([[1,2,3],[300, 700, 200]])
		comp.ACNum = np.array([6,4])
		comp.RVector_in = inputs.RVector
		comp.AvailPax_in = inputs.AvailPax
		comp.route_ind = range_extract.ind
		comp.MH_in = constants.MH
		
		comp.execute()
		
		inputs = load_data_file(self.data_dir, 'inputs_after_3routes.mat')
		inputs = inputs['Inputs']
		
		constants = load_data_file(self.data_dir, 'constants_after_3routes.mat')
		constants = constants['constants']
		
		self.assertTrue(np.close(comp.RVector_out, inputs.RVector))
		self.assertTrue(np.close(comp.AvailPax, inputs.AvailPax))
		self.assertTrue(np.close(comp.TurnAround, inputs.TurnAround))
		self.assertTrue(np.close(comp.J, 3))
		self.assertTrue(np.close(comp.K, 2))
		self.assertTrue(np.close(compLim, inputs.Lim))

class ArrayFilterTestCase(unittest.TestCase):
	def test_3routes(self):
		inputs_before = load_data_file(self.data_dir, 'inputs_before_3routes.mat')
		inputs_before = mat_file['Inputs']
		
		coefficients_before = load_data_file(self.data_dir, 'coefficients_before_3routes.mat')
		coefficients_before = mat_file['Coefficients'] 
		
		inputs_after = load_data_file(self.data_dir, 'inputs_after_3routes.mat')
		inputs_after = mat_file['Inputs']
		
		coefficients_after = load_data_file(self.data_dir, 'coefficients_after_3routes.mat')
		coefficients_after = mat_file['Coefficients'] 
		
		range_extract = RangeExtract()
		range_extract.RVector = inputs.RVector
		range_extract.distance = np.array([2000, 1500, 1000])
		range_extract.execute()
		
		comp = ArrayFilter()
		comp.route_ind = range_extrax.ind
		comp.ac_ind = np.array([9,10])
		
		#TicketPrice
		comp.original = inputs_before.TicketPrice
		comp.execute()
		self.assertTrue(np.close(comp.filtered, inputs_after.TicketPrice))
		
		#Fuelburn
		comp.original = coefficients_before.Fuelburn
		comp.execute()
		self.assertTrue(np.close(comp.filtered, coefficients_after.TicketPrice))
		
		#Doc
		comp.original = coefficients_before.Doc
		comp.execute()
		self.assertTrue(np.close(comp.filtered, coefficients_after.Doc))
		
		#Nox
		comp.original = coefficients_before.Nox
		comp.execute()
		self.assertTrue(np.close(comp.filtered, coefficients_after.Nox))
		
		#BlockTime
		comp.original = coefficients_before.BlockTime
		comp.execute()
		self.assertTrue(np.close(comp.filtered, coefficients_after.BlockTime))
		
	def test_11routes(self):
		inputs_before = load_data_file(self.data_dir, 'inputs_before_11routes.mat')
		inputs_before = mat_file['Inputs']
		
		coefficients_before = load_data_file(self.data_dir, 'coefficients_before_11routes.mat')
		coefficients_before = mat_file['Coefficients'] 
		
		inputs_after = load_data_file(self.data_dir, 'inputs_after_11routes.mat')
		inputs_after = mat_file['Inputs']
		
		coefficients_after = load_data_file(self.data_dir, 'coefficients_after_11routes.mat')
		coefficients_after = mat_file['Coefficients'] 
		
		range_extract = RangeExtract()
		range_extract.RVector = inputs.RVector
		range_extract.distance = np.array([
			162,753,974,1094,
			1357,1455, 2169,2249,
			2269,2337,2350])
		range_extract.execute()
		
		comp = ArrayFilter()
		comp.route_ind = range_extrax.ind
		comp.ac_ind = np.array([9,10])
		
		#TicketPrice
		comp.original = inputs_before.TicketPrice
		comp.execute()
		self.assertTrue(np.close(comp.filtered, inputs_after.TicketPrice))
		
		#Fuelburn
		comp.original = coefficients_before.Fuelburn
		comp.execute()
		self.assertTrue(np.close(comp.filtered, coefficients_after.TicketPrice))
		
		#Doc
		comp.original = coefficients_before.Doc
		comp.execute()
		self.assertTrue(np.close(comp.filtered, coefficients_after.Doc))
		
		#Nox
		comp.original = coefficients_before.Nox
		comp.execute()
		self.assertTrue(np.close(comp.filtered, coefficients_after.Nox))
		
		#BlockTime
		comp.original = coefficients_before.BlockTime
		comp.execute()
		self.assertTrue(np.close(comp.filtered, coefficients_after.BlockTime))
		
	def test_31routes(self):
		inputs_before = load_data_file(self.data_dir, 'inputs_before_31routes.mat')
		inputs_before = mat_file['Inputs']
		
		coefficients_before = load_data_file(self.data_dir, 'coefficients_before_31routes.mat')
		coefficients_before = mat_file['Coefficients'] 
		
		inputs_after = load_data_file(self.data_dir, 'inputs_after_31routes.mat')
		inputs_after = mat_file['Inputs']
		
		coefficients_after = load_data_file(self.data_dir, 'coefficients_after_31routes.mat')
		coefficients_after = mat_file['Coefficients'] 
		
		range_extract = RangeExtract()
		range_extract.RVector = inputs.RVector
		range_extract.distance = np.array([
			113,174,289,303,324,
			331,342,375,407,427,
			484,486,531,543,550,
			570,594,609,622,680,
			747,758,760,823,837,
			991,1098,1231,1407,1570,1626])
		range_extract.execute()
		
		comp = ArrayFilter()
		comp.route_ind = range_extrax.ind
		comp.ac_ind = np.array([6,10,4,9,3,8])
		
		#TicketPrice
		comp.original = inputs_before.TicketPrice
		comp.execute()
		self.assertTrue(np.close(comp.filtered, inputs_after.TicketPrice))
		
		#Fuelburn
		comp.original = coefficients_before.Fuelburn
		comp.execute()
		self.assertTrue(np.close(comp.filtered, coefficients_after.TicketPrice))
		
		#Doc
		comp.original = coefficients_before.Doc
		comp.execute()
		self.assertTrue(np.close(comp.filtered, coefficients_after.Doc))
		
		#Nox
		comp.original = coefficients_before.Nox
		comp.execute()
		self.assertTrue(np.close(comp.filtered, coefficients_after.Nox))
		
		#BlockTime
		comp.original = coefficients_before.BlockTime
		comp.execute()
		self.assertTrue(np.close(comp.filtered, coefficients_after.BlockTime))
	
class MaxTripTestCase(unittest.TestCase):
	
	def test_3routes(self):
		inputs_after = load_data_file(self.data_dir, 'inputs_after_3routes.mat')
		inputs_after = mat_file['Inputs']
		
		coefficients_after = load_data_file(self.data_dir, 'coefficients_after_3routes.mat')
		coefficients_after = mat_file['Coefficients']
		
		constants_after = load_data_file(self.data_dir, 'constants_after_3routes.mat')
		constants_after = mat_file['Constants']
		
		comp = MaxTrip_3Routes()
		comp.J = 3
		comp.K = 2
		comp.ACNum = inputs_before.ACNum
		comp.BlockTime = coefficients_after.BlockTime
		comp.MH = constants_after.MH
		comp.TurnAround = inputs_after.TurnAround
		
		comp.execute()
		
		self.assertTrue(np.close(comp.MaxTrip, inputs_after.MaxTrip))
		
        

    # add some tests here...

    
	def test_override_function_init(self):
		comp = OverrideFunction_Init()
		comp.ac_ind = np.array([9, 10])
		comp.distance = np.array([2
    #def test_Airline_alloc(self):

        #pass

        

if __name__ == "__main__":

    unittest.main()

    
	