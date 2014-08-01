
import unittestfrom scipy.io import loadmatfrom airline_alloc import OverrideFunction_Init
def load_data_file(data_dir, file_name):			file_name = os.path.join(data_dir, file_name)			return loadmat(file_name, squeeze_me=True, struct_as_record=False)			class OverrideFunction_Init_TestCase(unittest.testcase):	def setUp(self):		self.data_dir = os.path.join(			os.path.pardir,			os.path.pardir,			os.path.pardir,			'Data'		)			def test_range_extract(self):		mat_file = load_data_file(self.data_dir, 'inputs_before_3routes.mat')		mat_file = mat_file['Inputs']				comp = RangeExtract()		comp.RVector = mat_file.RVector		comp.distance = np.array([2000, 1500, 1000])				comp.execute()				self.assertTrue(np.close(comp.ind, np.array([80, 1782, 674])))			def test_init_3route(self):		inputs = load_data_file(self.data_dir, 'inputs_before_3routes.mat')		inputs = inputs['Inputs']				constants = load_data_file(self.data_dir, 'constants_before_3routes.mat')		constants = constants['constants']				range_extract = RangeExtract()		range_extract.RVector = inputs.RVector		range_extract.distance = np.array([2000, 1500, 1000])		range_extract.execute()				comp = OverrideFunction_Init()		comp.ac_ind = np.array([9, 10])		comp.distance = np.array([2000, 1500, 1000])		comp.DVector = np.array([[1,2,3],[300, 700, 200]])		comp.ACNum = np.array([6,4])		comp.RVector_in = inputs.RVector		comp.AvailPax_in = inputs.AvailPax		comp.route_ind = range_extract.ind		comp.MH_in = constants.MH				comp.execute()				inputs = load_data_file(self.data_dir, 'inputs_after_3routes.mat')		inputs = inputs['Inputs']				constants = load_data_file(self.data_dir, 'constants_after_3routes.mat')		constants = constants['constants']				self.assertTrue(np.close(comp.RVector_out, inputs.RVector))		self.assertTrue(np.close(comp.AvailPax, inputs.AvailPax))		self.assertTrue(np.close(comp.TurnAround, inputs.TurnAround))		self.assertTrue(np.close(comp.J, 3))		self.assertTrue(np.close(comp.K, 2))		self.assertTrue(np.close(compLim, inputs.Lim))		
class Airline_allocTestCase(unittest.TestCase):

    def setUp(self):
        pass
        
    def tearDown(self):
        pass
        
    # add some tests here...
    	def test_override_function_init(self):		comp = OverrideFunction_Init()		comp.ac_ind = np.array([9, 10])		comp.distance = np.array([2
    #def test_Airline_alloc(self):
        #pass
        
if __name__ == "__main__":
    unittest.main()
    