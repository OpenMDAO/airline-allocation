__all__ = ['ArrayFilter', 'RangeExtract', 'Initialization', 'MaxTrip_3Routes', 'MaxTrip_11Routes', 'MaxTrip_31Routes']

import numpy as np

from openmdao.main.api import Assembly, Component

from openmdao.lib.datatypes.api import Array, Float, Int

class ArrayFilter(Component):
    ac_ind = Array(iotype='in', desc="aircraft index")
    route_ind = Array(iotype='in', desc="route index")
    
    original = Array(iotype='in')
    filtered = Array(iotype='out')
    
    def execute(self):
        array_out = np.array([])
        temp_in = self.original
        
        for kk in xrange(len(self.ac_ind)):
            ta = np.array([])
            
            for jj in xrange(len(self.route_ind)):
                ta = np.append(ta, temp_in[self.ac_ind[kk], self.route_ind[jj]])
                
            array_out = np.append(array_out, ta)
        
        array_out = array_out.reshape(len(self.ac_ind), -1)
        
        self.filtered = array_out

class RangeExtract(Component):
    RVector = Array(iotype='in')
    distance = Array(iotype='in')
    
    ind = Array(iotype='out')
    
    def execute(self):
        self.ind = np.zeros(len(self.distance))
        
        for cc in xrange(len(self.distance)):
            _range = self.distance[cc]
            diff_min = np.inf
            
            for ii, ii_value in np.ndenumerate(self.RVector):
                diff = np.abs(ii_value - _range)
                if diff < diff_min:
                    
                    self.ind[cc] = ii[0]
                    diff_min = diff
                
class Initialization(Component):
    ac_ind = Array(iotype='in', desc='aircraft indices')
    distance = Array(iotype='in', desc='route distance')
    DVector = Array(iotype='in', desc='route demand')
    ACNum = Array(iotype='in', desc='number of aircraft')
    RVector_in = Array(iotype='in')
    AvailPax_in = Array(iotype='in')
    route_ind = Array(iotype='in', desc='route indices')
    MH_in = Array(iotype='in')
    TurnAround = Int(iotype='in')
    demfac = Int(iotype='in')
    FuelCost = Float(iotype='in')
    
    #Filtered Inputs
    RVector_out = Array(iotype='out')
    AvailPax_out = Array(iotype='out')
    J = Int(iotype='out', desc="Number of routes")
    K = Int(iotype='out', desc="Number of aircraft types")
    Lim = Array(iotype='out')
    
    #Constants
    Runway_out = Array(iotype='out')
    MH_out = Array(iotype='out')
    
    def execute(self):
        self.RVector_out = self.RVector_in[self.route_ind]
        self.AvailPax_out = self.AvailPax_in[self.ac_ind]
        self.TurnAround = 1
        
        self.J = len(self.DVector[:,1])
        self.K = len(self.AvailPax_out)
        self.Lim = np.ones((self.K, self.J))
        
        self.Runway = 1e4 * len(self.RVector_out)
        self.MH_out = self.MH_in[self.ac_ind]
        self.FuelCost = 0.2431
        self.demfac = 1
        
class MaxTrip_3Routes(Component):
    J = Int(iotype='in')
    K = Int(iotype='in')
    ACNum = Array(iotype='in')
    BlockTime = Array(iotype='in')
    MH = Array(iotype='in')
    TurnAround = Int(iotype='in')
    
    MaxTrip = Array(iotype='out')
    
    def execute(self):
        rw = 0
        max_trip = np.zeros(self.K*self.J)
        
        for kk in range(self.K):
            for jj in range(self.J):
                max_trip[rw] = self.ACNum[kk] * np.ceil(12/(self.BlockTime[kk, jj] * (1 + self.MH[kk]) + self.TurnAround))
                rw = rw + 1
                
        self.MaxTrip = max_trip

class MaxTrip_11Routes(Component):
    J = Int(iotype='in')
    K = Int(iotype='in')
    ACNum = Array(iotype='in')
    BlockTime = Array(iotype='in')
    MH = Array(iotype='in')
    TurnAround = Int(iotype='in')
    
    MaxTrip = Array(iotype='out')
    
    def execute(self):
        rw = 0
        max_trip = np.zeros(self.K*self.J)
        
        for kk in range(self.K):
            for jj in range(self.J):
                max_trip[rw] = self.ACNum[kk] * np.ceil(12/(self.BlockTime[kk, jj] * (1 + self.MH[kk]) + self.TurnAround)) + 1
                rw = rw + 1
                
        self.MaxTrip = max_trip

class MaxTrip_31Routes(Component):
    J = Int(iotype='in')
    K = Int(iotype='in')
    ACNum = Array(iotype='in')
    BlockTime = Array(iotype='in')
    MH = Array(iotype='in')
    TurnAround = Int(iotype='in')
    
    MaxTrip = Array(iotype='out')
    
    def execute(self):
        rw = 0
        max_trip = np.zeros(self.K*self.J)
        
        for kk in range(self.K):
            for jj in range(self.J):
                max_trip[rw] = self.ACNum[kk] * np.ceil(12/(self.BlockTime[kk, jj] * (1 + self.MH[kk]) + self.TurnAround))
                rw = rw + 1
                
        self.MaxTrip = max_trip