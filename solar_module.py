#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:23:35 2022

@author: mlima
"""

####################################################################################################
import numpy as np

import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *
logger = Logging.setup_logging()

from solar_cell import SolarCell

####################################################################################################

#%%
class SolarModule():
    """
    The SolarModule object will contain:
        
        - A data structure that will hold the information of all the cell 
        interconnections also known as the circuit topology.
        - The Ngspice netlist that will be built according to the data structure
        - Methods to update the data structure and netlist accordingly

    A function that can create series, parallel, SP and TCT modules by calling
    an instance of SolarModule will be made after this class defintion.
    """

    def __init__(self):
        self.mpp = get_mpp()
        pass
    
    def __repr__(self):
        pass
    
    def circuit_topology(self):
        pass
    
    def netlist(self):
        pass
    
    def update_shading_map(self):
        pass
    
    def plot_curves(self):
        pass
    
    def get_mpp(self):
        return self._mpp
        
    def set_mpp(self):
        pass
        #self._mpp = something
        
    # do the same thing for voc, isc, vmp, imp, FF.
    
    mpp = property(get_mpp, set_mpp)
    
COLUMNS = 6
ROWS = 10

cell_names = []
index = 0

for row in range(0, ROWS):
    for column in range(0, COLUMNS):
        cell_names.append(str(row) + str(column))

topology = []

for cell in cell_names:
    cell_dict = {cell_inner: 0 for cell_inner in cell_names}
    topology.append(cell_dict)
        
topology = np.array(topology)
topology = topology.reshape((ROWS, COLUMNS))

# TODO: recreate basic connections (series, parallel, SP, TCT) 
# TODO: topology to netlist
 