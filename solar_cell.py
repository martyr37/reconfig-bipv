#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:56:53 2022

@author: mlima
"""

####################################################################################################
import numpy as np

import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *
logger = Logging.setup_logging()

####################################################################################################

class SolarCell(SubCircuit):
    __nodes__ = ('t_in', 't_out')
    
    def __init__(self, name, intensity=10@u_A, series_resistance=1@u_mOhm, parallel_resistance=10@u_kOhm,\
                 saturation_current_1=1e-8, ideality_1=2, saturation_current_2=1e-12, ideality_2=1):
        
        SubCircuit.__init__(self, name, *self.__nodes__) 
        
        self.model('Diode1', 'D', IS=saturation_current_1, N=ideality_1)
        self.model('Diode2', 'D', IS=saturation_current_2, N=ideality_2)
        
        self.intensity = intensity
        self.series_resistance = series_resistance
        self.parllel_resistance = parallel_resistance
        self.I(1, 't_load', 't_in', intensity)
        self.R(2, 't_load', 't_out', series_resistance)
        self.R(3, 't_in', 't_load', parallel_resistance)
        self.Diode(4, 't_in', 't_load', model='Diode1')
        self.Diode(5, 't_in', 't_load', model='Diode2')
    
class BypassDiode(SubCircuit):
    __nodes__ = ('in', 'out')
    def __init__(self, name):
        SubCircuit.__init__(self, name, *self.__nodes__)
        self.model('BypassDiode', 'D', IS=680e-12, RS=0.001, N=1.003, CJO=1e-12, M=0.3, EG=0.69, XTI=2)
        self.Diode(1, 'in', 'out', model='BypassDiode')
        
#%% Total Cross Tied Interconnection
    
def TCT_interconnection(NUMBER_IN_SERIES, NUMBER_IN_PARALLEL, intensity_array):     
    circuit = Circuit('TCT Interconnected')
    for row in range(0,NUMBER_IN_PARALLEL): 
        for column in range(0,NUMBER_IN_SERIES):
            circuit.subcircuit(solar_cell(str(row) + str(column),intensity=intensity_array[row,column]))
        
    for row in range(0, NUMBER_IN_PARALLEL):
        for column in range(0, NUMBER_IN_SERIES):
            if column == 0:
                circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                          column + 1, circuit.gnd)
            else:
                circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                          column + 1, column)
    return circuit

#%% Series-Parallel Interconnection

def SP_interconnection(NUMBER_IN_SERIES, NUMBER_IN_PARALLEL, intensity_array):
    circuit = Circuit('SP Interconnected')
    for row in range(0,NUMBER_IN_PARALLEL):
        for column in range(0,NUMBER_IN_SERIES):
            circuit.subcircuit(solar_cell(str(row) + str(column),intensity=intensity_array[row,column]))
    
    for row in range(0, NUMBER_IN_PARALLEL):
        for column in range(0, NUMBER_IN_SERIES):
            if column == 0:
                circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                          str(row) + str(column), circuit.gnd)
            elif column == NUMBER_IN_SERIES - 1:
                circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                          '1', str(row) + str(column - 1)) # '1' represents the positive terminal of the module
            else:
                circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                          str(row) + str(column), str(row) + str(column - 1))
    return circuit

#%% All Series Connection
def all_series_connection(columns, rows, intensity_array):
    circuit = Circuit('All Series Connected')
    for row in range(0, rows):
        for column in range(0, columns):
            circuit.subcircuit(solar_cell(str(row) + str(column), intensity=intensity_array[row,column]))
        
    for row in range(0, rows):
        if row % 2 == 0: # 1st, 3rd, 5th row
            for column in range(0, columns):
                if row == 0 and column == 0: 
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              str(row) + str(column), circuit.gnd) # beginning
                elif column == 0:
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              str(row) + str(column), str(row - 1) + str(column)) # connect with cell above
                elif row == rows - 1 and column == columns - 1:
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              '1', str(row) + str(column - 1)) # if an odd number of rows, make last connection 
                else:
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              str(row) + str(column), str(row) + str(column - 1)) # connect as normal with previous cell
        elif row % 2 == 1: # 2nd, 4th, 6th row
            for column in range(columns - 1, -1, -1): # step backwards
                if column == columns - 1:
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              str(row) + str(column), str(row - 1) + str(column)) # connect with cell above
                elif row == rows - 1 and column == 0:
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              '1', str(row) + str(column + 1))  # if an even number of rows, make last connection
                else:
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              str(row) + str(column), str(row) + str(column + 1)) # connect as normal with previous cell
                    
    return circuit

#print(all_series_connection(8, 3, np.full((3,8),10)))

#%% All Series w/ bypass diodes
def all_series_bypass(columns, rows, intensity_array):
    circuit = Circuit('All Series w/ Bypass Diodes')
    for row in range(0, rows):
        for column in range(0, columns):
            circuit.subcircuit(solar_cell(str(row) + str(column), intensity=intensity_array[row,column]))
    
    circuit.subcircuit(BypassDiode('D'))
    bypass_diode_count = 0
    
    for row in range(0, rows):
        if row % 2 == 0: # 1st, 3rd, 5th row
            for column in range(0, columns):
                if row == 0 and column == 0: 
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              str(row) + str(column), circuit.gnd) # beginning
                elif column == 0:
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              str(row) + str(column), str(row - 1) + str(column)) # connect with cell above
                elif row == rows - 1 and column == columns - 1:
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              '1', str(row) + str(column - 1)) # if an odd number of rows, make last connection 
                else:
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              str(row) + str(column), str(row) + str(column - 1)) # connect as normal with previous cell
        elif row % 2 == 1: # 2nd, 4th, 6th row
            for column in range(columns - 1, -1, -1): # step backwards
                if column == columns - 1:
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              str(row) + str(column), str(row - 1) + str(column)) # connect with cell above
                elif row == rows - 1 and column == 0:
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              '1', str(row) + str(column + 1))  # if an even number of rows, make last connection
                    circuit.X('D' + str(bypass_diode_count) + 'sbckt', 'D', \
                              str(row - 1) + str(column), str(row) + str(column))
                    bypass_diode_count += 1
                elif column == 0:
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              str(row) + str(column), str(row) + str(column + 1))
                    circuit.X('D' + str(bypass_diode_count) + 'sbckt', 'D', \
                              str(row - 1) + str(column), str(row) + str(column))
                    bypass_diode_count += 1
                else:
                    circuit.X(str(row) + str(column) + 'sbckt', str(row) + str(column), \
                              str(row) + str(column), str(row) + str(column + 1)) # connect as normal with previous cell         
                        
    return circuit
    
#%% 4-block
def block_shading(rows, columns, current_list):
    # current_list is a one-dimensional array cnotaining the intensities in each block
    # e.g. [10, 7, 2, 6]
    
    # for now, assuming that it will always be split into 4 blocks 
    
    intensity_array = np.zeros((rows, columns))
    
    for row in range(0, rows):
        for column in range(0, columns):
            if row < rows/2 and column < columns/2:
                intensity_array[row, column] = current_list[0]
            elif row < rows/2 and column >= columns/2:
                intensity_array[row, column] = current_list[1]
            elif row >= rows/2 and column < columns/2:
                intensity_array[row, column] = current_list[2]
            elif row >= rows/2 and column >= columns/2:
                intensity_array[row, column] = current_list[3]
                
    return intensity_array

#foo = np.array([1, 2, 3, 4])
#print(block_shading(12, 4, foo))

#%% Checkboard Shading
def checkerboard_shading(rows, columns, current_list):
    
    length = int(current_list.size)
    current_list_index = 0
    # current_list is an array containing the non-zero elements read from left to right.
    intensity_array = np.full((rows, columns), 0.2) # 0.2 is the "minimum" light level
    for row in range(0, rows):
        for column in range(0, columns):
            if (row + column) % 2 == 0:
                intensity_array[row, column] = current_list[current_list_index]
                current_list_index += 1
                current_list_index = current_list_index % length
    
    return intensity_array

#foo = np.array([(x/10) for x in range(1, 10, 1)])
#print(checkerboard_shading(12, 12, foo))
                
#%% Random Intensities

def random_shading(rows, columns, mean, variance):
    intensity_array = np.zeros((rows, columns))
    for row in range(0, rows):
        for column in range(0, columns):
            random_value = np.random.normal(mean, variance)
            if random_value > 1:
                random_value = 1
            elif random_value < 0.2:
                random_value = 0.2
            intensity_array[row, column] = random_value
    
    return intensity_array