#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:23:35 2022

@author: mlima
"""

####################################################################################################
import numpy as np
import matplotlib.pyplot as plt

import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *
logger = Logging.setup_logging()

from solar_cell import SolarCell
from circuit_embedding import CircuitEmbedding
####################################################################################################

#%% SolarModule
class SolarModule(CircuitEmbedding):
    """
    The SolarModule object will contain:
        
        - A data structure that will hold the information of all the cell 
        interconnections also known as the circuit embedding.
        - The Ngspice netlist that will be built according to the data structure
        - Methods to update the data structure and netlist accordingly

    A function that can create series, parallel, SP and TCT modules by calling
    an instance of SolarModule will be made after this class defintion.
    
    Class Attributes:
        self.rows
        self.columns
        self.channels
        self.terminals
        self.embedding
        self.terminal_array
        self.shading_map
        self.circuit
        
    Class Methods:
        self.create_empty_embedding()
        self.make_connection(r1, c1, r2, c2, connection_type)
        self.connect_to_ground(r, c):
        self.connect_to_pos(r, c):
        self.check_embedding()
        
    """
    def __init__(self, rows, columns, channels = 3, terminals = 2,\
                 shading_map = None):
        CircuitEmbedding.__init__(self, rows, columns, channels, terminals)
        
        if shading_map == None:
            self.shading_map = np.full((self.rows, self.columns), 10)
        else:
            self.shading_map = shading_map
        
        self.circuit = self.make_netlist()
        
    def __str__(self):
        return "Module of size " + str(self.rows) + ' x ' + str(self.columns)
        
    def series_embedding(self):        
        for r in range(self.rows):
            if r % 2 == 0:
                for c in range(self.columns):
                    if r == 0 and c == 0:
                        # top-left cell connected to gnd
                        self.connect_to_ground(r, c)
                        self.make_connection(r, c, r, c+1, 's')
                    elif r == self.rows - 1 and c == self.columns - 1:
                        # make final connection to +ve terminal
                        self.connect_to_pos(r, c)
                    elif c == self.columns - 1:
                        # if at the right end, make connection with cell below
                        self.make_connection(r, c, r+1, c, 's')
                    else:
                        # cell to its right is connected in series
                        self.make_connection(r, c, r, c+1, 's')
            elif r % 2 == 1:
                for c in range(self.columns - 1, -1, -1):
                    if r == self.rows - 1 and c == 0:
                        # make final connection to +ve terminal
                        self.connect_to_pos(r, c)
                    elif c == 0:
                        # make connection to row below
                        self.make_connection(r, c, r+1, c, 's')
                    else:
                        self.make_connection(r, c, r, c-1, 's')

    def tct_embedding(self):
        for c in range(self.columns):
            for r in range(self.rows):
                # first column is connected to ground
                if c == 0:
                    self.connect_to_ground(r, c)
                # last column is connected to +ve terminal
                elif c == self.columns - 1:
                    self.connect_to_pos(r,c)
                # column to the right is all 1's (series)
                if c != self.columns - 1:
                    for r1 in range(self.rows):
                        self.make_connection(r, c, r1, c+1, 's')
                # cells below are all 2's (parallel)
                for r1 in range(r + 1, self.rows):
                    self.make_connection(r, c, r1, c, 'p')
    
    def make_netlist(self):        
        if self.check_embedding() != True:
            raise ValueError("Invalid embedding.")
        
        circuit = Circuit('Netlist')
        
        global node_dict
        node_dict = {}
        
        for r in range(self.rows):
            for c in range(self.columns):
                circuit.subcircuit(SolarCell(str(r) + '-' + str(c), \
                                    intensity=self.shading_map[r,c]))
                node_dict[str(r) + '-' + str(c)] = [None, None]       
        
        ground_connections = self.terminal_array[:,:,0]
        pos_connections = self.terminal_array[:,:,1]
        
        for r in range(self.rows):
            for c in range(self.columns):
                if ground_connections[r,c] == True:
                    node_dict[str(r) + '-' + str(c)][0] = 'gnd'
                if pos_connections[r,c] == True:
                    node_dict[str(r) + '-' + str(c)][1] = 'pos'
        # creation of connection_dict
        connection_dict = {}
        for r1 in range(self.rows):
            for c1 in range(self.columns):
                
                series_array = self.embedding[r1,c1,...,1]
                for r2 in range(self.rows):
                    for c2 in range(self.columns):
                        if series_array[r2, c2] == True:
                            cell_tuple = ("-".join([str(r1),str(c1)]),\
                                          "-".join([str(r2),str(c2)]))
                            cell_tuple = tuple(sorted(cell_tuple))
                            
                            connection_dict[cell_tuple] = 's'
                
                parallel_array = self.embedding[r1,c1,...,2]
                for r2 in range(self.rows):
                    for c2 in range(self.columns):
                        if parallel_array[r2, c2] == True:
                            cell_tuple = ("-".join([str(r1),str(c1)]),\
                                          "-".join([str(r2),str(c2)]))
                            cell_tuple = tuple(sorted(cell_tuple))
                            connection_dict[cell_tuple] = 'p'
                            
        # now iterate over connection_dictionary, updating nodes of each cell 
        # 'a' through 'z', then 'aa', then 'aaa', so on
        # loop until all the cells have nodes filled
        def get_node_name(counter):
            char_no = counter % 26
            char = chr(char_no + 97)
            multiplier = int((counter)/ 26) + 1
            return char * multiplier
        
        global node_counter
        node_counter = 0
        def parallel_connect(connection):
            global node_counter
            global node_dict
            cell1 = connection[0]
            cell2 = connection[1]
            node11 = node_dict[cell1][0]
            node12 = node_dict[cell1][1]
            node21 = node_dict[cell2][0]
            node22 = node_dict[cell2][1]
            
            # if cell 1's input is ground, then cell 2's input is also ground
            if node11 == 'gnd':
                node_dict[cell2][0] = node11
            # if cell 2's input is ground, then cell 1's input is also ground
            elif node21 == 'gnd':
                node_dict[cell1][0] = node21
            elif node11 != None:
                node_dict[cell2][0] = node11
            elif node21 != None:
                node_dict[cell1][0] = node21
            else:
                node_dict[cell1][0] = get_node_name(node_counter)
                node_dict[cell2][0] = get_node_name(node_counter)
                node_counter += 1
            # if cell 1's output is pos, then cell 2's output is also pos
            if node12 == 'pos':
                node_dict[cell2][1] = node12
            # if cell 2's output is pos, then cell 1's output is also pos
            elif node22 == 'pos':
                node_dict[cell1][1] = node22
            elif node12 != None:
                node_dict[cell2][1] = node12
            elif node22 != None:
                node_dict[cell1][1] = node22
            else:
                node_dict[cell1][1] = get_node_name(node_counter)
                node_dict[cell2][1] = get_node_name(node_counter)
                node_counter += 1
        def series_connect(connection):
            global node_counter
            global node_dict
            cell1 = connection[0]
            cell2 = connection[1]
            node11 = node_dict[cell1][0]
            node12 = node_dict[cell1][1]
            node21 = node_dict[cell2][0]
            node22 = node_dict[cell2][1]
            # ground or positive connection - only choice
            if node11 == 'gnd' or node22 == 'pos':
                if node12 != None:
                    node_dict[cell2][0] = node12
                elif node21 != None:
                    node_dict[cell1][1] = node21
                else:
                    node_dict[cell1][1] = get_node_name(node_counter)
                    node_dict[cell2][0] = get_node_name(node_counter)
                    node_counter += 1
            elif node12 == 'pos' or node21 == 'gnd':
                if node11 != None:
                    node_dict[cell2][1] = node11
                elif node22 != None:
                    node_dict[cell1][0] = node22
                else:
                    node_dict[cell1][0] = get_node_name(node_counter)
                    node_dict[cell2][1] = get_node_name(node_counter)
                    node_counter += 1
            elif (node11 != None and node12 == None) or (node21 == None and node22 != None):
                node_dict[cell1][1] = get_node_name(node_counter)
                node_dict[cell2][0] = get_node_name(node_counter)
                node_counter += 1
            elif (node11 == None and node12 != None) or (node21 != None and node22 == None):
                node_dict[cell1][0] = get_node_name(node_counter)
                node_dict[cell2][1] = get_node_name(node_counter)
                node_counter += 1
            """
            elif (node11 != None and node12 != None) and (node21 == None and node22 == None):
                if node11 > node12:
                    node_dict[cell2][1] = node11
                    node_dict[cell2][0] = get_node_name(node_counter)
                    node_counter += 1
                elif node12 > node11:
                    node_dict[cell2][0] = node12
                    node_dict[cell2][1] = get_node_name(node_counter)
                    node_counter += 1
            elif (node21 != None and node22 != None) and (node11 == None and node12 == None):
                if node21 > node22:
                    node_dict[cell1][1] = node21
                    node_dict[cell2][0] = get_node_name(node_counter)
                    node_counter += 1
                elif node22 > node21:
                    node_dict[cell1][0] = node22
                    node_dict[cell1][1] = get_node_name(node_counter)
                    node_counter += 1
            """
        # no ordering of connections
        
        loop_counter = 0
        while connection_dict != {} and loop_counter < len(connection_dict):
            cdictcopy = connection_dict.copy()
            for connection, ctype in cdictcopy.items(): 
                c1, c2 = connection[0], connection[1]
                # if all four nodes empty, continue
                if loop_counter == len(cdictcopy):
                    # do parallel connections once all exhausted
                    if ctype == 'p':
                        parallel_connect(connection)
                        connection_dict.pop(connection)
                elif node_dict[c1] == [None, None] and node_dict[c2] == [None, None]:
                    loop_counter += 1
                    continue # TODO: Fix infinite loop
                elif ctype == 's':
                    if node_dict[c1] == [None, None]:
                        # ['a', 'b'] and [None, None] should fail
                        if node_dict[c2][0] != None and node_dict[c2][1] != None:
                            loop_counter += 1
                            continue
                    if node_dict[c2] == [None, None]:
                        if node_dict[c1][0] != None and node_dict[c1][1] != None:
                            loop_counter += 1
                            continue
                    series_connect(connection) # popping without forming connection
                    connection_dict.pop(connection)
                elif ctype == 'p':
                    parallel_connect(connection)
                    connection_dict.pop(connection)
                loop_counter = 0
        
        """
        # parallel then series connections
        just_series = dict(filter(lambda x: x[1] == 's', connection_dict.items()))
        just_parallel = dict(filter(lambda x: x[1] == 'p', connection_dict.items()))        
        for connection in just_parallel:
            parallel_connect(connection)
            
        loop_counter = 0
        while just_series != {}:
            cdictcopy = just_series.copy()
            for connection in cdictcopy:
                c1, c2 = connection[0], connection[1]
                if node_dict[c1] == [None, None] and node_dict[c2] == [None, None]:
                    loop_counter += 1
                    continue
                elif node_dict[c1] == [None, None]:
                    # ['a', 'b'] and [None, None] should fail
                    if node_dict[c2][0] != None and node_dict[c2][1] != None:
                        loop_counter += 1
                        continue
                elif node_dict[c2] == [None, None]:
                    if node_dict[c1][0] != None and node_dict[c1][1] != None:
                        loop_counter += 1
                        continue
                
                elif node_dict[c1] == [None, None] or node_dict[c2] == [None, None]:
                    n1, n2 = node_dict[c1][0], node_dict[c1][1]
                    n3, n4 = node_dict[c2][0], node_dict[c2][1]
                    nlist = [n1, n2, n3, n4]
                    # if there is (None, 'a') or ('a', None), proceed as normal
                    if [n1, n2] == [None, None]:
                        if n3 != None and n4 != None:
                            continue
                    elif [n3, n4] == [None, None]:
                        if n1 != None and n2 != None:
                            continue 
                    # if there is gnd or pos, proceed as normal
                    elif n1 != 'gnd' or n4 != 'pos':
                        continue   
                series_connect(connection)
                just_series.pop(connection)
        """
        
        # transfer node_dict to PySpice netlist
        line = 0            
        for cell in node_dict:
            if node_dict[cell][0] == 'gnd':
                circuit.X("line" + str(line), cell, circuit.gnd, node_dict[cell][1])
            else:
                circuit.X("line" + str(line), cell, node_dict[cell][0], node_dict[cell][1])
            line += 1
        self.circuit = circuit
        self.connection_dict = connection_dict
        self.node_dict = node_dict
        
    def simulate(self):
        self.circuit.V('input', self.circuit.gnd, 'pos', 0)
        simulator = self.circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.dc(Vinput=slice(0,50,0.01))

        self.I = np.array(analysis.Vinput)
        self.V = np.array(analysis.sweep)
        self.P = self.I * self.V
        self.MPP = max(self.P)
        self.VMP = self.V[self.P.argmax()]
        self.IMP = self.I[self.P.argmax()]
        #TODO: Add VOC, ISC, FF
        
    def plot_netlist(self, xmax=50, ymax=150, pv=False):
        if pv == False:
            plt.plot(self.V, self.I)
            plt.xlim(0,xmax)
            plt.ylim(0,ymax)
        elif pv == True:
            plt.plot(self.V, self.P)
            plt.xlim(0,xmax)
            plt.ylim(0,ymax)
    
    def imshow(self, r, c, connection_type=None):
        if connection_type == None:
            plt.imshow(self.embedding[r,c,...,0])
        elif connection_type == 's':
            plt.imshow(self.embedding[r,c,...,1])
        elif connection_type == 'p':
            plt.imshow(self.embedding[r,c,...,2])
        else:
            raise ValueError("connection type should be 's' or 'p'")
    
#%% testing
"""
obj = SolarModule(10, 6)
obj.series_embedding()
#obj.tct_embedding()
obj.make_netlist()
obj.simulate()
obj.plot_netlist()
#obj.imshow(3, 3)
#obj.imshow(3, 3, 's')
"""


 