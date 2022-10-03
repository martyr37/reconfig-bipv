#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:23:35 2022

@author: mlima
"""

####################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

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
        self.validate_embedding()
        
        
    """
    def __init__(self, rows, columns, channels = 3, terminals = 2,\
                 shading_map = None):
        CircuitEmbedding.__init__(self, rows, columns, channels, terminals)
        
        if shading_map is None:
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
                        self.make_connection(r, c, r, c+1, 's1')
                    elif r == self.rows - 1 and c == self.columns - 1:
                        # make final connection to +ve terminal
                        self.connect_to_pos(r, c)
                    elif c == self.columns - 1:
                        # if at the right end, make connection with cell below
                        self.make_connection(r, c, r+1, c, 's1')
                    else:
                        # cell to its right is connected in series
                        self.make_connection(r, c, r, c+1, 's1')
            elif r % 2 == 1:
                for c in range(self.columns - 1, -1, -1):
                    if r == self.rows - 1 and c == 0:
                        # make final connection to +ve terminal
                        self.connect_to_pos(r, c)
                    elif c == 0:
                        # make connection to row below
                        self.make_connection(r, c, r+1, c, 's1')
                    else:
                        self.make_connection(r, c, r, c-1, 's1')

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
                        self.make_connection(r, c, r1, c+1, 's1')
                # cells below are all 2's (parallel)
                for r1 in range(r + 1, self.rows):
                    self.make_connection(r, c, r1, c, 'p')
    # 
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
                
                series1_array = self.embedding[r1,c1,...,0]
                for r2 in range(self.rows):
                    for c2 in range(self.columns):
                        if series1_array[r2, c2] == True:
                            cell_tuple = ("-".join([str(r1),str(c1)]),\
                                          "-".join([str(r2),str(c2)]))
                            cell_tuple = tuple(cell_tuple)
                            
                            connection_dict[cell_tuple] = 's1'
                
                series2_array = self.embedding[r1,c1,...,1]
                for r2 in range(self.rows):
                    for c2 in range(self.columns):
                        if series2_array[r2, c2] == True:
                            cell_tuple = ("-".join([str(r1),str(c1)]),\
                                          "-".join([str(r2),str(c2)]))
                            cell_tuple = tuple(cell_tuple)
                            
                            connection_dict[cell_tuple] = 's2'
                            
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
        def s1_connect(connection):
            global node_counter
            global node_dict
            cell1 = connection[0]
            cell2 = connection[1]
            node12 = node_dict[cell1][1]
            node21 = node_dict[cell2][0]
           
            if node12 != None and node12 != 'pos' and node21 == None:
                node_dict[cell2][0] = node12
            elif node21 != None and node21 != 'gnd' and node12 == None:
                node_dict[cell1][1] = node21
            elif node12 == None and node21 == None:
                node_dict[cell1][1] = get_node_name(node_counter)
                node_dict[cell2][0] = get_node_name(node_counter)
                node_counter += 1
        
        def s2_connect(connection):
            global node_counter
            global node_dict
            cell1 = connection[0]
            cell2 = connection[1]
            node11 = node_dict[cell1][0]
            node22 = node_dict[cell2][1]
           
            if node11 != None and node11 != 'gnd':
                node_dict[cell2][1] = node11
            elif node22 != None and node22 != 'pos':
                node_dict[cell1][0] = node22
            elif node11 == None and node22 == None:
                node_dict[cell1][0] = get_node_name(node_counter)
                node_dict[cell2][1] = get_node_name(node_counter)
                node_counter += 1
        
        for connection, ctype in connection_dict.items():
            if ctype == 's1':
                s1_connect(connection)
            elif ctype == 's2':
                s2_connect(connection)
            elif ctype == 'p':
                parallel_connect(connection)
        
        dangling = []
        for node in node_dict:
            if None in node_dict[node]:
                dangling.append(node)
                
        for d in dangling:
            node_dict.pop(d)
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
        self.VOC = max(self.V[self.I>=0])
        self.ISC = self.I[self.V.argmin()]
        self.FF = self.MPP/(self.VOC * self.ISC)
        
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
            plt.imshow(self.embedding[r,c,...,0] | self.embedding[r,c,...,1]\
                       | self.embedding[r,c,...,2])
        elif connection_type == 's':
            plt.imshow(self.embedding[r,c,...,0] | self.embedding[r,c,...,1])
        elif connection_type == 's1':
            plt.imshow(self.embedding[r,c,...,0])
        elif connection_type == 's2':
            plt.imshow(self.embedding[r,c,...,1])
        elif connection_type == 'p':
            plt.imshow(self.embedding[r,c,...,2])
        else:
            raise ValueError("connection type should be 's', 's1', 's2', or 'p'")

#%% generate random shading maps
def generate_shading(multiplier, limit, rows, columns):
    a = np.random.rand(rows,columns)*multiplier
    i = np.where(a > 1)
    a[i] = limit
    return a

def generate_gaussian(dots, rows, columns, spread=2, size=1000, diag='r'):
    x_points = [round(np.random.sample()*columns, 2) for x in range(dots)]
    y_points = [-round(np.random.sample()*rows, 2) for x in range(dots)]
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, columns)
    ax.set_ylim(-rows, 0)
    plt.scatter(x_points, y_points)
    """
    
    cov_matrices = []
    for x in range(dots):
        a = np.random.sample(size=(2,2)) * spread
        if diag == 'r':
            if np.random.randint(0,2) == 0:
                diag = False
            else:
                diag = True
        if diag == False:
            b = np.dot(a, np.transpose(a))
            cov_matrices.append(b)
        elif diag == True:
            a[0][1] = 0
            a[1][0] = 0
            cov_matrices.append(a)
        
    #print(cov_matrices)
    
    sample_array = np.zeros((dots, size, 2))
    for i in range(dots):
        sample = np.random.multivariate_normal((x_points[i], y_points[i]),\
                                                        cov_matrices[i],\
                                                        size=size)
        sample_array[i] = sample
        #xs, ys = sample[:,0], sample[:,1]
        #plt.scatter(xs, ys)    
        
    # Caclulate density
    shading_array = np.zeros((rows, columns))
    sample_array = np.reshape(sample_array, (dots*size, 2))   
    
    for point in sample_array:
        x, y = round(point[0]), -round(point[1])
        if x < 0 or x >= columns:
            continue
        if y < 0 or y >= rows:
            continue
        #print(x, y)
        current = shading_array[y, x]
        shading_array[y, x] = current + 1
    
    #plt.imshow(shading_array)
    #shading_array = shading_array / shading_array.max()
    #shading_array = np.around(shading_array, 2)
    #shading_array[shading_array < 0.5] = 0.5
    shading_array = np.interp(shading_array, \
                              (shading_array.min(), shading_array.max()), \
                              (0, 10))
    shading_array[shading_array > 4] = 4
    shading_array = np.interp(shading_array, \
                              (shading_array.min(), shading_array.max()), \
                              (0, 10))
    
    return shading_array

#%% SolarModule testing
"""
obj = SolarModule(10, 6)
#obj.series_embedding()
obj.tct_embedding()
obj.make_netlist()
obj.simulate()
obj.plot_netlist()
#obj.imshow(3, 3)
#obj.imshow(3, 3, 's')
## --PRINT NETLIST-- 
print(obj.circuit)
"""
#%%
boo = [True, False]
rand = np.random.choice(boo, size=(10, 6, 10, 6, 3))
foo = SolarModule(10, 6)
foo.embedding = rand
print(foo.filter_embedding())
print(foo.check_embedding())
