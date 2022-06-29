#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:00:08 2022

@author: mlima
"""

####################################################################################################
import numpy as np
import matplotlib.pyplot as plt

from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *

from solar_cell import SolarCell
from solar_module import SolarModule

####################################################################################################
#%% embedding dimensions 
ROWS = 3
COLUMNS = 3
CHANNELS = 3 # [connection, series, parallel]
TERMINALS = 2 # [ground, +ve]

#%% CircuitEmbedding object definition
class CircuitEmbedding():
    
    def __init__(self, rows, columns, channels = 3, terminals = 2):
        self.rows = rows
        self.columns = columns
        self.channels = channels
        self.terminals = terminals
        
        self.create_empty_embedding()
    
    def __str__(self):
        pass
        
    def create_empty_embedding(self):
        embedding = np.zeros((self.rows, self.columns, self.rows, self.columns,\
                              self.channels), dtype=bool)
        terminal_array = np.zeros((self.rows, self.columns, self.terminals),\
                                  dtype=bool)
        self.embedding = embedding
        self.terminal_array = terminal_array
    
    def make_connection(self, r1, c1, r2, c2, connection_type):
        if connection_type == 's':
            self.embedding[r1, c1, r2, c2, 0] = True
            self.embedding[r2, c2, r1, c1, 0] = True
            self.embedding[r1, c1, r2, c2, 1] = True
            self.embedding[r2, c2, r1, c1, 1] = True
        elif connection_type == 'p':
            self.embedding[r1, c1, r2, c2, 0] = True
            self.embedding[r2, c2, r1, c1, 0] = True
            self.embedding[r1, c1, r2, c2, 2] = True
            self.embedding[r2, c2, r1, c1, 2] = True
    
    def connect_to_ground(self, r, c):
        self.terminal_array[r, c, 0] = True
    
    def connect_to_pos(self, r, c):
        self.terminal_array[r, c, 1] = True
# TODO: update check_embedding
# TODO: incorporate into solar_module class and python file
    def check_embedding(self):
        # Cells cannot have a True connection value to themselves
        # The reverse connection should be the same type
        # Cannot have both a series and parallel connection simultaneously
        # gnd and +ve terminals must have completely True or False dimensions
        for r in range(self.rows):
            for c in range(self.columns):
                if self.embedding[r,c,r,c,0] == True:
                    return "Invalid embedding: The cell cannot have a connection"\
                        + " to itself. Error occured at " + str(r) + str(c)
                if self.embedding[r,c,r,c,1] == True:
                    return "Invalid embedding: The cell cannot have a connection"\
                        + " to itself. Error occured at " + str(r) + str(c)
                if self.embedding[r,c,r,c,2] == True:
                    return "Invalid embedding: The cell cannot have a connection"\
                        + " to itself. Error occured at " + str(r) + str(c)
                for r1 in range(self.rows):
                    for c1 in range(self.columns):
                # connection between two cells cannot be series & parallel
                        if self.embedding[r,c,r1,c1,1] and self.embedding[r,c,r1,c1,2] == True:
                            return "Invalid embedding: Connection between two"\
                                + "cells cannot be both series and parallel."\
                                + "Error occurred at " + " ".join([r,c,r1,c1])
                # reverse connection can be different, so long as it is not 
                # simultaneously series and parallel.
                # If series or parallel connection is made, check that the
                # reverse connection (but opposite type) is not also True. 
                        if True in self.embedding[r,c,r1,c1,1:]:
                            if self.embedding[r,c,r1,c1,1] == True and self.embedding[r1,c1,r,c,2] == True:
                                return "Invalid embedding: Connection"\
                                    + "has to be the same type as its reverse."\
                                    + "Error occurred at " + " ".join([r,c,r1,c1])
                            if self.embedding[r,c,r1,c1,2] == True and self.embedding[r1,c1,r,c,1] == True:
                                return "Invalid embedding: Connection"\
                                    + "has to be the same type as its reverse."\
                                    + "Error occurred at " + " ".join([r,c,r1,c1])   
                                    
        def store_as_text(self):
            # store as json file
            pass
        
        return True

#%% Convetional series module
def series_embedding(rows, columns):
    embedding = CircuitEmbedding(rows, columns)
    
    for r in range(rows):
        if r % 2 == 0:
            for c in range(columns):
                if r == 0 and c == 0:
                    # top-left cell connected to gnd
                    embedding.connect_to_ground(r, c)
                    embedding.make_connection(r, c, r, c+1, 's')
                elif r == rows - 1 and c == columns - 1:
                    # make final connection to +ve terminal
                    embedding.connect_to_pos(r, c)
                elif c == columns - 1:
                    # if at the right end, make connection with cell below
                    embedding.make_connection(r, c, r+1, c, 's')
                else:
                    # cell to its right is connected in series
                    embedding.make_connection(r, c, r, c+1, 's')
        elif r % 2 == 1:
            for c in range(columns - 1, -1, -1):
                if r == rows - 1 and c == 0:
                    # make final connection to +ve terminal
                    embedding.connect_to_pos(r, c)
                elif c == 0:
                    # make connection to row below
                    embedding.make_connection(r, c, r+1, c, 's')
                else:
                    embedding.make_connection(r, c, r, c-1, 's')
    return embedding

array = series_embedding(ROWS, COLUMNS)  

#%% Total-cross-tied module
def tct_embedding(rows, columns):
    embedding = CircuitEmbedding(rows, columns)
    for c in range(embedding.columns):
        for r in range(embedding.rows):
            # first column is connected to ground
            if c == 0:
                embedding.connect_to_ground(r, c)
            # last column is connected to +ve terminal
            elif c == embedding.columns - 1:
                embedding.connect_to_pos(r,c)
            # column to the right is all 1's (series)
            if c != embedding.columns - 1:
                for r1 in range(embedding.rows):
                    embedding.make_connection(r, c, r1, c+1, 's')
            # cells below are all 2's (parallel)
            for r1 in range(r + 1, embedding.rows):
                embedding.make_connection(r, c, r1, c, 'p')        

    return embedding
    
array = tct_embedding(ROWS, COLUMNS)  
    
#%% Create PySpice netlist from embedding
sun = np.full((ROWS, COLUMNS), 10)
# TODO: Only works up to 3x3. Node names are being overwritten. Need to fix.
def make_netlist(embedding, shading_map):
    # use separator
    
    if embedding.check_embedding() != True:
        return "Invalid embedding."
    
    circuit = Circuit('Netlist')
    rows, columns = embedding.rows, embedding.columns
    
    global node_dict
    node_dict = {}
    
    for r in range(rows):
        for c in range(columns):
            circuit.subcircuit(SolarCell(str(r) + '-' + str(c), \
                                intensity=shading_map[r,c]))
            node_dict[str(r) + '-' + str(c)] = [None, None]       
    
    ground_connections = embedding.terminal_array[:,:,0]
    pos_connections = embedding.terminal_array[:,:,1]
    
    for r in range(rows):
        for c in range(columns):
            if ground_connections[r,c] == True:
                node_dict[str(r) + '-' + str(c)][0] = 'gnd'
            if pos_connections[r,c] == True:
                node_dict[str(r) + '-' + str(c)][1] = 'pos'
    # creation of connection_dict
    connection_dict = {}
    for r1 in range(rows):
        for c1 in range(columns):
            
            series_array = embedding.embedding[r1,c1,...,1]
            for r2 in range(rows):
                for c2 in range(columns):
                    if series_array[r2, c2] == True:
                        cell_tuple = ("-".join([str(r1),str(c1)]),\
                                      "-".join([str(r2),str(c2)]))
                        cell_tuple = tuple(sorted(cell_tuple))
                        
                        connection_dict[cell_tuple] = 's'
            
            parallel_array = embedding.embedding[r1,c1,...,2]
            for r2 in range(rows):
                for c2 in range(columns):
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
        char = chr(char_no + 96)
        multiplier = int((counter + 1)/ 26) + 1
        return char * multiplier
    
    global node_counter
    node_counter = 1
    
    # do all the parallel connections first, to reduce ambiguous nodes
    just_parallel = dict(filter(lambda x: x[1] == 'p', connection_dict.items()))
    for connection in just_parallel:
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
            node_dict[cell1][0] = node22
        else:
            node_dict[cell1][1] = get_node_name(node_counter)
            node_dict[cell2][1] = get_node_name(node_counter)
            node_counter += 1
    just_series = dict(filter(lambda x: x[1] == 's', connection_dict.items()))
    # now do series connections and by process of elimination figure out
    # directionality of series connections  
    
    def series_connect(connection):
        global node_counter
        global node_dict
        cell1 = connection[0]
        cell2 = connection[1]
        node11 = node_dict[cell1][0]
        node12 = node_dict[cell1][1]
        node21 = node_dict[cell2][0]
        node22 = node_dict[cell2][1]
        
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
        
    for connection in just_series: 
        series_connect(connection)
    
    while any(None in pair for pair in list(node_dict.values())):
        for cell in node_dict:
            if None in node_dict[cell]:
                connection_subset = dict\
                    (filter(lambda x: cell in x[0], connection_dict.items()))
                for redo in connection_subset:
                    series_connect(redo)
                    break
    line = 0            
    for cell in node_dict:
        if node_dict[cell][0] == 'gnd':
            circuit.X("line" + str(line), cell, circuit.gnd, node_dict[cell][1])
        else:
            circuit.X("line" + str(line), cell, node_dict[cell][0], node_dict[cell][1])
        line += 1
    return circuit

c = make_netlist(array, sun)

#%% Plot netlist

def plot_netlist(netlist):
    netlist.V('input', netlist.gnd, 'pos', 0)
    print(netlist)
    simulator = netlist.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.dc(Vinput=slice(0,50,0.01))

    seriesI = np.array(analysis.Vinput)
    seriesV = np.array(analysis.sweep)
    seriesP = seriesI * seriesV
    seriesMPP = max(seriesP)
    seriesVMP = seriesV[seriesP.argmax()]
    seriesIMP = seriesI[seriesP.argmax()]
    
    plt.plot(seriesV, seriesI)
    plt.xlim(0,60)
    plt.ylim(0,100)
    
    print(seriesMPP)
plot_netlist(c)

