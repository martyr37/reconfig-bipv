#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:00:08 2022

@author: mlima
"""

####################################################################################################
import numpy as np

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

def make_netlist(embedding, shading_map):
    # use separator
    
    if embedding.check_embedding() != True:
        return "Invalid embedding."
    
    circuit = Circuit('Netlist')
    rows, columns = embedding.rows, embedding.columns
    
    node_dictionary = {}
    
    for r in range(rows):
        for c in range(columns):
            circuit.subcircuit(SolarCell(str(r) + '-' + str(c), \
                                intensity=shading_map[r,c]))
            node_dictionary[str(r) + '-' + str(c)] = [None, None]       
    
    ground_connections = embedding.terminal_array[:,:,0]
    pos_connections = embedding.terminal_array[:,:,0]
    
    for r in range(rows):
        for c in range(columns):
            if ground_connections[r,c] == True:
                node_dictionary[str(r) + '-' + str(c)][0] = 'gnd'
            if pos_connections[r,c] == True:
                node_dictionary[str(r) + '-' + str(c)][1] = 'pos'
    
    connection_dictionary = {}
    
    for r1 in range(rows):
        for c1 in range(columns):
            
            series_array = embedding.embedding[r1,c1,...,1]
            for r2 in range(rows):
                for c2 in range(columns):
                    if series_array[r2, c2] == True:
                        cell_tuple = ("-".join([str(r1),str(c1)]),\
                                      "-".join([str(r2),str(c2)]))
                        cell_tuple = tuple(sorted(cell_tuple))
                        
                        connection_dictionary[cell_tuple] = 's'
            
            parallel_array = embedding.embedding[r1,c1,...,2]
            for r2 in range(rows):
                for c2 in range(columns):
                    if parallel_array[r2, c2] == True:
                        cell_tuple = ("-".join([str(r1),str(c1)]),\
                                      "-".join([str(r2),str(c2)]))
                        cell_tuple = tuple(sorted(cell_tuple))
                        connection_dictionary[cell_tuple] = 'p'
                        
    # now iterate over connection_dictionary, updating nodes of each cell 
    node_counter = 97
    node_name_multiplier = 1
    # 'a' through 'z', then 'aa', then 'aaa', so on
    print(node_dictionary)

    return connection_dictionary, circuit

dct, c = make_netlist(array, sun)
# TODO: topology to netlist
