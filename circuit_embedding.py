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


####################################################################################################
#%% embedding dimensions 
"""
ROWS = 10
COLUMNS = 6
CHANNELS = 3 # [series1, series2, parallel]
TERMINALS = 2 # [ground, +ve]
"""

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
        if connection_type == 's1':
            self.embedding[r1, c1, r2, c2, 0] = True
            self.embedding[r2, c2, r1, c1, 1] = True
        elif connection_type == 's2':
            self.embedding[r1, c1, r2, c2, 1] = True
            self.embedding[r2, c2, r1, c1, 0] = True
        elif connection_type == 'p':
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
                        if self.embedding[r,c,r1,c1,1] == True and self.embedding[r,c,r1,c1,2] == True:
                            return "Invalid embedding: Connection between two"\
                                + "cells cannot be both series and parallel."\
                                + "Error occurred at " + str(r) + str(c)\
                                + str(r1) + str(c1)
                        elif self.embedding[r,c,r1,c1,0] == True and self.embedding[r,c,r1,c1,2] == True:
                            return "Invalid embedding: Connection between two"\
                                + "cells cannot be both series and parallel."\
                                + "Error occurred at " + str(r) + str(c)\
                                + str(r1) + str(c1)
                        elif self.embedding[r,c,r1,c1,0] == True and self.embedding[r,c,r1,c1,1] == True:
                            return "Invalid embedding: Connection between two"\
                                + "cells cannot be both types of series."\
                                + "Error occurred at " + str(r) + str(c)\
                                + str(r1) + str(c1)
        return True
    
    def validate_embedding(self):
        discovered1 = []
        discovered2 = []
        start_nodes = np.argwhere(self.terminal_array[:,:,1] == True)
        start_nodes = [(cell[0], cell[1]) for cell in start_nodes]
        end_nodes = np.argwhere(self.terminal_array[:,:,0] == True)
        end_nodes = [(cell[0], cell[1]) for cell in end_nodes]

        def recursive_dfs(l, discovered):
            # label all as discovered
            for cell in l:
                discovered.append(cell)
                
            for cell in l:
                r, c = cell[0], cell[1]
                alls1 = self.embedding[r,c,...,0]
                alls2 = self.embedding[r,c,...,1]
                allparallel = self.embedding[r,c,...,2]
                
                adj_array = alls1 | alls2 | allparallel
                
                adj_cells = np.argwhere(adj_array == True)
                adj_cells = [(cell[0], cell[1]) for cell in adj_cells]
                
                for c in adj_cells:
                    if c not in discovered:
                        recursive_dfs([c], discovered)
                        
                   
        recursive_dfs(start_nodes, discovered1)
        recursive_dfs(end_nodes, discovered2)

        discovered = discovered1 + discovered2
        discovered = set(discovered)
        all_cells = []
        for r in range(self.rows):
            for c in range(self.columns):
                all_cells.append((r, c))
        
        dangling = []
        for x in discovered1:
            if x not in all_cells:
                dangling.append(x)
        
        for cell in dangling:
            r, c = cell[0], cell[1]
            self.embedding([r,c,...,0]) = np.full((self.rows, self.columns),0, \
                                                  dtype=bool)
            self.embedding([r,c,...,1]) = np.full((self.rows, self.columns),0, \
                                                  dtype=bool)
            self.embedding([r,c,...,2]) = np.full((self.rows, self.columns),0, \
                                                  dtype=bool)
            self.embedding([...,r,c,0]) = np.full((self.rows, self.columns),0, \
                                                  dtype=bool)
            self.embedding([...,r,c,1]) = np.full((self.rows, self.columns),0, \
                                                  dtype=bool)
            self.embedding([...,r,c,2]) = np.full((self.rows, self.columns),0, \
                                                  dtype=bool)
        
# TODO: check connections to ground
# TODO: generate new embeddings directly by randomising True/False for embedding dimensions

"""
Model's output is the embedding structure - this will be run through the 
filtering function as a measure of performance

"""
#%% 
# TODO: delete dangling connections or cells that have no connection to terminals
# modify embedding before it is passed to "make_netlist" function

"""
string = '-0001021222+-0010202122+'
obj = string_to_embedding(3, 3, string)
obj.embedding[1,2,...] = np.zeros((3, 3, 3), dtype=bool)
obj.embedding[...,1,2,:] = np.zeros((3, 3, 3), dtype=bool)

discovered1 = []
discovered2 = []
start_nodes = np.argwhere(obj.terminal_array[:,:,1] == True)
start_nodes = [(cell[0], cell[1]) for cell in start_nodes]
end_nodes = np.argwhere(obj.terminal_array[:,:,0] == True)
end_nodes = [(cell[0], cell[1]) for cell in end_nodes]

def recursive_dfs(obj, l, discovered):
    # label all as discovered
    for cell in l:
        discovered.append(cell)
        
    for cell in l:
        r, c = cell[0], cell[1]
        alls1 = obj.embedding[r,c,...,0]
        alls2 = obj.embedding[r,c,...,1]
        allparallel = obj.embedding[r,c,...,2]
        
        adj_array = alls1 | alls2 | allparallel
        
        ground_connected = obj.terminal_array[r,c,0]
        
        adj_cells = np.argwhere(adj_array == True)
        adj_cells = [(cell[0], cell[1]) for cell in adj_cells]
        
        for c in adj_cells:
            if c not in discovered:
                recursive_dfs(obj, [c], discovered)
                
           
recursive_dfs(obj, start_nodes, discovered1)
recursive_dfs(obj, end_nodes, discovered2)

discovered1.sort()
discovered2.sort()
"""


# STACK (pop off list)
# branch to find the terminal


