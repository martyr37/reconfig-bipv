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

####################################################################################################

#%% Circuit embedding tests
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


#%% 
ROWS = 10
COLUMNS = 6
CHANNELS = 5 # [connection, series, parallel, ground, +ve]

# ground and +ve channels are entirely True or False

def create_empty_embedding(rows, columns, channels):
    embedding = np.zeros((rows, columns, rows, columns, channels), dtype=bool)
    return embedding

def make_connection(embedding, r1, c1, r2, c2, connection_type):
    if connection_type == 's':
        embedding[r1, c1, r2, c2, 0] = True
        embedding[r2, c2, r1, c1, 0] = True
        embedding[r1, c1, r2, c2, 1] = True
        embedding[r2, c2, r1, c1, 1] = True
    elif connection_type == 'p':
        pass
    return embedding

def series_embedding(rows, columns, channels):
    embedding = create_empty_embedding(rows, columns, channels)
    
    for r in range(rows):
        if r % 2 == 0:
            for c in range(columns):
                if r == 0 and c == 0:
                    # top-left cell connected to gnd
                    embedding[r,c,:,:,3] = True
                    make_connection(embedding, r, c, r, c+1, 's')
                elif r == rows - 1 and c == columns - 1:
                    # make final connection to +ve terminal
                    embedding[r,c,:,:,4] = True
                elif c == columns - 1:
                    # if at the right end, make connection with cell below
                    make_connection(embedding, r, c, r + 1, c, 's')
                else:
                    # cell to its right is connected in series
                    make_connection(embedding, r, c, r, c+1, 's')
        elif row % 2 == 1:
            for c in range(columns - 1, -1, -1):
                if row == rows - 1 and c == 0:
                    # make final connection to +ve terminal
                    embedding[r,c,:,:,4] = True
                elif c == 0:
                    # make connection to row below
                    make_connection(embedding, r, c, r + 1, c, 's')
                else:
                    make_connection(embedding, r, c, r, c-1, 's')
    return embedding

def make_netlist(embedding):
    
array = series_embedding(ROWS, COLUMNS, CHANNELS)   


# TODO: recreate basic connections (series, parallel, SP, TCT) 
# TODO: topology to netlist
