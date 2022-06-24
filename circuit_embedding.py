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
ROWS = 10
COLUMNS = 6
CHANNELS = 5 # [connection, series, parallel, ground, +ve]
# TODO: separate ground and +ve channels into another array

#%% embedding creation
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
        embedding[r1, c1, r2, c2, 0] = True
        embedding[r2, c2, r1, c1, 0] = True
        embedding[r1, c1, r2, c2, 2] = True
        embedding[r2, c2, r1, c1, 2] = True
    return embedding

def connect_to_ground(embedding, r, c):
    embedding[r,c,:,:,3] = True
    return embedding

def connect_to_pos(embedding, r, c):
    embedding[r,c,:,:,4] = True
    return embedding
 
#%% validation function
def check_embedding(embedding):
    # Cells cannot have a True connection value to themselves
    # The reverse connection should be the same type
    # Cannot have both a series and parallel connection simultaneously
    # gnd and +ve terminals must have completely True or False dimensions
    rows, columns = embedding.shape[0], embedding.shape[1]
    for r in range(rows):
        for c in range(columns):
            if embedding[r,c,r,c,0] == True:
                return "Invalid embedding: The cell cannot have a connection"\
                    + " to itself. Error occured at " + str(r) + str(c)
            if embedding[r,c,r,c,1] == True:
                return "Invalid embedding: The cell cannot have a connection"\
                    + " to itself. Error occured at " + str(r) + str(c)
            if embedding[r,c,r,c,2] == True:
                return "Invalid embedding: The cell cannot have a connection"\
                    + " to itself. Error occured at " + str(r) + str(c)
            # TODO: allow series + none, parallel + none connections
            for r1 in range(rows):
                for c1 in range(columns):
                    if embedding[r, c, r1, c1, 0] != embedding[r1, c1, r, c, 0]:
                        return "Invalid embedding: The reverse connection"\
                            + "must be the same. Error occured between " + \
                                " ".join([r,c,r1,c1])
                                
                    if embedding[r, c, r1, c1, 1] != embedding[r1, c1, r, c, 1]:
                        return "Invalid embedding: The reverse connection"\
                            + "must be the same. Error occured between " + \
                                " ".join([r,c,r1,c1])
                                
                    if embedding[r, c, r1, c1, 2] != embedding[r1, c1, r, c, 2]:
                        return "Invalid embedding: The reverse connection"\
                            + "must be the same. Error occured between " + \
                                " ".join([r,c,r1,c1])
                        
                    if embedding[r,c,r1,c1,0] == False:
                        if embedding[r,c,r1,c1,1] == True or embedding[r,c,r1,c1,2] == True:
                            return "Invalid embedding: No connection + some connection "\
                                + "is invalid. Error occured at " + str(r) + str(c)

                    if embedding[r,c,r1,c1,0] == True:
                        if embedding[r,c,r1,c1,1] == False and embedding[r,c,r1,c1,2] == False:
                            return "Invalid embedding: Some connection + no connection "\
                                + "is invalid. Error occured at " + str(r) + str(c)
    return "Valid"
#array[6,10,3,3,1] = True

#%% Convetional series module
def series_embedding(rows, columns, channels):
    embedding = create_empty_embedding(rows, columns, channels)
    
    for r in range(rows):
        if r % 2 == 0:
            for c in range(columns):
                if r == 0 and c == 0:
                    # top-left cell connected to gnd
                    connect_to_ground(embedding, r, c)
                    make_connection(embedding, r, c, r, c+1, 's')
                elif r == rows - 1 and c == columns - 1:
                    # make final connection to +ve terminal
                    connect_to_pos(embedding, r, c)
                elif c == columns - 1:
                    # if at the right end, make connection with cell below
                    make_connection(embedding, r, c, r + 1, c, 's')
                else:
                    # cell to its right is connected in series
                    make_connection(embedding, r, c, r, c+1, 's')
        elif r % 2 == 1:
            for c in range(columns - 1, -1, -1):
                if r == rows - 1 and c == 0:
                    # make final connection to +ve terminal
                    connect_to_pos(embedding, r, c)
                elif c == 0:
                    # make connection to row below
                    make_connection(embedding, r, c, r + 1, c, 's')
                else:
                    make_connection(embedding, r, c, r, c-1, 's')
    return embedding

array = series_embedding(ROWS, COLUMNS, CHANNELS)  

#%% Total-cross-tied module
def tct_embedding(rows, columns, channels):
    embedding = create_empty_embedding(rows, columns, channels)
    for c in range(columns):
        for r in range(rows):
            # column to the right is all 1's (series)
            if c != columns - 1:
                make_connection(embedding, r, c, r, c+1, 's')
            if r != rows - 1:
                make_connection(embedding, r, c, r+1, c, 'p')
        
    return embedding
    
#array = tct_embedding(ROWS, COLUMNS, CHANNELS)  
    
#%% Create PySpice netlist from embedding
intensity_array = np.full((ROWS, COLUMNS), 10)

def make_netlist(embedding):
    cell_id = 1 # use separator
    circuit = Circuit('Netlist')
    rows, columns = embedding.shape[0], embedding.shape[1]
    for r in range(rows):
        for c in range(columns):
            circuit.subcircuit(SolarCell(cell_id, \
                                intensity=intensity_array[r,c]))
            cell_id += 1
    
    for r in range(rows):
        for c in range(columns):
            # make netlist
            pass

#make_netlist(array)
# TODO: recreate basic connections (series, parallel, SP, TCT) 
# TODO: topology to netlist
