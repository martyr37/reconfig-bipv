#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:38:44 2022

@author: mlima
"""

####################################################################################################
import numpy as np
import matplotlib.pyplot as plt

import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *
logger = Logging.setup_logging()

from solar_module import SolarModule
from circuit_embedding import CircuitEmbedding

import re

####################################################################################################

#test_string = '-12+-3435+-7045928443+-622203735365245563428240211133000151442590+-23819371943083521514746485919531(5441100561)60751372(80200232)(0450)+'
test_string = '-3354053262234364131491857283758265706010735502+-80613431049071632101157445535293001122510324944020844250951244924125358130+'

def string_to_embedding(rows, columns, string):
    moduleobj = SolarModule(rows, columns)

    bracket_groups = re.findall(r'\(([0-9]+)\)', string)
    
    connect_to_ground = False
    in_brackets = False
    bracket_counter = -1
    cell = ''
    previous_cell = ''
    for index in range(len(string)):
        char = string[index]
        if char == '-':
            connect_to_ground = True
        elif char == '+':
            # connect previous cell/cells to positive
            if string[index - 1].isnumeric() == True:
                r1, c1 = int(previous_cell[0]), int(previous_cell[1])
                moduleobj.connect_to_pos(r1, c1)
            elif string[index - 1] == ')':
                # if string has '-(1234)+', connect 12 and 34 to positive
                group = bracket_groups[bracket_counter]
                elements = [group[i:i+2] for i in range(0, len(group), 2)]
                for element in elements:
                    r1, c1 = int(element[0]), int(element[1])
                    moduleobj.connect_to_pos(r1, c1)
        elif char == '(':
            bracket_counter += 1
            in_brackets = True
        elif char == ')':
            if connect_to_ground == True: # if string starts '-(c1, c2)...'
                group = bracket_groups[bracket_counter]
                elements = [group[i:i+2] for i in range(0, len(group), 2)]
                for element in elements:
                    r1, c1 = int(element[0]), int(element[1])
                    moduleobj.connect_to_ground(r1, c1)
                connect_to_ground = False
            in_brackets = False
            
        elif char.isnumeric() == True: # if current character is a number
            if cell == '':
                cell = char
            elif len(cell) == 1:
                cell += char
                r, c = int(cell[0]), int(cell[1])
                if in_brackets == True:
                    # e.g. '-(2324)(1314)', 23 and 24 must have series 
                    # connections to 13 and 14
                    group = bracket_groups[bracket_counter]
                    elements = [group[i:i+2] for i in range(0, len(group), 2)]
                    if index >= 3:
                        if string[index - 3] == ')':
                            previous_group = bracket_groups[bracket_counter - 1]
                            i_list = [previous_group[i:i+2]\
                                      for i in range(0, len(previous_group), 2)]
                            j_list = elements
                            for i in i_list:
                                for j in j_list:
                                    ri, ci = int(i[0]), int(i[1])
                                    rj, cj = int(j[0]), int(j[1])
                                    moduleobj.make_connection(\
                                                              ri, ci, rj, cj, 's')
                    # connect cell in series to all in brackets '...22(3344)'
                    # 22 should be in series with 33 and 44
                    if previous_cell != '': # TODO: Fix if previous series cell
                    # is before another set of brackets (should ignore)
                        r1, c1 = int(previous_cell[0]), int(previous_cell[1])
                        moduleobj.make_connection(r, c, r1, c1, 's')
                        previous_cell = ''
                    # connect cells in brackets to each other in parallel
                    # (3344), 33 and 44 in parallel
                    for element in elements:
                        r1, c1 = int(element[0]), int(element[1])
                        if r != r1 or c != c1:
                            moduleobj.make_connection(r, c, r1, c1, 'p')   
                    cell = ''
                    continue
                
                elif connect_to_ground == True and in_brackets == False:
                    # if string starts '-22...'
                    moduleobj.connect_to_ground(r, c)
                    connect_to_ground = False
                else:
                    # string like '-(1234)56', 56 must have a series connection
                    # to 12 and 34
                    if string[index - 2] == ')':
                        group = bracket_groups[bracket_counter]
                        elements = [group[i:i+2] for i in range(0, len(group), 2)]
                        for element in elements:
                            r1, c1 = int(element[0]), int(element[1])
                            moduleobj.make_connection(r, c, r1, c1, 's')
                    # normal series connection e.g. 2233
                    elif previous_cell != '':
                        r1, c1 = int(previous_cell[0]), int(previous_cell[1])
                        moduleobj.make_connection(r, c, r1, c1, 's')
                previous_cell = cell
                cell = ''
        else:
            return "Error - Invalid character " + char
                    
    return moduleobj
                    
obj = string_to_embedding(10, 6, test_string)
        
                
    
    