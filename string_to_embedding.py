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

import regex as re
import random

####################################################################################################

#test_string = '-726002147505527370130091036423+-926283843233152110543590(71019325)113095208004426365(12824140504331858145442461947455533451)22+'
#test_string = '-3354053262234364131491857283758265706010735502+-80613431049071632101157445535293001122510324944020844250951244924125358130+'
test_string = '[-836153(5051607152939282)63+-72(9081)8062(7073)91+][-7555+-7494656484958554+][-40+-13151411200325+-0224450132214142430022343012(4433)10+-352331(0504)+]'
#est_string = '[-65938182715461554094(45724495)5041807584517043836274606452+-928542639153(9073)+][-030201253413213315312005242311121435302200043210+]'
def string_to_embedding(rows, columns, string, moduleobj=None):
    if moduleobj == None:
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
            previous_cell = '' #REMOVE IF THIS BREAKS FUNCTIONALITY
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
            previous_cell = ''
            
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
                                                              ri, ci, rj, cj, 's1')
                    # connect cell in series to all in brackets '...22(3344)'
                    # 22 should be in series with 33 and 44
                    if previous_cell != '':
                        r1, c1 = int(previous_cell[0]), int(previous_cell[1])
                        for element in elements:
                            r2, c2 = int(element[0]), int(element[1])
                            moduleobj.make_connection(r1, c1, r2, c2, 's1')
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
                    if string[max(0, index - 2)] == ')':
                        group = bracket_groups[bracket_counter]
                        elements = [group[i:i+2] for i in range(0, len(group), 2)]
                        for element in elements:
                            r1, c1 = int(element[0]), int(element[1])
                            moduleobj.make_connection(r, c, r1, c1, 's2')
                    # normal series connection e.g. 2233
                    elif previous_cell != '':
                        r1, c1 = int(previous_cell[0]), int(previous_cell[1])
                        moduleobj.make_connection(r, c, r1, c1, 's2')
                previous_cell = cell
                cell = ''
        else:
            return "Error - Invalid character " + char
                    
    return moduleobj

#%% super_string_to_embedding function
def super_to_embedding(rows, columns, string):
    moduleobj = SolarModule(rows, columns)
    
    sans_neg = re.sub(r'(?<=\[)-', '', string)
    #sans_pos = re.sub(r'\+', '', sans_neg)
    blocks = re.findall(r'\[.+?\]', sans_neg)
    
    block_locations = [sans_neg.index(x) for x in blocks]
    
    l = []
    for index in range(0, len(sans_neg)):
        if block_locations != []:
            if index == block_locations[0]:
                l.append(blocks.pop(0)) 
                block_locations.pop(0)
        if sans_neg[index] == '{' or sans_neg[index] == '}':
            l.append(sans_neg[index])
    
    for index in range(len(l)):
        if l[index] != '{':
            l[index] = '[-' + l[index][1:]
            break
        
    curly = 0
    previous_cells = []
    
    for index in range(len(l) - 1, -1, -1):
        if l[index] != '}':
            last_chunk = l[index]
            break
        
    for index in range(len(l)):
        chunk = l[index]
        pos_cells = re.findall(r'[0-9]{2}(?=\)?\+)', chunk)
        if chunk != last_chunk:
            s = re.sub(r'\+', '', chunk)
        elif chunk == last_chunk:
            s = chunk
        s = s.lstrip('[')
        s = s.rstrip(']')
        if chunk == '{':
            curly = 1
        elif chunk == '}':
            curly = 0
        else:
            # connect initial cell to previous cell
            first_cell = re.search(r'\(?[0-9]{2}', s).group(0)
            first_cell = first_cell.lstrip('(')
            for cell in previous_cells:
                r1, c1 = int(cell[0]), int(cell[1])
                r2, c2 = int(first_cell[0]), int(first_cell[1])
                if curly == 0 or curly == 1:
                    moduleobj.make_connection(r1, c1, r2, c2, 's1') # it's still s1 after {
                elif curly == 2:
                    moduleobj.make_connection(r1, c1, r2, c2, 'p')
            #print(previous_cells)
            if curly == 1:
                curly = 2
            string_to_embedding(rows, columns, s, moduleobj=moduleobj)
            previous_cells = pos_cells
    
    return moduleobj

#obj = super_to_embedding(10, 6, test_string)
#obj.make_netlist()
#obj.simulate()
#obj.plot_netlist()

#%% generate_string function (copy-pasted)
def generate_string(rows, columns, adjacent = False, start_col = 0, start_row = 0):
    cell_ids = []
    for row in range(start_row, rows + start_row):
        for column in range(start_col, columns + start_col):
            cell_ids.append(str(row) + str(column))
    
    l_bracket = '('
    r_bracket = ')'
    
    if adjacent == False:
        random.shuffle(cell_ids) # shuffle cell order
            
    maximum_brackets = int((columns * rows) / 2)
    number_of_brackets = random.randint(0, maximum_brackets)
    for x in range(0, number_of_brackets):
        if x == 0:
            inserting_index = random.randint(0, len(cell_ids) - 2)
            cell_ids.insert(inserting_index, l_bracket)
            rb_inserting_index = random.randint(inserting_index + 3, len(cell_ids))
            cell_ids.insert(rb_inserting_index, r_bracket)
        else:
            inserting_index = random.randint(rb_inserting_index + 1, len(cell_ids) - 2) # ensuring next set of brackets is after the last
            cell_ids.insert(inserting_index, l_bracket)
            rb_inserting_index = random.randint(inserting_index + 3, len(cell_ids))
            cell_ids.insert(rb_inserting_index, r_bracket) 
        
        sliced_cell_ids = cell_ids[rb_inserting_index + 1:]
        if len(sliced_cell_ids) < 2:
            break
        
    pm = '+-'
    
    maximum_pms = max(columns, rows) - 1
    
    number_of_pms = random.randint(0, maximum_pms)
    
    for x in range(0, number_of_pms):
        random_index = random.randint(0, len(cell_ids))
        sliced_cell_ids = cell_ids[:random_index]
        number_of_l_brackets = sliced_cell_ids.count(l_bracket)
        number_of_r_brackets = sliced_cell_ids.count(r_bracket)
        if number_of_l_brackets == number_of_r_brackets:
            cell_ids.insert(random_index, pm)
    
    cell_ids.insert(0, '-')
    cell_ids.append('+')
    
    pattern1 = re.compile(r'(?<=^-.*)(?:\+-)+(?=\+$)') # delete excess trailing +- 
    pattern2 = re.compile(r'(?<=^-)(?:\+-)+(?=.*$)') # delete excess leading +-
    pattern3 = re.compile(r'(\+-)+(?=(\+-)+)') # delete excess +- in middle of string
    out = re.sub(pattern1, '', "".join(cell_ids))
    out = re.sub(pattern2, '', out)
    out = re.sub(pattern3, '', out)
    """
    try:
        interconnection("".join(cell_ids), columns, rows, uniform_shading(rows, columns))
        return "".join(cell_ids)
    except:
        generate_string(columns, rows)
    """
    return out

#%% string_to_embedding function testing testing
"""
string_list = []
for x in range(20):
    string = generate_string(6, 10)
    string_list.append(string)
    obj = string_to_embedding(10, 6, string)
    obj.make_netlist()
    obj.simulate()
    obj.plot_netlist() 
"""