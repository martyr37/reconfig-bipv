#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 22:58:44 2022

@author: mlima
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *
import PySpice.Logging.Logging as Logging
from solar_cell import SolarCell, all_series_bypass
from solar_module import SolarModule, generate_gaussian
from string_to_embedding import string_to_embedding, super_to_embedding
from circuit_embedding import CircuitEmbedding

#%%

read_in = pd.read_csv('shading_series.csv', header=0, names=['Shading Map'], usecols=[1])
print(read_in)
def convert_to_array(string):
    a = np.matrix(string).reshape(10, 6)
    a = np.array(a)
    return a
shading_series = [convert_to_array(s) for s in read_in['Shading Map']]

#%%
r = np.random.randint(0, 10000)
plt.title("Shading Map " + str(4))
plt.imshow(shading_series[4])

#%%
read_in = pd.read_csv('embedding_series.csv', header=0, names=['SuperString'], usecols=[1])
configurations = [x for x in read_in['SuperString']]
print(read_in)

#%%
s = np.random.randint(0, 999)
bypass_circuit = all_series_bypass(6, 10, shading_series[4])

bypass_circuit.V('input', 1, bypass_circuit.gnd, 0)
simulator = bypass_circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.dc(Vinput=slice(0,50,0.01))

bypassI = np.array(analysis.Vinput)
bypassV = np.array(analysis.sweep)
bypassP = bypassI * bypassV
bypassMPP = max(bypassP)
bypassVMP = bypassV[bypassP.argmax()]
bypassIMP = bypassI[bypassP.argmax()]

plt.plot(np.array(analysis.sweep), np.array(analysis.Vinput),alpha=0.5)

foo = super_to_embedding(10, 6, configurations[40])
foo.shading_map = shading_series[4]
foo.make_netlist()
foo.simulate()
foo.plot_netlist(50, 50)

print(foo.MPP)
print(bypassMPP)

#%%
#failed_strings = []
outfile = 'training_data.csv'
with open(outfile, 'a', newline='') as f:
    writer = csv.writer(f)
    header = ['Shading Map #', 'SuperString #', 'MPP', 'VMP', 'IMP', 'VOC', 'ISC', 'FF']
    #writer.writerow(header)
    for map_no in range(4332, 10000):
        shading_map = shading_series[map_no]
        for configuration_no in range(0, len(configurations)):
            superstring = configurations[configuration_no]
            moduleobj = super_to_embedding(10, 6, superstring)
            moduleobj.shading_map = shading_map
            moduleobj.make_netlist()
            try:
                moduleobj.simulate()
                row = [map_no, configuration_no, round(moduleobj.MPP, 2), \
                       round(moduleobj.VMP, 2), round(moduleobj.IMP, 2),\
                           round(moduleobj.VOC, 2), round(moduleobj.ISC, 2),\
                               round(moduleobj.FF, 2)]
                writer.writerow(row)
            except:
                #failed_strings.append([map_no, configuration_no])
                pass
            if configuration_no % 100 == 0:
                print('Completed configuration', configuration_no, end=' ')
        print('Completed shading map', map_no, end = ' ')