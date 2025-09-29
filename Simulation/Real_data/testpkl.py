import os
import numpy as np
import array
import pickle

fpklcfgname = "quads_impact_fist_chip.pkl"
fpkl = open(fpklcfgname,'rb')
data = pickle.load(fpkl)
for name, arr in data.items():
    print(name)
    print(arr)
fpkl.close()
