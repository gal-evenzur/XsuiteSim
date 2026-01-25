import os
import numpy as np
import array
import pickle
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'afmhot'

pydir = os.path.dirname(os.path.abspath(__file__)) 

fpklcfgname = f"{pydir}/quads_impact_fist_chip.pkl"
fpkl = open(fpklcfgname,'rb')
data = pickle.load(fpkl)

# Print the structure of the loaded data
print(f"Data type: {type(data)}")
print(f"Data shape/length: {getattr(data, 'shape', len(data)) if hasattr(data, '__len__') else 'N/A'}")

if isinstance(data, dict):
    print("Dictionary keys:", list(data.keys()))
    for key, value in data.items():
        print(f"  {key}: {type(value)} - {getattr(value, 'shape', len(value)) if hasattr(value, '__len__') else value}")
elif isinstance(data, (list, tuple)):
    print(f"First few elements: {data[:3] if len(data) > 3 else data}")
    if data and hasattr(data[0], 'shape'):
        print(f"Element shape: {data[0].shape}")
elif isinstance(data, np.ndarray):
    print(f"Array dtype: {data.dtype}")
    print(f"Array contents preview:\n{data}")
else:
    print(f"Data preview: {data}")


run490 = data['2D_m34_1']
plt.imshow(run490, aspect='auto')
plt.colorbar()
plt.show()
fpkl.close()
