import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['image.cmap'] = 'afmhot'

import pickle
pydir = os.path.dirname(os.path.abspath(__file__)) # This results ""
# Save Data to pickle file
pickle_filename = os.path.join(pydir, 'Data.pkl')

with open(pickle_filename, 'rb') as f:
    Data = pickle.load(f)

# shape(Data) = (len(magnet_settings), n_final_alive, 4, (par))

# par = (s_values, x[i], y[i], pz[i]) [ i = particle index ]
# now par is a tuple with (s_values, x_values, y_values, pz) for each particle
# where s_values is the same for all particles, but x_values, y_values, pz are different



def plot_histogram(x, y, bins, title=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist2d(x, y, bins=bins)
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_title(title)


def plot_histogram_from_data(Data, magnet_idx, monitor_idx):
    x = []
    y = []
    for par in Data[magnet_idx]:
        x.append(par[1][monitor_idx])
        y.append(par[2][monitor_idx])
    x = np.array(x)
    y = np.array(y)
    title = f"Magnet setting: {magnet_idx}, Monitor: {monitor_idx}"
    plot_histogram(x, y, bins=100, title=title)

plot_histogram_from_data(Data, magnet_idx=4, monitor_idx=0)

plot_histogram_from_data(Data, magnet_idx=4, monitor_idx=1)

plot_histogram_from_data(Data, magnet_idx=0, monitor_idx=1)

plt.show()
