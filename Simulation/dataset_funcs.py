from sim_functions import *
from params import *


# %% +++++++++DataSET++++++++++++++++

# Save multiple histograms for a shift list
def shifts_to_histogram(shift_list, ref=ref, filename=None, change_beam=False,
                        normalize=False, std=False, minmax=True,
                        verbose=False, monitor_bins=monitor_bins):
    
    if filename is None or change_beam:
        states = generate_secondary_particles(shifts, n_particles, verbose=verbose)
        particles = particles_from_states(states, ref, verbose=verbose)

    else:
        particles = import_particles_from_hdf5(filename, ref, verbose=verbose)

    height = monitor_bins[0]
    width = monitor_bins[1]
    n_magnet_settings = len(shift_list)
    n_changes = len(shift_list[0])
    histograms = np.zeros((n_magnet_settings, n_changes, width, height))

    for magnet_idx in range(n_magnet_settings):
        for i, shift in enumerate(shift_list[magnet_idx]):
            if is_array:
                shift = array_to_shifts(shift, shifts_template=shifts)

            if verbose: print(f"Processing shift {i+1}/{len(shift_list[magnet_idx])}: {shift}")
            line, env, ref = line_init(shifts=shift)
            if change_beam:
                states = generate_secondary_particles(shift, n_particles, verbose=verbose)
                particles = particles_from_states(states, ref, verbose=verbose)
            h, xedges, yedges = track_monitor(line, particles)
            histograms[magnet_idx, i] = h.T  # Transpose to match the orientation

        # After creating the entire batch of histograms for this magnet setting, normalize if needed
        if normalize:
            histograms[magnet_idx] = normalize_batch_hists(histograms[magnet_idx], std=std, minmax=minmax)
            if verbose: print(f"Normalized histograms for magnet setting {magnet_idx+1}/{n_magnet_settings}")

    return histograms, xedges, yedges


# Save histograms and shifts to HDF5 file
def save_to_hdf5(histograms, shift_list, magnet_settings, xedges, yedges, 
                 filename=histogram_dat, verbose=False):
    if verbose:
        start_time = time.time()
        print(f"Starting HDF5 save to {filename}")
    
    with h5py.File(filename, 'w') as f:
        # Store edges as global datasets
        f.create_dataset('xedges', data=xedges)
        f.create_dataset('yedges', data=yedges)

        f.create_dataset('magnet_settings', data=magnet_settings)
        f.create_dataset('shifts_list', data=shift_list)
        f.create_dataset('histograms', data=histograms)
    if verbose:
        end_time = time.time()
        print(f"HDF5 save completed in {end_time - start_time:.2f} seconds")


def import_histogram_data(filename=histogram_dat):
    with h5py.File(filename, 'r') as f:
        xedges = f['xedges'][:]
        yedges = f['yedges'][:]
        magnet_settings = f['magnet_settings'][:]
        histograms = f['histograms'][:]
        shifts_list = f['shifts_list'][:]
    return xedges, yedges, magnet_settings, histograms, shifts_list


def ranges_to_array(ranges, n):
    r_arr = np.zeros((n,2), dtype=np.float32)
    
    def req_ranges_to_array(data, arr, idx):
        for key, value in data.items():
            if isinstance(value, dict):
                idx = req_ranges_to_array(value, arr, idx)
            elif isinstance(value, (list, tuple, int, float)):
                arr[idx, :] = value
                idx += 1
        return idx

    req_ranges_to_array(ranges, r_arr, 0)
    return r_arr

def shifts_to_array(shifts, n):
    s_arr = np.zeros((n,), dtype=np.float32)
    
    def req_shifts_to_array(data, arr, idx):
        for key, value in data.items():
            if isinstance(value, dict):
                idx = req_shifts_to_array(value, arr, idx)
            else:
                arr[idx] = value
                idx += 1
        return idx

    req_shifts_to_array(shifts, s_arr, 0)
    return s_arr

def array_to_shifts(s_arr, shifts_template):
    shifts = deepcopy(shifts_template)
    
    def req_array_to_shifts(arr, data, idx):
        for key, value in data.items():
            if isinstance(value, dict):
                idx = req_array_to_shifts(arr, value, idx)
            else:
                data[key] = arr[idx]
                idx += 1
        return idx

    req_array_to_shifts(s_arr, shifts, 0)
    return shifts


# RANDOMAZATION MASTER
def shifts_array_random(shifts, shifts_range, n_samples, magnet_settings=[490], is_array = False):
    """ 
    Create a matrix of random shift configurations.
    Each row corresponds to a different magnet setting.
    Each column corresponds to a different random sample of shifts within the specified ranges.
    Args:
        shifts: Base shifts dictionary to copy and modify
        shifts_range: Dictionary specifying the range for each element and setting to randomize
                      e.g., {'q0': {'x': (-0.001, 0.001), 'y': (-0.001, 0.001)}, 'q1': {'ang_z': (-0.01, 0.01)}}
        n_samples: Number of random samples to generate for each magnet setting
        magnet_settings: List of magnet settings to iterate over
    Returns:
        shift_matrix: A 2D list where shift_matrix[magnet_idx][i] 
            has magnetSettings=magnet_settings[magnet_idx]
                      and is the i-th random sample of shifts within the specified ranges
                      defined in shifts_range
    """
    # Generate random number for seeding
    seed = np.random.randint(0, 100000)
    if is_array:
        shift_matrix = np.zeros((len(magnet_settings), n_samples, num_shifts), dtype=np.float32)
        range_array = ranges_to_array(shifts_range, n=num_shifts_range)
        for m_idx, mag_setting in enumerate(magnet_settings):
            np.random.seed(seed)
            shift_matrix[m_idx, :, 0] = mag_setting  # First column is magnet setting
            for s_idx, lowhigh in enumerate(range_array):
                # s_idx points to the setting in range_array, like s_idx=1 --> 'q0' 'x'
                # go through each setting and generate n_samples random values within the specified range
                low, high = lowhigh
                if low != high:
                    shift_matrix[m_idx, :, s_idx + 1] = np.random.uniform(low, high, size=n_samples)
                else:
                    shift_matrix[m_idx, :, s_idx + 1] = low  # If range is zero, just use the fixed value
    else:
        shift_matrix = []
        for mag_setting in magnet_settings:
            np.random.seed(seed)
            row = [] # Each row corresponds to a magnet setting
            shifts_copy = deepcopy(shifts)
            shifts_copy['magnetSettings'] = mag_setting

            # Generate n_samples random configurations corresponding to shift_ranges
            for _ in range(n_samples):
                shifts_sample = deepcopy(shifts_copy)
                for element, settings in shifts_range.items(): 
                    # element is like 'q0', settings is like {'x': (-0.001, 0.001), 'y': (-0.001, 0.001), ...}
                    if type(settings) is not dict:
                        continue  # Skip if settings is not a dictionary
                    for position, rand_range in settings.items():
                        # position is like 'x', min_val and max_val are the range limits
                        if type(rand_range) != tuple or len(rand_range) != 2:
                            continue  # Skip if range is not defined properly
                        # Sample a random value within the specified range
                        random_val = np.random.uniform(*rand_range)
                        shifts_sample[element][position] = random_val
                row.append(deepcopy(shifts_sample))
            shift_matrix.append(row)
    return shift_matrix




def normalize_batch_hists(hist_arr, std=False, minmax=True):
    """
    Normalize histograms in a batch array.
    
    Args:
        hist_arr: Array of shape (n_batch, h_height, h_width) containing histograms
    
    Returns:
        tuple: (z_normalized, minmax_normalized) where:
            - z_normalized: Zero mean, unit std normalized histograms
            - minmax_normalized: Min-max normalized histograms to [0,1]
    """
    # Z-score normalization (zero mean, unit std)
    # norm = hist_arr.copy()
    norm = hist_arr
    if std:
        for i in range(norm.shape[0]):
            batch = norm[i]
            mean = np.mean(batch)
            std = np.std(batch)
            if std > 0:  # Avoid division by zero
                norm[i] = (batch - mean) / std
            else:
                norm[i] = batch - mean
    
    # Min-max normalization to [0,1]
    if minmax:
        for i in range(norm.shape[0]):
            batch = norm[i]
            min_val = np.min(batch)
            max_val = np.max(batch)
            if max_val > min_val:  # Avoid division by zero
                norm[i] = (batch - min_val) / (max_val - min_val)
            else:
                norm[i] = np.zeros_like(batch)
    
    return norm


def plot_shift_array(shift_list, magnet_settings, n_max=5, name='q0', setting='x', verbose=False):

    histograms, xedges, yedges = shifts_to_histogram(shift_list, filename=dat_file, change_beam=False,
                                                      verbose=verbose, 
                                                      normalize=True, std=False, minmax=True)
    # Plot the histograms

    fig, axs = plt.subplots(len(shift_list), min(len(shift_list[0]), n_max), figsize=(len(magnet_settings)*6, 5), 
                            tight_layout=True, sharex=True, sharey=True)

    for magnet_idx, m in enumerate(magnet_settings):
        for i, shift in enumerate(shift_list[magnet_idx]):
            if is_array:
                shift = array_to_shifts(shift, shifts_template=shifts)
            if verbose: print(f"{m}: Started plotting change ", i+1, " of ", len(shift_list[magnet_idx]))
            ax = axs[magnet_idx, i]
            h = histograms[magnet_idx][i]
            ax.imshow(h, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
            change = shift[name][setting]
            ax.set_title(f'{magnet_settings[magnet_idx]} {name} {setting} = {change*1e3:.2f} mm')

            ax.xaxis.set_minor_locator(AutoMinorLocator(10))
            ax.yaxis.set_minor_locator(AutoMinorLocator(10))
            ax.grid(True,linewidth=0.25,alpha=0.25,which='major')
            if verbose: print(f"{m}: Finished plotting change ", i+1, " of ", len(shift_list))
    # Create FFT plot
    fig_fft, axs_fft = plt.subplots(len(shift_list), len(shift_list[0]), figsize=(len(magnet_settings)*6, 5), 
                                    tight_layout=True, sharex=True, sharey=True)

    for magnet_idx, m in enumerate(magnet_settings):
        for i, shift in enumerate(shift_list[magnet_idx]):
            if is_array:
                shift = array_to_shifts(shift, shifts_template=shifts)
            
            ax_fft = axs_fft[magnet_idx, i]
            h = histograms[magnet_idx][i]
            
            # Compute 2D FFT
            fft_h = np.fft.fft2(h)
            fft_h_shifted = np.fft.fftshift(fft_h)
            magnitude = np.abs(fft_h_shifted)
            
            # Plot FFT magnitude (log scale for better visualization)
            im = ax_fft.imshow(np.log(magnitude + 1), origin='lower', aspect='auto')
            
            change = shift[name][setting]
            ax_fft.set_title(f'FFT: {magnet_settings[magnet_idx]} {name} {setting} = {change*1e3:.2f} mm')
            
            ax_fft.xaxis.set_minor_locator(AutoMinorLocator(10))
            ax_fft.yaxis.set_minor_locator(AutoMinorLocator(10))
            ax_fft.grid(True,linewidth=0.25,alpha=0.25,which='major')

    return histograms, xedges, yedges

def plot_from_file(histograms=None, shift_list=None, xedges=None, yedges=None, magnet_settings=None,
                    filename=histogram_dat,
                    n_max=5, name='q0', setting='x', verbose=False):

    if histograms is None or shift_list is None or xedges is None or yedges is None or magnet_settings is None:
        xedges, yedges, magnet_settings, histograms, shift_list = import_histogram_data(filename)
    # Plot the histograms

    fig, axs = plt.subplots(len(shift_list), min(len(shift_list[0]), n_max), figsize=(len(magnet_settings)*6, 5), 
                            tight_layout=True, sharex=True, sharey=True)

    for magnet_idx, m in enumerate(magnet_settings):
        for i, shift in enumerate(shift_list[magnet_idx]):
            if is_array:
                shift = array_to_shifts(shift, shifts_template=shifts)
            if verbose: print(f"{m}: Started plotting change ", i+1, " of ", len(shift_list[magnet_idx]))
            ax = axs[magnet_idx, i]
            h = histograms[magnet_idx][i]
            ax.imshow(h, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
            change = shift[name][setting]
            ax.set_title(f'{magnet_settings[magnet_idx]} {name} {setting} = {change*1e3:.2f} mm')

            ax.xaxis.set_minor_locator(AutoMinorLocator(10))
            ax.yaxis.set_minor_locator(AutoMinorLocator(10))
            ax.grid(True,linewidth=0.25,alpha=0.25,which='major')
            if verbose: print(f"{m}: Finished plotting change ", i+1, " of ", len(shift_list))
    # Create FFT plot
    fig_fft, axs_fft = plt.subplots(len(shift_list), len(shift_list[0]), figsize=(len(magnet_settings)*6, 5), 
                                    tight_layout=True, sharex=True, sharey=True)

    for magnet_idx, m in enumerate(magnet_settings):
        for i, shift in enumerate(shift_list[magnet_idx]):
            if is_array:
                shift = array_to_shifts(shift, shifts_template=shifts)
            
            ax_fft = axs_fft[magnet_idx, i]
            h = histograms[magnet_idx][i]
            
            # Compute 2D FFT
            fft_h = np.fft.fft2(h)
            fft_h_shifted = np.fft.fftshift(fft_h)
            magnitude = np.abs(fft_h_shifted)
            
            # Plot FFT magnitude (log scale for better visualization)
            im = ax_fft.imshow(np.log(magnitude + 1), origin='lower', aspect='auto')
            
            change = shift[name][setting]
            ax_fft.set_title(f'FFT: {magnet_settings[magnet_idx]} {name} {setting} = {change*1e3:.2f} mm')
            
            ax_fft.xaxis.set_minor_locator(AutoMinorLocator(10))
            ax_fft.yaxis.set_minor_locator(AutoMinorLocator(10))
            ax_fft.grid(True,linewidth=0.25,alpha=0.25,which='major')

    return histograms, xedges, yedges


