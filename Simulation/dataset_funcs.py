from sim_functions import *
from params import *
import time


def rand_from_scratch_histogram(shifts_template, shifts_range, n_batch, ref=ref, magnet_settings=[490], particles_file=None,
                                change_beam=False, normalize=False, std=False, minmax=True,
                                rng=default_rng(),
                                max_attempts=10,
                                verbose=False, monitor_bins=monitor_bins):
    
    if not change_beam and particles_file:
        particles = import_particles_from_hdf5(particles_file, ref, verbose=verbose)


    height = monitor_bins[0]
    width = monitor_bins[1]
    n_magnet_settings = len(magnet_settings)
    histograms = np.zeros((n_magnet_settings, n_batch, width, height))
    shift_matrix = np.zeros((len(magnet_settings), n_batch, num_shifts), dtype=np.float32)


    for i in range(n_batch):
        valid = False
        attempt = 0

        hist_start_time = time.time()
        while (not valid):
            attempt += 1
            if attempt > 1:
                if verbose: print(f"  Attempt {attempt} for batch {i+1}/{n_batch}")
            # Create a random shift configuration (n_magnet_settings, 1)
            shifts = shifts_array_random(shifts_template, shifts_range, 1,
             rng=rng,
             magnet_settings=magnet_settings, is_array=False)
            if change_beam:
                states = generate_secondary_particles(shifts[0][0], n_particles, verbose=False, rng=rng) # Generate for first magnet setting
                # DANGER __ WILL NOT WORK FOR 502 RUN ___ # 
                particles = particles_from_states(states, ref, verbose=verbose)
            for magnet_idx in range(n_magnet_settings):

                shift = shifts[magnet_idx][0]  # Extract the single configuration
                # Create a beam and caculate histogram
                line, env, ref = line_init(shifts=shift)
                h, xedges, yedges = track_monitor(line, particles)
                # It's enough so that only one magnet setting produces a valid histogram
                valid_temp = is_valid(h)

                if valid_temp:
                    valid = True

                histograms[magnet_idx, i] = h.T  # Transpose to match the orientation
                s_array_form = shifts_to_array(shift, n=num_shifts)
                shift_matrix[magnet_idx, i] = s_array_form
                
        if verbose:
            hist_end_time = time.time()
            print(f"Histogram {i+1}/{n_batch} for magnet setting {magnet_settings[magnet_idx]} took {hist_end_time - hist_start_time:.2f} seconds")
                    
    if normalize:
        for magnet_idx in range(n_magnet_settings):
            histograms[magnet_idx] = normalize_batch_hists(histograms[magnet_idx], std=std, minmax=minmax)
            if verbose: print(f"Normalized histograms for magnet setting {magnet_idx+1}/{n_magnet_settings}")

    return shift_matrix, histograms, xedges, yedges



# Save multiple histograms for a shift list
def shifts_to_histogram(shift_list, ref=ref, filename=None, change_beam=False,
                        rng=default_rng(),
                        normalize=False, std=False, minmax=True,
                        max_attempts=10,
                        verbose=False, monitor_bins=monitor_bins):
    
    if filename is None or change_beam:
        states = generate_secondary_particles(shifts, n_particles, verbose=False, rng=rng)
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

            valid = False
            # Repeat until a valid histogram is obtained
            attempt = 0
            while (not valid) and (attempt < max_attempts):
                attempt += 1
                if attempt > 1:
                    if verbose: print(f"  Attempt {attempt} for shift {i+1}/{len(shift_list[magnet_idx])}")
                    new_shifts = shifts_array_random(shifts, shifts_range, 1, 
                        rng=rng, 
                        magnet_settings=[shift['magnetSettings']], is_array=False)
                    shift = new_shifts[0][0]
                    s_array_form = shifts_to_array(shift, n=num_shifts)
                    shift_list[magnet_idx][i] = s_array_form

                # Create a beam and caculate histogram
                line, env, ref = line_init(shifts=shift)
                if change_beam:
                    states = generate_secondary_particles(shift, n_particles, verbose=False, rng=rng)
                    particles = particles_from_states(states, ref, verbose=verbose)
                h, xedges, yedges = track_monitor(line, particles)
                valid = is_valid(h)

            if verbose: print(f"Processing shift {i+1}/{len(shift_list[magnet_idx])}: {shift}")

            histograms[magnet_idx, i] = h.T  # Transpose to match the orientation

        # After creating the entire batch of histograms for this magnet setting, normalize if needed
        if normalize:
            histograms[magnet_idx] = normalize_batch_hists(histograms[magnet_idx], std=std, minmax=minmax)
            if verbose: print(f"Normalized histograms for magnet setting {magnet_idx+1}/{n_magnet_settings}")

    return histograms, xedges, yedges


# Save histograms and shifts to HDF5 file
def save_histogarms_hd5(histograms, shift_list, magnet_settings, xedges, yedges, dataset="train", write_add='w',
                 filename=histogram_dat, verbose=False):
    if verbose:
        start_time = time.time()
        print(f"Starting HDF5 save to {filename}")

    with h5py.File(filename, write_add) as f:
        # Store edges as global datasets
        if write_add == 'w':
            f.create_dataset('xedges', data=xedges)
            f.create_dataset('yedges', data=yedges)
        set = f.create_group(dataset)

        set.create_dataset('magnet_settings', data=magnet_settings)
        set.create_dataset('shifts_list', data=shift_list)
        set.create_dataset('histograms', data=histograms)
    if verbose:
        end_time = time.time()
        print(f"HDF5 save completed in {end_time - start_time:.2f} seconds")

def import_histograms_hd5(filename=histogram_dat, split='train'):
    with h5py.File(filename, 'r') as f:
        xedges = f['xedges'][:]
        yedges = f['yedges'][:]
        magnet_settings = f[f'{split}/magnet_settings'][:]
        histograms = f[f'{split}/histograms'][:]
        shifts_list = f[f'{split}/shifts_list'][:]
        try:
            time_stamps = f['time_stamps'][:]
        except: 
            print("No time_stamps")
            time_stamps = None
    return xedges, yedges, magnet_settings, histograms, shifts_list, time_stamps

# CREATE SHIFT LIST:--
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

def shift_list_to_matrix(shift_list, n):
    n_magnet_settings = len(shift_list)
    n_changes = len(shift_list[0])
    shift_matrix = np.zeros((n_magnet_settings, n_changes, n), dtype=np.float32)

    for magnet_idx in range(n_magnet_settings):
        for i, shift in enumerate(shift_list[magnet_idx]):
            s_array_form = shifts_to_array(shift, n=n)
            shift_matrix[magnet_idx, i] = s_array_form
    return shift_matrix

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

def shifts_array_random(shifts_template, shifts_range, n_samples, rng=default_rng(), magnet_settings=[490], is_array = False):
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
    if is_array:
        shift_matrix = np.zeros((len(magnet_settings), n_samples, num_shifts), dtype=np.float32)
        range_array = ranges_to_array(shifts_range, n=num_shifts_range)
        for m_idx, mag_setting in enumerate(magnet_settings):
            shift_matrix[m_idx, :, 0] = mag_setting  # First column is magnet setting
            for s_idx, lowhigh in enumerate(range_array):
                # s_idx points to the setting in range_array, like s_idx=1 --> 'q0' 'x'
                # go through each setting and generate n_samples random values within the specified range
                low, high = lowhigh
                if low != high:
                    shift_matrix[m_idx, :, s_idx + 1] = rng.uniform(low, high, size=n_samples)
                else:
                    shift_matrix[m_idx, :, s_idx + 1] = low  # If range is zero, just use the fixed value
    else:
        shift_matrix = []

        # Generate n_samples random configurations corresponding to shift_ranges
        row = [] # Each row corresponds to a magnet setting
        for _ in range(n_samples):
            shifts_sample = deepcopy(shifts_template)
            for element, settings in shifts_range.items(): 
                # element is like 'q0', settings is like {'x': (-0.001, 0.001), 'y': (-0.001, 0.001), ...}
                if type(settings) is not dict:
                    continue  # Skip if settings is not a dictionary
                for position, rand_range in settings.items():
                    # position is like 'x', min_val and max_val are the range limits
                    if type(rand_range) != tuple or len(rand_range) != 2:
                        continue  # Skip if range is not defined properly
                    # Sample a random value within the specified range
                    random_val = rng.uniform(*rand_range)
                    shifts_sample[element][position] = random_val
            row.append(deepcopy(shifts_sample))
        
        for mag_setting in magnet_settings:
            row_rundown = deepcopy(row)
            for shift in row_rundown:
                shift['magnetSettings'] = mag_setting
            shift_matrix.append(row_rundown)
    return shift_matrix

# Processing functions:--
def is_valid(h, threshold=1, point_threshold=10, verbose=False):
    a = np.sum(h > threshold)
    # Check if we have enough relevant points above the threshold
    if a <= point_threshold:
        if verbose: print("Error: Not enough data points above threshold")
        return False
    else:
        return True    
    
        


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


# Plotting functions:--
def plot_shift_array(shift_list, magnet_settings, 
                    rng=default_rng(),
                    normalize=False, n_max=5, name='q0', setting='x', verbose=False):
    '''
    Assumes that shift_list is in array form!!
    '''
    
    histograms, xedges, yedges = shifts_to_histogram(shift_list, filename=dat_file, change_beam=True,
                                                      verbose=verbose, 
                                                      normalize=normalize, std=False, minmax=True)
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

def plot(xedges, yedges, magnet_settings, histograms, shift_list,
         pdfname=None,
        split='train', n_max=5, name='q0', setting='x', verbose=False, fft=False):
    n_changes = min(len(shift_list[0]), n_max)
    fig, axs = plt.subplots(len(shift_list), n_changes, figsize=(len(magnet_settings)*6, 5), 
                            tight_layout=True, sharex=True, sharey=True)

    for magnet_idx, m in enumerate(magnet_settings):
        for i in range(n_changes):
            shift = shift_list[magnet_idx][i]
            shift = array_to_shifts(shift, shifts_template=shifts)
            if verbose: print(f"{m}: Started plotting change ", i+1, " of ", len(shift_list[magnet_idx]))
            if len(shift_list) == 1:
                ax = axs[i]
            elif n_changes == 1:
                ax = axs[magnet_idx]
            else:
                ax = axs[magnet_idx, i]
            h = histograms[magnet_idx][i]
            im = ax.imshow(h, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
            change = shift[name][setting]
            valid = is_valid(h)
            ax.set_title(f'{magnet_settings[magnet_idx]} {name} {setting} = {change*1e3:.2f} mm\n Valid: {valid}')

            ax.xaxis.set_minor_locator(AutoMinorLocator(10))
            ax.yaxis.set_minor_locator(AutoMinorLocator(10))
            ax.grid(True,linewidth=0.25,alpha=0.25,which='major')
            plt.colorbar(im, ax=ax)
            if verbose: print(f"{m}: Finished plotting change ", i+1, " of ", len(shift_list))

    if pdfname is not None:
        fig.savefig(pdfname)
        if verbose: print(f"Saved histogram figure to {pdfname}")

    # Create FFT plot

    if fft:
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

  


def plot_from_file(filename, pdf=True, 
                    split='train',
                    n_max=5, name='q0', setting='x', verbose=False, fft=False):
    '''
    Assumes that histograms, shift_list, xedges, yedges, magnet_settings are all None or all provided.
    Assume that shift_list is in array form!!
    '''

    xedges, yedges, magnet_settings, histograms, shift_list, time_stamps = import_histograms_hd5(filename, split=split)
    # Plot the histograms

    if pdf:
        pdfname = filename.replace('.h5', f'_{split}.pdf')
    else:
        pdfname = None

    plot(xedges, yedges, magnet_settings, histograms, shift_list,
         pdfname=pdfname,
         split=split, n_max=n_max, name=name, setting=setting, verbose=verbose, fft=fft)



