import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import h5py
import numpy as np
import torch.nn.functional as F
from ignite.metrics.metric import Metric


def import_histograms_hd5(filename, split='train'):
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


def scale_tensor(dat_raw, std=False, minmax=False, const=True):
    orig_dim = dat_raw.dim()
    dat_raw = dat_raw.unsqueeze(0) if dat_raw.dim() == 3 else dat_raw
    if isinstance(dat_raw, torch.Tensor):
        lib = torch
    else:
        lib = np

    if std:
        # Here dat_raw is expected to be of shape (n_samples, n_channels, height, width)
        for i in range(dat_raw.shape[0]):
            channel = dat_raw[i]
            mean = lib.mean(channel)
            std = lib.std(channel)
            if std > 0:  # Avoid division by zero
                dat_raw[i] = (channel - mean) / std
            else:
                dat_raw[i] = channel - mean
    
    # Min-max normalization to [0,1]
    if minmax:
        for i in range(dat_raw.shape[0]): # Go through each sample
            batch = dat_raw[i]
            min_val = lib.min(batch)
            max_val = lib.max(batch)
            if max_val > min_val:  # Avoid division by zero
                dat_raw[i] = (batch - min_val) / (max_val - min_val)
            else:
                dat_raw[i] = lib.zeros_like(batch)
            
    if const: # Scale by dividing by a constant factor 
        const_factor = 10
        if const_factor > 0:
            dat_raw = dat_raw / const_factor
        else:
            dat_raw = lib.zeros_like(dat_raw)

    dat_raw = dat_raw.squeeze(0) if orig_dim == 3 else dat_raw
    return dat_raw

def scale_Y(Y_raw, std=False, minmax=True):
    unscale_params = {}
    if not isinstance(Y_raw, torch.Tensor):
        Y_raw = torch.from_numpy(Y_raw).float()

    if std:
        mean = torch.mean(Y_raw, dim=0)
        std_val = torch.std(Y_raw, dim=0)
        Y_raw = (Y_raw - mean) / torch.where(std_val > 0, std_val, torch.ones_like(std_val))
        unscale_params['norm'] = (mean, std_val) 
        # shape(mean) = (n_params,), shape(std) = (n_params,)


    if minmax:
        min_val = torch.min(Y_raw, dim=0)[0]
        max_val = torch.max(Y_raw, dim=0)[0]
        range_val = max_val - min_val
        Y_raw = (Y_raw - min_val) / torch.where(range_val > 0, range_val, torch.ones_like(range_val))
        unscale_params['minmax'] = (min_val, max_val)
        # shape(min_val) = (n_params,), shape(max_val) = (n_params,)

    return Y_raw, unscale_params


def unscale_tensor(procss_data, params):

    if params.get('minmax'):
        mn = params.get('minmax')[0]
        mx = params.get('minmax')[1]
        procss_data = procss_data * (mx - mn) + mn

    if params.get('norm'):
        # First multiply by std, then add the mean
        procss_data = (procss_data * params.get('norm')[1]) + params.get('norm')[0]

    return procss_data

def filter_variable_params(Y_raw, threshold=1e-10):
    """
    Filter out columns where parameters don't vary (min ≈ max).
    
    Args:
        Y_raw: Tensor of shape (n_samples, n_params)
        threshold: Minimum range to consider a parameter variable
        
    Returns:
        Filtered tensor of shape (n_samples, n_active) and indices of active parameters
    """
    if not isinstance(Y_raw, torch.Tensor):
        Y_raw = torch.from_numpy(Y_raw).float()
    
    # Calculate min and max for each parameter (column)
    min_vals = torch.min(Y_raw, dim=0)[0]
    max_vals = torch.max(Y_raw, dim=0)[0]
    
    # Find columns where range is greater than threshold
    ranges = max_vals - min_vals
    active_mask = ranges > threshold
    active_indices = torch.where(active_mask)[0]
    
    # Filter to keep only variable columns
    Y_filtered = Y_raw[:, active_mask]
    
    print(f"Filtered from {Y_raw.shape[1]} to {Y_filtered.shape[1]} active parameters")
    print(f"Active parameter indices: {active_indices.tolist()}")
    
    return Y_filtered, active_indices

def squareinator(tensor):
    """
    Downsample the width dimension to match the height, creating square images.
    Uses bilinear interpolation to preserve information from all pixels.
    
    Args:
        tensor: Tensor of shape (n_samples, n_channels, height, width)
        
    Returns
        Downsampled tensor of shape (n_samples, n_channels, height, height)
    """
    tensor_4d = tensor.unsqueeze(0) if tensor.dim() == 3 else tensor  # Add batch dim if missing
    n_samples, n_channels, height, width = tensor_4d.shape
    
    if height == width:
        return tensor_4d

    # Use interpolate to downsample width to match height
    # mode='bilinear' preserves spatial information better than cropping
    tensor_square = F.interpolate(
        tensor_4d, 
        size=(height, height),  # Target size: (256, 256)
        mode='bilinear',
        align_corners=False
    )
    tensor_square = tensor_square.squeeze(0) if tensor.dim() == 3 else tensor_square  # Remove batch dim if added
    return tensor_square

class SignalDataset(Dataset):
    def __init__(self, data_path, split="train", transform=scale_tensor, square_shape=True):
        
        self.file = None
        self.data_path = data_path
        self.transform = transform
        self.square_shape = square_shape
        self.split = split


        with h5py.File(data_path, 'r') as f:
# --- Load small metadata (if needed) ---
            self.xedges = torch.from_numpy(f['xedges'][:]).float()
            self.yedges = torch.from_numpy(f['yedges'][:]).float()
            self.magnet_settings = torch.from_numpy(f[f'{split}/magnet_settings'][:]).float()
            shifts_list = f[f'{split}/shifts_list'][:]

        shift_array = torch.from_numpy(shifts_list).float()
        # Shape(shift_array) = (n_samples, n_magnets, n_params) = (n_samples, 3, 30)

        Y_raw = shift_array[:, 0, 1:]  # Only keep the shifting parameters (exclude magnet settings)
        # shape(Y_raw) = (num_samples, n_params - 1) = (num_samples, 29)
        
        # Filter out non-variable parameters
        Y_filtered, self.active_param_indices = filter_variable_params(Y_raw)
        
        self.Y, self.unscale_Y = scale_Y(Y_filtered, std=False, minmax=True)
    
        print(f"SignalDataset initialized with {self.Y.shape[0]} samples and {self.Y.shape[1]} active parameters.")

    def __len__(self):
        # Return the number of samples
        return self.Y.shape[0]

    def __getitem__(self, idx): #returns the 4 channels and the corresponding shifts
        # 1. Open file handle if this worker doesn't have one
        if self.file is None:
            self.file = h5py.File(self.data_path, 'r')
        
        histogram = self.file[f'{self.split}/histograms'][idx]

        X_sample = torch.from_numpy(histogram).float()
        # shape(histogram) = n_magnet_settings [=n_channels] x 128 x 256
        
        # Downsample to square shape (256x256) to preserve all information
        if self.square_shape:
            X_sample = squareinator(X_sample)
            # shape(X) = n_samples x n_magnet_settings [=n_channels] x 256 x 256
        
        # Next, We'd like to scale each sample individually (notice we don't seperate magnet settings here):
        # Just divides by a constant factor of 10 -- > Maybe change later
        self.X_sample = self.transform(X_sample, std=True, minmax=False, const=False)
        self.Y_sample = self.Y[idx]

        return self.X_sample, self.Y_sample

    def get_magnet_settings(self):
        return self.magnet_settings
    
    def get_unscale_params(self):
        return self.unscale_Y


def CreateTrainValTest(data_path, train_per, val_per, seed=42):
    full_dataset = SignalDataset(data_path=data_path)
    # 2. Get the total length
    dataset_size = len(full_dataset)

    # 3. Calculate the sizes for each split
    train_size = int(dataset_size * train_per)
    val_size = int(dataset_size * val_per)
    test_size = dataset_size - train_size - val_size # Ensures all data is used

    # 4. Perform the splits
    #    Note: random_split can take more than two lengths
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed) # for reproducible splits
    )
    # To get unscale parameters:
    # unscale_params = train_dataset.dataset.get_unscale_params()

    return train_dataset, val_dataset, test_dataset


class perc_error_per_parameter(Metric):
    '''Computes the percentage error per parameter for regression tasks.
    Returns tensor of shape (n_parameters,) with average percentage error for each parameter. 
    '''
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        super(perc_error_per_parameter, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        self.sum_errors = None
        self.num_examples = 0

    def update(self, output):
        y_pred, y = output
        abs_errors = torch.abs(y_pred - y)
        perc_errors = abs_errors / torch.clamp(torch.abs(y), min=1e-8) * 100  # Avoid division by zero
        batch_sum = torch.sum(perc_errors, dim=0)
        
        if self.sum_errors is None:
            self.sum_errors = batch_sum
        else:
            self.sum_errors += batch_sum
        self.num_examples += y.shape[0]

    def compute(self):
        return self.sum_errors / self.num_examples if self.num_examples > 0 else 0.0


class prediction_diff_histograms(Metric):
    '''
    Returns a histogram of prediction differences for each parameter. 
    returns a list of histograms, one per parameter.
    '''

    def __init__(self, output_transform=lambda x: x, device="cpu"):
        super(prediction_diff_histograms, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        self.diff_matrix = None
        self.n_samples = 0

    def update(self, output):
        y_pred, y = output
        # y, y_pred shape: (batch_size, n_parameters)
        abs_errors = torch.abs(y_pred - y)
        perc_errors = abs_errors / torch.clamp(torch.abs(y), min=1e-8) * 100  # Avoid division by zero
        perc_errors_param_first = perc_errors.transpose(0, 1)  # shape: (n_parameters, batch_size)
        
        if self.diff_matrix is None:
            self.diff_matrix = perc_errors_param_first
        else:
            self.diff_matrix = torch.cat((self.diff_matrix, perc_errors_param_first), dim=1) 
            # dim=0 is param dimension, dim=1 is batch dimension
        self.n_samples += y.shape[0]

    def compute(self):
        n_parameters = self.diff_matrix.shape[0]
        n_bins = 50  # Number of histogram bins
        histograms = np.zeros((n_parameters, n_bins))
        bin_edges = np.zeros((n_parameters, n_bins + 1))

        for i in range(n_parameters):
            max_val = torch.max(self.diff_matrix[i]).item()
            bin_edges[i] = np.logspace(np.log10(0.01), np.log10(max_val), n_bins + 1)
            histograms[i], bin_edges[i] = np.histogram(
                self.diff_matrix[i].cpu().numpy(), 
                bins=bin_edges[i]
            )

        print(f"Calculated diff histograms for {n_parameters} parameters over {self.n_samples} samples.")
        print(f"Histogram shape: {histograms.shape}, Bin edges shape: {bin_edges.shape}")
        return (histograms, bin_edges)


class find_outliers(Metric):
    '''
    Identifies outliers based on a percentage error threshold.
    Returns the id of outlier samples for each parameter.
    '''
    def __init__(self, threshold=1000.0, output_transform=lambda x: x, device="cpu"):
        self.threshold = threshold
        super(find_outliers, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        self.outliers = {}
        self.sample_count = 0

    def update(self, output):
        y_pred, y = output
        batch_size = y.shape[0]
        n_params = y.shape[1]

        abs_errors = torch.abs(y_pred - y)
        perc_errors = abs_errors / torch.clamp(torch.abs(y), min=1e-8) * 100
        
        # Identify outliers
        is_outlier = perc_errors > self.threshold
        # shape = (batch_size, n_params)
        
        # Generate global indices for this batch
        # This assumes the metric is updated sequentially over the dataset
        batch_indices = torch.arange(self.sample_count, self.sample_count + batch_size, device=y.device)
        # shape = (batch_size,)
        
        for i in range(n_params):
            # Get indices for this parameter where error is high
            param_outliers = batch_indices[is_outlier[:, i]] 
            if len(param_outliers) > 0:
                if i not in self.outliers:
                    self.outliers[i] = []
                self.outliers[i].extend(param_outliers.cpu().tolist())

        self.sample_count += batch_size

    def compute(self):
        for param_idx, outlier_ids in self.outliers.items():
            print(f"Parameter {param_idx}: {len(outlier_ids)} outliers.")
        return self.outliers

