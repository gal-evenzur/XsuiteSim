import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import h5py
import numpy as np
import torch.nn.functional as F


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
    if isinstance(dat_raw, torch.Tensor):
        lib = torch
    else:
        lib = np

    if std:
        for i in range(dat_raw.shape[0]):
            batch = dat_raw[i]
            mean = lib.mean(batch)
            std = lib.std(batch)
            if std > 0:  # Avoid division by zero
                dat_raw[i] = (batch - mean) / std
            else:
                dat_raw[i] = batch - mean
    
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


    return dat_raw

def scale_Y(Y_raw, std=False, minmax=True):
    if not isinstance(Y_raw, torch.Tensor):
        Y_raw = torch.from_numpy(Y_raw).float()

    if std:
        mean = torch.mean(Y_raw, dim=0)
        std_val = torch.std(Y_raw, dim=0)
        Y_raw = (Y_raw - mean) / torch.where(std_val > 0, std_val, torch.ones_like(std_val))

    if minmax:
        min_val = torch.min(Y_raw, dim=0)[0]
        max_val = torch.max(Y_raw, dim=0)[0]
        range_val = max_val - min_val
        Y_raw = (Y_raw - min_val) / torch.where(range_val > 0, range_val, torch.ones_like(range_val))

    return Y_raw


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
        
    Returns:
        Downsampled tensor of shape (n_samples, n_channels, height, height)
    """
    n_samples, n_channels, height, width = tensor.shape
    
    if height == width:
        return tensor
    
    # Use interpolate to downsample width to match height
    # mode='bilinear' preserves spatial information better than cropping
    tensor_square = F.interpolate(
        tensor, 
        size=(height, height),  # Target size: (256, 256)
        mode='bilinear',
        align_corners=False
    )
    
    return tensor_square

class SignalDataset(Dataset):
    def __init__(self, data_path, split="train", ranges=None,
                 transform=scale_tensor, square_shape=True):
        # Determine file type based on extension
        # Load HDF5 data
        
        xedges, yedges, magnet_settings, histograms, shift_array, time_stamps = import_histograms_hd5(data_path, split=split)
        if ranges is not None:
            histograms = histograms[:, ranges[0]:ranges[1], :, :]
            shift_array = shift_array[:, ranges[0]:ranges[1], :]

        


        self.xedges = torch.from_numpy(xedges).float()
        self.yedges = torch.from_numpy(yedges).float()

        histograms = torch.from_numpy(histograms).float()
        # shape(histograms) = n_magnet_settings [=n_channels] x n_samples x 128 x 256
        
        # First, we want to reshape the data so that each sample is independent (using all magnet settings as channels)
        self.X = histograms.permute(1, 0, 2, 3)
        # shape(X) = n_samples x n_magnet_settings [=n_channels] x 256 x 128  

        # Downsample to square shape (256x256) to preserve all information
        if square_shape:
            self.X = squareinator(self.X)
            # shape(X) = n_samples x n_magnet_settings [=n_channels] x 256 x 256
        
        # Next, We'd like to scale each sample individually (notice we don't seperate magnet settings here):
        self.X = transform(self.X, std=False, minmax=False, const=True)

        self.shift_array = torch.from_numpy(shift_array).float()
        self.magnet_settings = torch.from_numpy(magnet_settings).float()
        # Shape(shift_array) = (n_magnets, n_samples, n_params) = (3, n_samples, 30)

        Y_raw = self.shift_array[0, :, 1:]  # Only keep the shifting parameters (exclude magnet settings)
        # shape(Y_raw) = (num_samples, n_params - 1) = (num_samples, 29)
        
        # Filter out non-variable parameters
        Y_filtered, self.active_param_indices = filter_variable_params(Y_raw)
        
        self.Y = scale_Y(Y_filtered, std=False, minmax=True)
        # Get a sample to check the full shape
        sample_input, sample_target = self.X[0], self.Y[0]
        print(f"--------{split} set: {len(self.X)} samples. range: {ranges} --------")
        print(f"Input shape: {sample_input.shape}")
        print(f"Target shape: {sample_target.shape}")


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx): #returns the 4 channels and the corresponding shifts
        return self.X[idx], self.Y[idx]

    def get_magnet_settings(self):
        return self.magnet_settings
    


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

    return train_dataset, val_dataset, test_dataset


