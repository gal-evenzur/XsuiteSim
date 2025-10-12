import torch
import torch.nn as nn
from ignite.metrics.metric import reinit__is_reduced
from torch.utils.data import Dataset
import sys
import os
pyPath = os.path.dirname(os.path.abspath(__file__))
simPath = os.path.join(pyPath, 'Simulation')
sys.path.append(simPath)

from Simulation.dataset_funcs import import_histograms_hd5
import numpy as np

def scale_tensor(dat_raw, std=False, minmax=True):
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
            
    # print(f"data is {dat_raw.shape} - {min.shape} / ({max.shape} - {min.shape} ")
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

class SignalDataset(Dataset):
    def __init__(self, data_path, split="train", 
                 transform=scale_tensor):
        # Determine file type based on extension
        # Load HDF5 data
        
        xedges, yedges, magnet_settings, histograms, shift_array = import_histograms_hd5(data_path, split=split)
        
        self.xedges = torch.from_numpy(xedges).float()
        self.yedges = torch.from_numpy(yedges).float()

        histograms = torch.from_numpy(histograms).float()
        print("Histograms shape:", histograms.shape)
        # shape(histograms) = n_magnet_settings [=n_channels] x n_samples x 128 x 256
        
        # First, we want to reshape the data so that each sample is independent (using all magnet settings as channels)
        self.X = histograms.permute(1, 0, 2, 3)
        # shape(X) = n_samples x n_magnet_settings [=n_channels] x 128 x 256  
        
        # Next, We'd like to scale each sample individually (notice we don't seperate magnet settings here):
        self.X = transform(self.X, std=False, minmax=True)
        print("Scaled histograms shape:", self.X.shape)

        self.shift_array = torch.from_numpy(shift_array).float()
        self.magnet_settings = torch.from_numpy(magnet_settings).float()
        # Shape(shift_array) = (n_magnets, n_samples, n_params) = (3, n_samples, 30)

        Y_raw = self.shift_array[0, :, 1:]  # Only keep the shifting parameters (exclude magnet settings)
        # shape(Y_raw) = (num_samples, n_params - 1) = (num_samples, 29)
        self.Y = scale_Y(Y_raw, std=False, minmax=True)
        print("Shift array shape:", self.Y.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx): #returns the 4 channels and the corresponding shifts
        return self.X[idx], self.Y[idx]

    def get_magnet_settings(self):
        return self.magnet_settings
    
    def unscale(self):
        return self.X_params, self.Y_params
    
    


def naive_solution(noisy_fft):
    """
    A naive solution to find the peaks in a noisy FFT signal.
    Returns the second and third largest peaks in order of frequency appearance.
    
    Args:
        noisy_fft (torch.Tensor): The noisy FFT signal.
        Assume input of shape (batch_size, num_freq_bins, num_time_bins)

    Returns:
        torch.Tensor: The peak heights in order of frequency (left to right).
    """
    # Find indices of the maximum values in the noisy FFT signal
    ordered_peaks = []
    n_batches = noisy_fft.size(0)
    
    for i in range(n_batches):  # Iterate over each batch
        # Get the mean over time dimension
        fft_mean = noisy_fft[i].mean(dim=1)  # Average over time
        
        # Sort values in descending order and get both values and indices
        sorted_values, sorted_indices = fft_mean.clone().detach().sort(descending=True)
        
        # Get indices of the second and third highest peaks
        peak2_idx = sorted_indices[1].item()
        peak3_idx = sorted_indices[2].item()
        
        # Get the actual values at these indices
        peak2_val = fft_mean[peak2_idx]
        peak3_val = fft_mean[peak3_idx]
        
        # Sort by frequency position (index), not by magnitude
        if peak2_idx < peak3_idx:
            ordered_peaks.append(torch.tensor([peak2_val, peak3_val], device=noisy_fft.device))
        else:
            ordered_peaks.append(torch.tensor([peak3_val, peak2_val], device=noisy_fft.device))
    
    return torch.stack(ordered_peaks)

class NaiveSolutionWrapper(nn.Module):
    def __init__(self, no_middle=True):
        super().__init__()
        self.no_middle = no_middle
        
    def forward(self, x):
        # x is expected to be in shape [batch_size, 1, freq_bins, time_bins]
        # Extract just the magnitude spectrogram (first channel)
        x_mag = x.squeeze(1)  # Now [batch_size, freq_bins, time_bins]
        
        # Use the naive_solution function
        peak_heights = naive_solution(x_mag)
        
        # Reshape output to match model output
        return peak_heights