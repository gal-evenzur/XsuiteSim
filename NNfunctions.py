import torch
import torch.nn as nn
from ignite.metrics.metric import reinit__is_reduced
from torch.utils.data import Dataset

from ignite.metrics import Metric
import json
import h5py
import numpy as np

def scale_tensor(dat_raw, std=False, minmax=True):
    # print(f"data is {dat_raw.shape} - {min.shape} / ({max.shape} - {min.shape} ")
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
        for i in range(dat_raw.shape[0]):
            batch = dat_raw[i]
            min_val = lib.min(batch)
            max_val = lib.max(batch)
            if max_val > min_val:  # Avoid division by zero
                dat_raw[i] = (batch - min_val) / (max_val - min_val)
            else:
                dat_raw[i] = lib.zeros_like(batch)
            
    return dat_raw


class SignalDataset(Dataset):
    def __init__(self, data_path, split="train", 
                 transfrom=scale_tensor):
        # Determine file type based on extension
        # Load HDF5 data
        with h5py.File(data_path, 'r') as f:
            xedges = f['xedges'][:]
            yedges = f['yedges'][:]

            shift_array = f[f'{split}/shifts_list'][:]
            magnet_settings = f[f'{split}/magnet_settings'][:]
            histograms = f[f'{split}/histograms'][:]
            
            self.xedges = torch.from_numpy(xedges)
            self.yedges = torch.from_numpy(yedges)

            self.shift_array = torch.from_numpy(shift_array)
            self.magnet_settings = torch.from_numpy(magnet_settings)
            self.histograms = torch.from_numpy(histograms)

        self.X = transfrom(self.histograms, std=False, minmax=True)
        # shape(X) = n_magnet_settings [=n_channels] x n_samples x 128 x 256  
        self.Y = self.shift_array[0, :, 1:]  # Only keep the parameters (exclude magnet settings)
        # shape(Y) = (num_samples, num_shifts) = (num_samples, 29)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx): #returns the 4 channels and the corresponding shifts
        return self.X[:, idx], self.Y[idx]

    def get_magnet_settings(self):
        return self.magnet_settings
    
    


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