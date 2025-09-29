import h5py
import numpy as np
import os
import glob
from typing import List, Optional

def get_hdf5_files(directory_path: str, recursive: bool = False, extensions: List[str] = None) -> List[str]:
    """
    Get all HDF5 files in a directory.
    
    Args:
        directory_path: Path to the directory to search
        recursive: If True, search subdirectories recursively
        extensions: List of file extensions to look for (default: ['h5', 'hdf5', 'hdf'])
    
    Returns:
        List of full paths to HDF5 files, sorted alphabetically
    """
    
    if extensions is None:
        extensions = ['h5', 'hdf5', 'hdf']
    
    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_path} does not exist")
        return []
    
    if not os.path.isdir(directory_path):
        print(f"Warning: {directory_path} is not a directory")
        return []
    
    hdf5_files = []
    
    # Build search patterns for each extension
    for ext in extensions:
        if recursive:
            # Search recursively using **/ pattern
            pattern = os.path.join(directory_path, '**', f'*.{ext}')
            hdf5_files.extend(glob.glob(pattern, recursive=True))
        else:
            # Search only in the specified directory
            pattern = os.path.join(directory_path, f'*.{ext}')
            hdf5_files.extend(glob.glob(pattern))
    
    # Remove duplicates and sort
    hdf5_files = sorted(list(set(hdf5_files)))
    
    print(f"Found {len(hdf5_files)} HDF5 files in {directory_path}")
    if recursive:
        print("  (including subdirectories)")
    
    return hdf5_files

def merge_hdf5_files(input_files: List[str], output_file: str, verbose: bool = True, 
                    memory_efficient: bool = False):
    """
    Merge multiple HDF5 files with train/validation/test datasets into a single file.
    
    Args:
        input_files: List of paths to input HDF5 files
        output_file: Path for the merged output file
        verbose: Whether to print progress information
        memory_efficient: If True, use streaming approach for large files
    """
    
    if verbose:
        print(f"Merging {len(input_files)} HDF5 files into {output_file}")
    
    if memory_efficient:
        return _merge_hdf5_memory_efficient(input_files, output_file, verbose)
    
    # Initialize containers for merged data
    merged_data = {
        'train': {'histograms': [], 'shifts_list': [], 'magnet_settings': []},
        'validation': {'histograms': [], 'shifts_list': [], 'magnet_settings': []},
        'test': {'histograms': [], 'shifts_list': [], 'magnet_settings': []}
    }
    
    xedges = None
    yedges = None
    
    # Read and collect data from all input files
    for i, file_path in enumerate(input_files):
        if verbose:
            print(f"Processing file {i+1}/{len(input_files)}: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping...")
            continue
            
        try:
            with h5py.File(file_path, 'r') as f:
                # Get edges from first file (assuming they're the same across all files)
                if xedges is None:
                    xedges = f['xedges'][:]
                    yedges = f['yedges'][:]
                
                # Process each dataset (train, validation, test)
                for dataset_name in ['train', 'validation', 'test']:
                    if dataset_name in f:
                        dataset = f[dataset_name]
                        
                        # Collect data from each dataset
                        if 'histograms' in dataset:
                            merged_data[dataset_name]['histograms'].append(dataset['histograms'][:])
                        if 'shifts_list' in dataset:
                            merged_data[dataset_name]['shifts_list'].append(dataset['shifts_list'][:])
                        if 'magnet_settings' in dataset:
                            merged_data[dataset_name]['magnet_settings'] = dataset['magnet_settings'][:]
                    else:
                        if verbose:
                            print(f"Warning: Dataset '{dataset_name}' not found in {file_path}")
                            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Save merged data to output file
    if verbose:
        print(f"Saving merged data to {output_file}")
    
    with h5py.File(output_file, 'w') as f:
        # Save edges
        f.create_dataset('xedges', data=xedges)
        f.create_dataset('yedges', data=yedges)
        
        # Save merged datasets
        for dataset_name in ['train', 'validation', 'test']:
            if any(len(merged_data[dataset_name][key]) > 0 for key in merged_data[dataset_name]):
                group = f.create_group(dataset_name)
                
                hist_list = merged_data[dataset_name]['histograms']
                shift_list = merged_data[dataset_name]['shifts_list']
                magnet_settings = merged_data[dataset_name]['magnet_settings']

                group.create_dataset('magnet_settings', data=magnet_settings)
                group.create_dataset('histograms', data=np.concatenate(hist_list, axis=1) if hist_list else np.array([]))
                group.create_dataset('shifts_list', data=np.concatenate(shift_list, axis=1) if shift_list else np.array([]))
                # Concatenate and save each data type
                        
                if verbose:
                    print(f"  {dataset_name}/histograms: {np.concatenate(hist_list, axis=1).shape}")
                    print(f"  {dataset_name}/shifts_list: {np.concatenate(shift_list, axis=1).shape}")
                    print(f"  {dataset_name}/magnet_settings: {magnet_settings.shape}")

    if verbose:
        print("Merge completed successfully!")


def _merge_hdf5_memory_efficient(input_files: List[str], output_file: str, verbose: bool = True):
    """Memory-efficient version that streams data directly without loading everything into RAM."""
    
    # First pass: get dimensions and edges
    total_samples = {'train': 0, 'validation': 0, 'test': 0}
    xedges = yedges = None
    sample_shapes = {}
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            continue
            
        with h5py.File(file_path, 'r') as f:
            if xedges is None:
                xedges = f['xedges'][:]
                yedges = f['yedges'][:]
            
            for dataset_name in ['train', 'validation', 'test']:
                if dataset_name in f and 'histograms' in f[dataset_name]:
                    shape = f[dataset_name]['histograms'].shape
                    total_samples[dataset_name] += shape[0]
                    if dataset_name not in sample_shapes:
                        sample_shapes[dataset_name] = shape[1:]
    
    # Create output file with pre-allocated arrays
    with h5py.File(output_file, 'w') as out_f:
        out_f.create_dataset('xedges', data=xedges)
        out_f.create_dataset('yedges', data=yedges)
        
        # Pre-allocate datasets
        datasets = {}
        for dataset_name in ['train', 'validation', 'test']:
            if total_samples[dataset_name] > 0:
                group = out_f.create_group(dataset_name)
                datasets[dataset_name] = {}
                
                # Create datasets with known total size
                hist_shape = (total_samples[dataset_name],) + sample_shapes[dataset_name]
                datasets[dataset_name]['histograms'] = group.create_dataset('histograms', hist_shape)
                
                # We'll determine other shapes dynamically
        
        # Second pass: stream data directly to output
        current_idx = {'train': 0, 'validation': 0, 'test': 0}
        
        for i, file_path in enumerate(input_files):
            if verbose:
                print(f"Streaming file {i+1}/{len(input_files)}: {file_path}")
                
            if not os.path.exists(file_path):
                continue
                
            with h5py.File(file_path, 'r') as in_f:
                for dataset_name in ['train', 'validation', 'test']:
                    if dataset_name in in_f and dataset_name in datasets:
                        in_dataset = in_f[dataset_name]
                        out_dataset = datasets[dataset_name]
                        
                        if 'histograms' in in_dataset:
                            data = in_dataset['histograms'][:]
                            start_idx = current_idx[dataset_name]
                            end_idx = start_idx + data.shape[0]
                            
                            out_dataset['histograms'][start_idx:end_idx] = data
                            
                            # Handle other datasets similarly if they exist
                            for data_type in ['shifts_list', 'magnet_settings']:
                                if data_type in in_dataset:
                                    if data_type not in out_dataset:
                                        # Create dataset on first encounter
                                        sample_data = in_dataset[data_type][:]
                                        full_shape = (total_samples[dataset_name],) + sample_data.shape[1:]
                                        out_dataset[data_type] = datasets[dataset_name].parent.create_dataset(
                                            data_type, full_shape, dtype=sample_data.dtype)
                                    
                                    data = in_dataset[data_type][:]
                                    out_dataset[data_type][start_idx:end_idx] = data
                            
                            current_idx[dataset_name] = end_idx
    
    if verbose:
        print("Memory-efficient merge completed successfully!")

def get_dataset_info(file_path: str):
    """
    Print information about the structure and contents of an HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
    """
    print(f"\nFile: {file_path}")
    print("-" * 50)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Print global datasets
            print("Global datasets:")
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    print(f"  {key}: {f[key].shape}")
            
            # Print group information
            print("\nGroups:")
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    print(f"  {key}:")
                    group = f[key]
                    for subkey in group.keys():
                        if isinstance(group[subkey], h5py.Dataset):
                            print(f"    {subkey}: {group[subkey].shape}")
                            
    except Exception as e:
        print(f"Error reading file: {str(e)}")

# Get all HDF5 files from a directory
pydir = os.path.dirname(os.path.abspath(__file__)) # This results "~/fresh-start/Simulation"
maindir = os.path.dirname(pydir)  # This results "~/fresh-start"
input_directory = os.path.join(maindir, "Data")
input_files = get_hdf5_files(input_directory)

# Output merged file
output_file = os.path.join(maindir, "merged_data", "merged_data.h5")

# Optional: Check structure of input files before merging
print("Input file structures:")
for file_path in input_files:
    if os.path.exists(file_path):
        get_dataset_info(file_path)

# Merge the files
merge_hdf5_files(input_files, output_file, verbose=True)

# Check the merged file structure
print("\nMerged file structure:")
get_dataset_info(output_file)