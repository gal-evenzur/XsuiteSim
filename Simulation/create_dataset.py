# %% IMPORTS
import xobjects as xo
import xtrack as xt

import h5py
from sim_functions import *
from params import *
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


from scipy import ndimage
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull

plt.rcParams['image.cmap'] = 'afmhot'

def analyze_particle_blob_enhanced(histogram, debug=False):
    """
    Enhanced thresholding approach for particle blob detection
    
    Args:
        histogram: 2D numpy array with particle counts
        debug: bool, if True shows debug plots
    
    Returns:
        dict with detection results
    """
    plot = True 
    # Step 1: Basic statistics
    max_intensity = np.max(histogram)
    if plot:
        if max_intensity < 1:  # No meaningful signal
            return {"detected": False, "reason": "no_signal"}
    
    # Step 2: Multi-level thresholding
    high_threshold = max_intensity * 0.5   # Core region
    mid_threshold = max_intensity * 0.4    # Main blob
    low_threshold = max_intensity * 0.3    # Extended region
    
    # Create masks for different threshold levels
    high_mask = histogram > high_threshold
    mid_mask = histogram > mid_threshold
    low_mask = histogram > low_threshold
    
    # Step 3: Check if we have meaningful regions
    high_pixels = np.sum(high_mask)
    mid_pixels = np.sum(mid_mask)
    low_pixels = np.sum(low_mask)
    if plot:
        if high_pixels < 3:  # Too few high-intensity pixels
            return {"detected": False, "reason": "too_blurry"}
    
    # Step 4: Calculate weighted centroids for each level
    def weighted_centroid(data, mask):
        if not np.any(mask):
            return None, None
        
        y_coords, x_coords = np.mgrid[:data.shape[0], :data.shape[1]]
        
        total_weight = np.sum(data[mask])
        if total_weight == 0:
            return None, None
            
        center_x = np.sum(data[mask] * x_coords[mask]) / total_weight
        center_y = np.sum(data[mask] * y_coords[mask]) / total_weight
        
        return center_x, center_y
    
    # Calculate centroids at different levels
    high_center_x, high_center_y = weighted_centroid(histogram, high_mask)
    mid_center_x, mid_center_y = weighted_centroid(histogram, mid_mask)
    if plot:
        if high_center_x is None:
            return {"detected": False, "reason": "no_valid_center"}
    
    # Step 5: Check if blob is well-centered (not out of frame)
    img_height, img_width = histogram.shape
    
    # Use mid-level mask for frame checking (more generous)
    mid_coords = np.column_stack(np.where(mid_mask))
    if plot:
        if len(mid_coords) == 0:
            return {"detected": False, "reason": "no_mid_level_signal"}
    
    # Check if any significant part touches edges
    min_y, min_x = np.min(mid_coords, axis=0)
    max_y, max_x = np.max(mid_coords, axis=0)
    
    edge_buffer = 3  # pixels from edge
    touches_edge = (min_x <= edge_buffer or max_x >= img_width - edge_buffer or
                   min_y <= edge_buffer or max_y >= img_height - edge_buffer)
    
    # But allow it if the center is well within frame
    center_well_inside = (high_center_x > img_width * 0.1 and 
                         high_center_x < img_width * 0.9 and
                         high_center_y > img_height * 0.1 and 
                         high_center_y < img_height * 0.9)
    
    # if touches_edge and not center_well_inside:
    #     return {"detected": False, "reason": "out_of_frame"}
    
    # Step 6: Analyze blob shape and compactness
    
    # Calculate spread at different threshold levels
    def calculate_spread(data, mask, center_x, center_y):
        if not np.any(mask):
            return 0, 0
        
        y_coords, x_coords = np.mgrid[:data.shape[0], :data.shape[1]]
        
        # Weighted standard deviations
        weights = data[mask]
        total_weight = np.sum(weights)
        
        if total_weight == 0:
            return 0, 0
        
        x_var = np.sum(weights * (x_coords[mask] - center_x)**2) / total_weight
        y_var = np.sum(weights * (y_coords[mask] - center_y)**2) / total_weight
        
        return np.sqrt(x_var), np.sqrt(y_var)
    
    high_std_x, high_std_y = calculate_spread(histogram, high_mask, high_center_x, high_center_y)
    mid_std_x, mid_std_y = calculate_spread(histogram, mid_mask, mid_center_x, mid_center_y)
    
    # Step 7: Check if too blurry
    # Blob is too blurry if:
    # 1. The spread is too large
    # 2. The ratio of high to mid threshold regions is too small

    max_reasonable_std = min(img_width, img_height) * 0.4  # 40% of smallest dimension
    too_spread_out = (mid_std_x > max_reasonable_std or mid_std_y > max_reasonable_std)
    
    # Check concentration: high-intensity region should be significant part of mid-intensity
    concentration_ratio = high_pixels / max(mid_pixels, 1)
    too_diffuse = concentration_ratio < 0.1
    if plot:
        if too_spread_out or too_diffuse:
            return {"detected": False, "reason": "too_blurry"}
    
    # Step 8: Calculate final measurements
    
    # Use high threshold center for precision, but mid threshold for size
    final_center_x = high_center_x
    final_center_y = high_center_y
    
    # Calculate effective blob dimensions using mid-level mask
    blob_coords = np.column_stack(np.where(mid_mask))
    
    if len(blob_coords) > 2:
        # Calculate oriented bounding box dimensions
        try:
            hull = ConvexHull(blob_coords)
            hull_points = blob_coords[hull.vertices]
            
            # Calculate dimensions from hull
            if len(hull_points) > 1:
                distances = pdist(hull_points)
                max_distance = np.max(distances)
                # Approximate width as a fraction of max distance
                effective_width = max_distance * 0.7
                effective_height = len(blob_coords) / max(effective_width, 1) * 2
            else:
                effective_width = mid_std_x * 4  # ~2 sigma on each side
                effective_height = mid_std_y * 4
        except:
            # Fallback to standard deviation method
            effective_width = mid_std_x * 4
            effective_height = mid_std_y * 4
    else:
        effective_width = mid_std_x * 4
        effective_height = mid_std_y * 4
    
    # Step 9: Final quality metrics
    peak_intensity = max_intensity
    total_counts = np.sum(histogram[mid_mask])
    
    result = {
        "detected": True,
        "center": (final_center_x, final_center_y),
        "center_pixel": (int(round(final_center_x)), int(round(final_center_y))),
        "width": effective_width,
        "height": effective_height,
        "std_x": mid_std_x,
        "std_y": mid_std_y,
        "peak_intensity": peak_intensity,
        "total_counts": total_counts,
        "concentration_ratio": concentration_ratio,
        "high_threshold_pixels": high_pixels,
        "mid_threshold_pixels": mid_pixels,
    }
    # Step 10: Debug visualization
    if debug:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original histogram
        im1 = axes[0,0].imshow(histogram, origin='lower', cmap='hot')
        axes[0,0].plot(final_center_x, final_center_y, 'b+', markersize=15, markeredgewidth=3)
        axes[0,0].set_title('Original + Center')
        plt.colorbar(im1, ax=axes[0,0])
        
        # High threshold
        axes[0,1].imshow(high_mask, origin='lower', cmap='gray')
        axes[0,1].plot(high_center_x, high_center_y, 'r+', markersize=10)
        axes[0,1].set_title(f'High Threshold (>{high_threshold:.1f})')
        
        # Mid threshold
        axes[0,2].imshow(mid_mask, origin='lower', cmap='gray')
        axes[0,2].plot(mid_center_x, mid_center_y, 'g+', markersize=10)
        axes[0,2].set_title(f'Mid Threshold (>{mid_threshold:.1f})')
        
        # Low threshold
        axes[1,0].imshow(low_mask, origin='lower', cmap='gray')
        axes[1,0].set_title(f'Low Threshold (>{low_threshold:.1f})')
        
        # Original with ellipse overlay showing standard deviations
        axes[1,1].imshow(histogram, origin='lower', cmap='hot')
        axes[1,1].plot(final_center_x, final_center_y, 'b+', markersize=15, markeredgewidth=3)
        
        # Draw ellipses showing 1σ, 2σ, and 3σ contours
        from matplotlib.patches import Ellipse
        
        # 1 sigma ellipse (68% of data)
        ellipse_1sig = Ellipse((final_center_x, final_center_y), 
                              width=2*mid_std_x, height=2*mid_std_y,
                              fill=False, color='cyan', linewidth=2, linestyle='-')
        axes[1,1].add_patch(ellipse_1sig)
        
        # 2 sigma ellipse (95% of data)  
        ellipse_2sig = Ellipse((final_center_x, final_center_y),
                              width=4*mid_std_x, height=4*mid_std_y,
                              fill=False, color='yellow', linewidth=2, linestyle='--')
        axes[1,1].add_patch(ellipse_2sig)
        
        # 3 sigma ellipse (99.7% of data)
        ellipse_3sig = Ellipse((final_center_x, final_center_y),
                              width=6*mid_std_x, height=6*mid_std_y,
                              fill=False, color='red', linewidth=1, linestyle=':')
        axes[1,1].add_patch(ellipse_3sig)
        
        axes[1,1].set_title('Original + Std Ellipses\nCyan=1σ, Yellow=2σ, Red=3σ')
        
        # Cross-sections through the center
        center_x_int = int(round(final_center_x))
        center_y_int = int(round(final_center_y))
        
        # Vertical cross-section (along y-axis)
        if 0 <= center_x_int < histogram.shape[1]:
            y_profile = histogram[:, center_x_int]
            y_coords = np.arange(len(y_profile))
            
            axes[1,2].plot(y_profile, y_coords, 'b-', linewidth=2, label='Intensity')
            axes[1,2].axhline(final_center_y, color='red', linestyle='--', label='Center')
            axes[1,2].axhline(final_center_y - mid_std_y, color='cyan', linestyle=':', alpha=0.7, label='±1σ')
            axes[1,2].axhline(final_center_y + mid_std_y, color='cyan', linestyle=':', alpha=0.7)
            axes[1,2].axhline(final_center_y - 2*mid_std_y, color='yellow', linestyle=':', alpha=0.5, label='±2σ')
            axes[1,2].axhline(final_center_y + 2*mid_std_y, color='yellow', linestyle=':', alpha=0.5)
            axes[1,2].set_xlabel('Intensity')
            axes[1,2].set_ylabel('Y Position')
            axes[1,2].set_title(f'Y-Profile at X={center_x_int}\nσ_y = {mid_std_y:.1f}')
            axes[1,2].legend()
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot showing horizontal cross-section
        fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Horizontal cross-section (along x-axis)
        if 0 <= center_y_int < histogram.shape[0]:
            x_profile = histogram[center_y_int, :]
            x_coords = np.arange(len(x_profile))
            
            ax.plot(x_coords, x_profile, 'b-', linewidth=2, label='Intensity')
            ax.axvline(final_center_x, color='red', linestyle='--', label='Center')
            ax.axvline(final_center_x - mid_std_x, color='cyan', linestyle=':', alpha=0.7, label='±1σ')
            ax.axvline(final_center_x + mid_std_x, color='cyan', linestyle=':', alpha=0.7)
            ax.axvline(final_center_x - 2*mid_std_x, color='yellow', linestyle=':', alpha=0.5, label='±2σ')
            ax.axvline(final_center_x + 2*mid_std_x, color='yellow', linestyle=':', alpha=0.5)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Intensity')
            ax.set_title(f'X-Profile at Y={center_y_int}, σ_x = {mid_std_x:.1f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
        # Print results
        print(f"Detection: {result['detected']}")
        if result['detected']:
            print(f"Center: ({result['center'][0]:.1f}, {result['center'][1]:.1f})")
            print(f"Size: {result['width']:.1f} x {result['height']:.1f}")
            print(f"Peak intensity: {result['peak_intensity']:.1f}")
            print(f"Concentration ratio: {result['concentration_ratio']:.3f}")
    
    return result    



# Example usage and testing function
def test_algorithm():
    """
    Test the algorithm with synthetic data
    """
    # shifts[name][setting] = change.min()
    line, env, ref = line_init(shifts=shifts)

    particles = import_particles_from_hdf5(dat_file, ref)

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    h, xedges, yedges = track_monitor(line, particles)
    plt.imshow(h.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
    ax.locator_params(axis='x', nbins=10)
    ax.locator_params(axis='y', nbins=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.grid(True,linewidth=0.25,alpha=0.25,which='major')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.title('Monitor at the end of the line')
    plt.colorbar(label='Counts per bin')
    plt.tight_layout()


    print("=== Testing Good Histogram ===")
    result_good = analyze_particle_blob_enhanced(h.T, debug=True)
    
    # # Bad case: too spread out
    # bad_histogram = 4 * np.exp(-((X)**2 + (Y)**2) / 8)  # Much wider
    # bad_histogram += np.random.poisson(0.1, bad_histogram.shape)
    
    # print("\n=== Testing Bad Histogram (too blurry) ===")
    # result_bad = analyze_particle_blob_enhanced(bad_histogram, debug=True)
    
    return result_good


results = test_algorithm()
print(results)
plt.show()