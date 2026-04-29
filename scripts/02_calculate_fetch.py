#!/usr/bin/env python3
"""
Calculate Wind Fetch for Lake

For each water cell, calculates the fetch distance (unobstructed distance over water)
in 16 compass directions.

Output: Fetch rasters for each direction, stored as GeoTIFFs.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import proj_fix  # noqa: E402,F401 — must run before geo imports

import numpy as np
import rasterio
from scipy import ndimage

from lib.paths import LakePaths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 36 directions at 10-degree spacing for effective fetch accuracy
DIRECTIONS = [float(d) for d in range(0, 360, 10)]


def calculate_fetch_single_direction(water_mask: np.ndarray, direction_deg: float, 
                                     cell_size: float) -> np.ndarray:
    """
    Calculate fetch distance for a single wind direction.
    
    For each water cell, traces a ray in the upwind direction until hitting land.
    
    Args:
        water_mask: Binary array (1 = water, 0 = land)
        direction_deg: Wind direction in degrees (direction wind is coming FROM)
        cell_size: Cell size in meters
        
    Returns:
        Array of fetch distances in meters
    """
    # Convert direction to radians
    # Direction is where wind comes FROM, so we trace in opposite direction
    # to find how far wind has traveled over water
    direction_rad = np.radians(direction_deg)
    
    # Calculate unit vector components
    # In array coordinates: row increases downward (south), col increases rightward (east)
    # Wind from north (0°): traces southward (positive row direction)
    dy = np.cos(direction_rad)  # Row direction (positive = south)
    dx = np.sin(direction_rad)  # Col direction (positive = east)
    
    height, width = water_mask.shape
    fetch = np.zeros_like(water_mask, dtype=np.float32)
    
    # For efficiency, we'll use a ray-marching approach
    # For each water cell, march along the wind direction until hitting land
    
    # Get water cell coordinates
    water_rows, water_cols = np.where(water_mask == 1)
    
    # Maximum fetch distance (diagonal of raster)
    max_steps = int(np.sqrt(height**2 + width**2)) + 1
    
    for i in range(len(water_rows)):
        row, col = water_rows[i], water_cols[i]
        
        # March along wind direction
        distance = 0.0
        step = 1
        
        while step < max_steps:
            # Calculate new position
            new_row = row + dy * step
            new_col = col + dx * step
            
            # Check bounds
            new_row_int = int(round(new_row))
            new_col_int = int(round(new_col))
            
            if new_row_int < 0 or new_row_int >= height or \
               new_col_int < 0 or new_col_int >= width:
                # Hit boundary - use distance to boundary
                distance = step * cell_size
                break
            
            # Check if still over water
            if water_mask[new_row_int, new_col_int] == 0:
                # Hit land
                distance = step * cell_size
                break
            
            step += 1
        else:
            # Reached max steps (shouldn't happen for reasonable lakes)
            distance = max_steps * cell_size
        
        fetch[row, col] = distance
    
    return fetch


def calculate_fetch_vectorized(water_mask: np.ndarray, direction_deg: float,
                                cell_size: float, max_distance: float = 50000.0) -> np.ndarray:
    """
    Vectorized fetch calculation using cumulative sum along direction.
    
    Much faster than cell-by-cell ray marching.
    """
    # Convert direction to radians
    direction_rad = np.radians(direction_deg)
    
    # For cardinal and 45° directions, we can use efficient array operations
    # For other angles, we need to rotate the array
    
    height, width = water_mask.shape
    
    # Calculate step direction
    # Normalize to get primary axis
    dy = np.cos(direction_rad)
    dx = np.sin(direction_rad)
    
    # Determine dominant direction
    if abs(dy) >= abs(dx):
        # Primarily vertical movement
        if dy >= 0:  # Wind from north, trace south
            # Calculate cumulative water distance from top
            fetch = np.zeros_like(water_mask, dtype=np.float32)
            cumsum = np.zeros(width, dtype=np.float32)
            
            for row in range(height):
                # Add current row's water
                is_water = water_mask[row, :] == 1
                cumsum = np.where(is_water, cumsum + cell_size, 0.0)
                fetch[row, :] = cumsum * is_water
        else:  # Wind from south, trace north
            fetch = np.zeros_like(water_mask, dtype=np.float32)
            cumsum = np.zeros(width, dtype=np.float32)
            
            for row in range(height - 1, -1, -1):
                is_water = water_mask[row, :] == 1
                cumsum = np.where(is_water, cumsum + cell_size, 0.0)
                fetch[row, :] = cumsum * is_water
    else:
        # Primarily horizontal movement
        if dx >= 0:  # Wind from west, trace east
            fetch = np.zeros_like(water_mask, dtype=np.float32)
            cumsum = np.zeros(height, dtype=np.float32)
            
            for col in range(width):
                is_water = water_mask[:, col] == 1
                cumsum = np.where(is_water, cumsum + cell_size, 0.0)
                fetch[:, col] = cumsum * is_water
        else:  # Wind from east, trace west
            fetch = np.zeros_like(water_mask, dtype=np.float32)
            cumsum = np.zeros(height, dtype=np.float32)
            
            for col in range(width - 1, -1, -1):
                is_water = water_mask[:, col] == 1
                cumsum = np.where(is_water, cumsum + cell_size, 0.0)
                fetch[:, col] = cumsum * is_water
    
    # For diagonal directions, rotate, calculate, rotate back
    # This is a simplification - for full accuracy, use proper ray tracing
    
    # Clip to max distance
    fetch = np.clip(fetch, 0, max_distance)
    
    return fetch


def calculate_fetch_rotated(water_mask: np.ndarray, direction_deg: float,
                            cell_size: float, max_distance: float = 50000.0) -> np.ndarray:
    """
    Calculate fetch by rotating array, computing cumsum, and rotating back.
    
    Works for any angle.
    """
    from scipy.ndimage import rotate
    
    # Rotate array so that wind direction aligns with rows (top to bottom)
    # scipy rotate uses counter-clockwise, and we want wind coming from top
    rotation_angle = -direction_deg  # Negative for clockwise rotation
    
    # Pad to avoid edge effects
    pad_size = max(water_mask.shape) // 2
    padded = np.pad(water_mask, pad_size, mode='constant', constant_values=0)
    
    # Rotate (use order=0 for nearest neighbor to preserve binary values)
    rotated = rotate(padded.astype(float), rotation_angle, reshape=True, order=0)
    rotated = (rotated > 0.5).astype(np.uint8)  # Re-binarize
    
    # Calculate cumulative sum from top (wind direction)
    height, width = rotated.shape
    fetch_rotated = np.zeros_like(rotated, dtype=np.float32)
    cumsum = np.zeros(width, dtype=np.float32)
    
    for row in range(height):
        is_water = rotated[row, :] == 1
        cumsum = np.where(is_water, cumsum + cell_size, 0.0)
        fetch_rotated[row, :] = cumsum * is_water
    
    # Rotate back
    fetch_back = rotate(fetch_rotated, -rotation_angle, reshape=True, order=1)
    
    # Extract original region (accounting for any size changes)
    h_orig, w_orig = water_mask.shape
    h_back, w_back = fetch_back.shape
    
    # Center crop
    start_row = (h_back - h_orig - 2*pad_size) // 2 + pad_size
    start_col = (w_back - w_orig - 2*pad_size) // 2 + pad_size
    
    # Handle edge cases with bounds checking
    start_row = max(0, min(start_row, h_back - h_orig))
    start_col = max(0, min(start_col, w_back - w_orig))
    
    fetch = fetch_back[start_row:start_row+h_orig, start_col:start_col+w_orig]
    
    # Ensure we got the right shape
    if fetch.shape != water_mask.shape:
        # Resize if needed
        from scipy.ndimage import zoom
        zoom_factor = (h_orig / fetch.shape[0], w_orig / fetch.shape[1])
        fetch = zoom(fetch, zoom_factor, order=1)
        fetch = fetch[:h_orig, :w_orig]
    
    # Mask to only water cells
    fetch = fetch * water_mask
    
    # Clip to max distance
    fetch = np.clip(fetch, 0, max_distance)
    
    return fetch


def calculate_all_fetch_directions(raster_path: Path, output_dir: Path,
                                   directions: list = None) -> dict:
    """
    Calculate fetch for all compass directions and save as rasters.
    
    Returns dict mapping direction to output path.
    """
    if directions is None:
        directions = DIRECTIONS
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read lake raster
    with rasterio.open(raster_path) as src:
        water_mask = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        cell_size = abs(transform.a)  # Assuming square cells
    
    logger.info(f"Loaded raster: {water_mask.shape}, cell size: {cell_size}m")
    logger.info(f"Water cells: {np.sum(water_mask == 1):,}")
    
    # Update profile for float output
    profile.update(dtype='float32', compress='lzw')
    
    output_paths = {}
    
    for direction in directions:
        logger.info(f"Calculating fetch for {direction}°...")
        
        # Use the rotated method for any angle
        fetch = calculate_fetch_rotated(water_mask, direction, cell_size)
        
        # Save raster
        output_path = output_dir / f"fetch_{int(direction):03d}.tif"
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(fetch, 1)
            dst.update_tags(direction=direction, units='meters')
        
        output_paths[direction] = output_path
        
        # Log statistics
        water_fetch = fetch[water_mask == 1]
        if len(water_fetch) > 0:
            logger.info(f"  Mean fetch: {np.mean(water_fetch):.0f}m, "
                       f"Max: {np.max(water_fetch):.0f}m")
    
    # Save index file
    index_path = output_dir / "fetch_index.json"
    with open(index_path, 'w') as f:
        json.dump({
            'directions': directions,
            'files': {d: str(p.name) for d, p in output_paths.items()},
            'cell_size_m': cell_size,
            'crs': str(profile['crs'])
        }, f, indent=2)
    
    logger.info(f"Saved fetch index to {index_path}")
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(description='Calculate wind fetch for lake')
    parser.add_argument('--lake', type=str, default='champlain',
                        help='Lake name')
    parser.add_argument('--input-dir', type=Path, default=Path('data/lakes'),
                        help='Input directory with lake raster')
    parser.add_argument('--output-dir', type=Path, default=Path('data/fetch_rasters'),
                        help='Output directory for fetch rasters')
    parser.add_argument('--directions', type=str, default=None,
                        help='Comma-separated list of directions (default: 16 compass points)')
    
    args = parser.parse_args()

    # Parse directions
    if args.directions:
        directions = [float(d) for d in args.directions.split(',')]
    else:
        directions = DIRECTIONS

    # Auto-detect paths from project root
    paths = LakePaths(args.lake)
    raster_path = paths.raster
    if not raster_path.exists():
        # Fallback to legacy path
        raster_path = args.input_dir / f"{args.lake}_raster.tif"

    if not raster_path.exists():
        logger.error(f"Lake raster not found: {raster_path}")
        logger.error("Run 01_prepare_lake.py first")
        return

    output_dir = paths.fetch_dir

    # Calculate fetch
    output_paths = calculate_all_fetch_directions(raster_path, output_dir, directions)

    logger.info(f"Fetch calculation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Calculated {len(output_paths)} direction rasters")


if __name__ == '__main__':
    main()
