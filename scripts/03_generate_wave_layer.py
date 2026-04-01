#!/usr/bin/env python3
"""
Generate Wave Impact Layer

Combines pre-calculated fetch with current wind conditions to produce:
1. Wave intensity grid (for water surface visualization)
2. Bank impact segments (shoreline exposure)
3. Calm zone markers

Output: GeoJSON files ready for Mapbox styling
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import proj_fix  # noqa: E402,F401 — must run before geo imports

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union

from lib.lake_config import load_lake_config
from lib.paths import LakePaths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Wave intensity thresholds (meters)
WAVE_THRESHOLDS = {
    'calm': 0.05,       # < 5cm
    'light': 0.15,      # 5-15cm
    'moderate': 0.30,   # 15-30cm
    'rough': 0.50,      # 30-50cm
    'very_rough': 1.0   # > 50cm
}

# Default depth for constant depth strategy (meters)
DEFAULT_DEPTH = 20.0


def calculate_wave_height_smb(wind_speed_ms: float, fetch_m: float, 
                               depth_m: float = DEFAULT_DEPTH) -> float:
    """
    Calculate significant wave height using SMB (Shore Protection Manual) method.
    
    For deep water (depth >> wave height):
        Hs = 0.283 * (U²/g) * tanh(0.0125 * (gF/U²)^0.42)
    
    Simplified for fetch-limited conditions:
        Hs ≈ 0.0016 * U² * √(F/1000)
        
    Args:
        wind_speed_ms: Wind speed in m/s
        fetch_m: Fetch distance in meters
        depth_m: Water depth in meters
        
    Returns:
        Significant wave height in meters
    """
    if fetch_m <= 0 or wind_speed_ms <= 0:
        return 0.0
    
    g = 9.81  # m/s²
    
    # Use simplified deep-water formula for lakes
    # Hs = 0.0016 * U² * √(F_km)
    fetch_km = fetch_m / 1000.0
    hs = 0.0016 * (wind_speed_ms ** 2) * np.sqrt(fetch_km)
    
    # Depth limitation (waves break when H > 0.78 * d)
    max_wave = 0.78 * depth_m
    hs = min(hs, max_wave)
    
    return hs


def classify_wave_intensity(wave_height: float) -> str:
    """Classify wave height into intensity category."""
    if wave_height < WAVE_THRESHOLDS['calm']:
        return 'calm'
    elif wave_height < WAVE_THRESHOLDS['light']:
        return 'light'
    elif wave_height < WAVE_THRESHOLDS['moderate']:
        return 'moderate'
    elif wave_height < WAVE_THRESHOLDS['rough']:
        return 'rough'
    else:
        return 'very_rough'


def interpolate_fetch(fetch_rasters: dict, wind_direction: float) -> np.ndarray:
    """
    Interpolate fetch for arbitrary wind direction from pre-calculated rasters.
    
    Args:
        fetch_rasters: Dict mapping direction to rasterio dataset
        wind_direction: Wind direction in degrees (direction wind comes FROM)
        
    Returns:
        Interpolated fetch array
    """
    directions = sorted(fetch_rasters.keys())
    
    # Find bracketing directions
    lower_dir = max([d for d in directions if d <= wind_direction], default=directions[-1])
    upper_dir = min([d for d in directions if d > wind_direction], default=directions[0])
    
    # Handle wrap-around
    if upper_dir < lower_dir:
        upper_dir += 360
    
    # Calculate interpolation weight
    if upper_dir == lower_dir or (upper_dir - lower_dir) > 180:
        # Exact match or wrap-around
        return fetch_rasters[lower_dir % 360].read(1)
    
    weight = (wind_direction - lower_dir) / (upper_dir - lower_dir)
    
    # Linear interpolation
    fetch_lower = fetch_rasters[lower_dir % 360].read(1)
    fetch_upper = fetch_rasters[upper_dir % 360].read(1)
    
    return fetch_lower * (1 - weight) + fetch_upper * weight


def generate_wave_grid(fetch_dir: Path, wind_speed_ms: float, wind_direction: float,
                       lake_polygon: gpd.GeoDataFrame, output_path: Path,
                       grid_spacing: float = 500.0):
    """
    Generate wave intensity grid as points for Mapbox visualization.
    
    Args:
        fetch_dir: Directory with fetch rasters
        wind_speed_ms: Wind speed in m/s
        wind_direction: Wind direction in degrees
        lake_polygon: Lake boundary GeoDataFrame
        output_path: Output GeoJSON path
        grid_spacing: Spacing between points in meters
    """
    logger.info("Generating wave intensity grid...")
    
    # Load fetch index
    index_path = fetch_dir / "fetch_index.json"
    with open(index_path) as f:
        fetch_index = json.load(f)
    
    # Open all fetch rasters
    fetch_rasters = {}
    for dir_str, filename in fetch_index['files'].items():
        direction = float(dir_str)
        raster_path = fetch_dir / filename
        fetch_rasters[direction] = rasterio.open(raster_path)
    
    # Get reference raster for dimensions
    ref_raster = list(fetch_rasters.values())[0]
    transform = ref_raster.transform
    crs = ref_raster.crs
    height, width = ref_raster.shape
    cell_size = abs(transform.a)
    
    # Interpolate fetch for current wind direction
    fetch = interpolate_fetch(fetch_rasters, wind_direction)
    
    # Calculate wave height for each cell
    wave_height = np.vectorize(calculate_wave_height_smb)(wind_speed_ms, fetch)
    
    # Generate points at grid spacing
    step = max(1, int(grid_spacing / cell_size))
    
    points = []
    for row in range(0, height, step):
        for col in range(0, width, step):
            if fetch[row, col] > 0:  # Only water cells
                # Convert to coordinates
                x, y = rasterio.transform.xy(transform, row, col)
                
                wh = wave_height[row, col]
                intensity = classify_wave_intensity(wh)
                
                points.append({
                    'geometry': Point(x, y),
                    'wave_height_m': float(wh),
                    'intensity': intensity,
                    'fetch_m': float(fetch[row, col]),
                    'wind_speed_ms': wind_speed_ms,
                    'wind_direction': wind_direction
                })
    
    # Create GeoDataFrame
    if not points:
        logger.warning("No wave grid points generated — fetch rasters may be empty")
        gdf = gpd.GeoDataFrame(columns=['geometry', 'wave_height_m', 'intensity',
                                         'fetch_m', 'wind_speed_ms', 'wind_direction'],
                                geometry='geometry', crs='EPSG:4326')
    else:
        gdf = gpd.GeoDataFrame(points, crs=crs)
        # Reproject to WGS84 for Mapbox
        gdf = gdf.to_crs('EPSG:4326')
    
    # Save to GeoJSON
    if output_path is not None:
        gdf.to_file(output_path, driver='GeoJSON')
        logger.info(f"Saved {len(gdf)} wave grid points to {output_path}")
    
    # Log statistics
    for intensity in ['calm', 'light', 'moderate', 'rough', 'very_rough']:
        count = len(gdf[gdf['intensity'] == intensity])
        logger.info(f"  {intensity}: {count} points")
    
    # Close rasters
    for r in fetch_rasters.values():
        r.close()
    
    return gdf


def generate_bank_impact(lake_polygon_path: Path, wind_speed_ms: float,
                         wind_direction: float, fetch_dir: Path,
                         output_path: Path, segment_length: float = 200.0,
                         utm_crs: str = None):
    """
    Generate shoreline impact segments.
    
    Args:
        lake_polygon_path: Path to lake polygon GeoJSON
        wind_speed_ms: Wind speed in m/s
        wind_direction: Wind direction in degrees
        fetch_dir: Directory with fetch rasters
        output_path: Output GeoJSON path
        segment_length: Target length of shoreline segments in meters
    """
    logger.info("Generating bank impact segments...")
    
    # Load lake polygon
    lake = gpd.read_file(lake_polygon_path)

    # Determine UTM CRS: use provided, or read from fetch rasters
    if utm_crs is None:
        index_path = fetch_dir / "fetch_index.json"
        if index_path.exists():
            with open(index_path) as f:
                fetch_index = json.load(f)
            utm_crs = fetch_index.get('crs', 'EPSG:32618')
        else:
            utm_crs = 'EPSG:32618'

    # Reproject to UTM for accurate geometry operations
    lake_utm = lake.to_crs(utm_crs)
    
    # Get the exterior boundary
    boundary = lake_utm.geometry.iloc[0]
    if isinstance(boundary, (Polygon, MultiPolygon)):
        if isinstance(boundary, MultiPolygon):
            # Get all exterior rings
            rings = [p.exterior for p in boundary.geoms]
            # Also get interior rings (islands)
            for p in boundary.geoms:
                rings.extend(list(p.interiors))
        else:
            rings = [boundary.exterior] + list(boundary.interiors)
    else:
        rings = [boundary]
    
    # Segment the shoreline
    segments = []
    
    for ring in rings:
        coords = list(ring.coords)
        
        # Calculate cumulative distance
        total_length = ring.length
        n_segments = max(1, int(total_length / segment_length))
        
        for i in range(n_segments):
            start_dist = i * total_length / n_segments
            end_dist = (i + 1) * total_length / n_segments
            
            # Get segment midpoint
            mid_dist = (start_dist + end_dist) / 2
            mid_point = ring.interpolate(mid_dist)
            
            # Get segment endpoints
            start_point = ring.interpolate(start_dist)
            end_point = ring.interpolate(end_dist)
            
            # Create segment line
            segment_line = LineString([start_point, end_point])
            
            # Calculate shoreline normal (perpendicular direction)
            # This points outward from land into water
            dx = end_point.x - start_point.x
            dy = end_point.y - start_point.y
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Normal vector pointing INTO the lake (right of traversal direction).
                # Shapely exterior rings are CCW, so the right-hand normal points inward.
                normal_x = dy / length
                normal_y = -dx / length
                
                # Normal direction in degrees
                normal_deg = np.degrees(np.arctan2(normal_x, normal_y)) % 360
                
                # Calculate angle between wind direction and shore normal
                # Wind blowing INTO shore = high impact
                angle_diff = abs(wind_direction - normal_deg)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Impact factor: cos(angle) where 0° = wind directly into shore
                impact_factor = np.cos(np.radians(angle_diff))
                
                # Only consider positive impact (wind blowing toward shore)
                if impact_factor > 0:
                    # Estimate wave height at shore
                    # Use maximum fetch in wind direction
                    wave_height = calculate_wave_height_smb(wind_speed_ms, 10000 * impact_factor)
                    
                    # Bank impact = wave energy × cos(angle)
                    bank_impact = wave_height * impact_factor
                    
                    if bank_impact > 0.02:  # Minimum threshold
                        intensity = classify_wave_intensity(bank_impact)
                    else:
                        intensity = 'calm'
                        bank_impact = 0
                else:
                    # Leeward shore - calm
                    intensity = 'calm'
                    bank_impact = 0
                
                segments.append({
                    'geometry': segment_line,
                    'impact': float(bank_impact),
                    'intensity': intensity,
                    'angle_diff': float(angle_diff),
                    'shore_normal': float(normal_deg)
                })
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(segments, crs=utm_crs)
    
    # Reproject to WGS84
    gdf = gdf.to_crs('EPSG:4326')
    
    # Save to GeoJSON
    gdf.to_file(output_path, driver='GeoJSON')
    logger.info(f"Saved {len(gdf)} bank segments to {output_path}")
    
    # Log statistics
    for intensity in ['calm', 'light', 'moderate', 'rough', 'very_rough']:
        count = len(gdf[gdf['intensity'] == intensity])
        logger.info(f"  {intensity}: {count} segments")
    
    return gdf


def generate_calm_zones(wave_grid: gpd.GeoDataFrame, output_path: Path,
                        threshold: float = 0.05):
    """
    Extract calm zones from wave grid.
    
    Args:
        wave_grid: Wave grid GeoDataFrame
        output_path: Output GeoJSON path
        threshold: Wave height threshold for calm (meters)
    """
    logger.info("Generating calm zone markers...")
    
    # Filter calm points
    calm_points = wave_grid[wave_grid['wave_height_m'] < threshold].copy()
    
    if len(calm_points) == 0:
        logger.warning("No calm zones found!")
        # Create empty GeoDataFrame
        calm_points = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
    
    # Save to GeoJSON
    calm_points.to_file(output_path, driver='GeoJSON')
    logger.info(f"Saved {len(calm_points)} calm zone markers to {output_path}")
    
    return calm_points


def main():
    parser = argparse.ArgumentParser(description='Generate wave impact layer')
    parser.add_argument('--lake', type=str, default='champlain',
                        help='Lake name')
    parser.add_argument('--wind-speed', type=float, required=True,
                        help='Wind speed in m/s')
    parser.add_argument('--wind-dir', type=float, required=True,
                        help='Wind direction in degrees (direction wind comes FROM)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory (default: auto-detect)')
    parser.add_argument('--grid-spacing', type=float, default=500.0,
                        help='Grid point spacing in meters')

    args = parser.parse_args()

    # Auto-detect paths from project root
    paths = LakePaths(args.lake)
    lake_polygon_path = paths.polygon
    fetch_dir = paths.fetch_dir
    output_dir = args.output_dir if args.output_dir else paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check inputs exist
    if not lake_polygon_path.exists():
        logger.error(f"Lake polygon not found: {lake_polygon_path}")
        return
    
    if not fetch_dir.exists():
        logger.error(f"Fetch rasters not found: {fetch_dir}")
        logger.error("Run 02_calculate_fetch.py first")
        return
    
    logger.info(f"Generating wave impact for {args.lake}")
    logger.info(f"Wind: {args.wind_speed} m/s from {args.wind_dir}°")

    # Load lake polygon
    lake = gpd.read_file(lake_polygon_path)

    # Generate wave grid (used internally by styled layers)
    wave_grid = generate_wave_grid(
        fetch_dir, args.wind_speed, args.wind_dir,
        lake, None, args.grid_spacing
    )

    # Generate bank impact
    bank_impact_path = output_dir / "bank_impact.geojson"
    generate_bank_impact(
        lake_polygon_path, args.wind_speed, args.wind_dir,
        fetch_dir, bank_impact_path
    )

    # Save metadata
    metadata = {
        'lake': args.lake,
        'wind_speed_ms': args.wind_speed,
        'wind_direction_deg': args.wind_dir,
        'grid_spacing_m': args.grid_spacing,
        'generated_at': pd.Timestamp.now().isoformat(),
        'outputs': {
            'bank_impact': str(bank_impact_path),
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Wave impact layer generation complete!")
    logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
