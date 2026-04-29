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
from lib.wave_physics import (
    wave_height_young_verhagen,
    classify_wave_intensity,
    effective_fetch,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _open_fetch_rasters(fetch_dir: Path):
    """Open all fetch rasters and return (dict, ref_raster, fetch_index)."""
    index_path = fetch_dir / "fetch_index.json"
    with open(index_path) as f:
        fetch_index = json.load(f)

    fetch_rasters = {}
    for dir_str, filename in fetch_index['files'].items():
        direction = float(dir_str)
        fetch_rasters[direction] = rasterio.open(fetch_dir / filename)

    ref_raster = list(fetch_rasters.values())[0]
    return fetch_rasters, ref_raster, fetch_index


def generate_wave_grid(fetch_dir: Path, wind_speed_ms: float, wind_direction: float,
                       lake_polygon: gpd.GeoDataFrame, output_path: Path,
                       grid_spacing: float = 500.0, depth_m: float = 20.0):
    """
    Generate wave intensity grid as points for Mapbox visualization.

    Uses SPM effective fetch (9 cosine-weighted radials) and Young & Verhagen
    (1996) wave height formula for physically accurate results.

    Args:
        fetch_dir: Directory with fetch rasters
        wind_speed_ms: Wind speed in m/s
        wind_direction: Wind direction in degrees
        lake_polygon: Lake boundary GeoDataFrame
        output_path: Output GeoJSON path
        grid_spacing: Spacing between points in meters
        depth_m: Average lake depth in meters
    """
    logger.info("Generating wave intensity grid...")

    fetch_rasters, ref_raster, fetch_index = _open_fetch_rasters(fetch_dir)

    transform = ref_raster.transform
    crs = ref_raster.crs
    height, width = ref_raster.shape
    cell_size = abs(transform.a)

    # Compute SPM effective fetch (cosine-weighted across 9 radials)
    fetch = effective_fetch(fetch_rasters, wind_direction)

    # Young & Verhagen (1996) wave height — handles depth limitation naturally
    wave_height = wave_height_young_verhagen(wind_speed_ms, fetch, depth_m)
    
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
                         utm_crs: str = None, depth_m: float = 20.0):
    """
    Generate shoreline impact segments using actual fetch data.

    Samples the effective fetch raster at each shore segment midpoint, then
    computes wave height via Young & Verhagen (1996). Impact is scored using
    Hs^2 * cos(angle) — proportional to wave energy flux normal to the shore.

    Args:
        lake_polygon_path: Path to lake polygon GeoJSON
        wind_speed_ms: Wind speed in m/s
        wind_direction: Wind direction in degrees
        fetch_dir: Directory with fetch rasters
        output_path: Output GeoJSON path
        segment_length: Target length of shoreline segments in meters
        depth_m: Average lake depth in meters
    """
    logger.info("Generating bank impact segments...")

    # Load lake polygon
    lake = gpd.read_file(lake_polygon_path)

    # Load fetch rasters and compute effective fetch
    fetch_rasters, ref_raster, fetch_index = _open_fetch_rasters(fetch_dir)

    if utm_crs is None:
        utm_crs = fetch_index.get('crs', 'EPSG:32618')

    fetch_transform = ref_raster.transform
    fetch_arr = effective_fetch(fetch_rasters, wind_direction)

    # Close rasters
    for r in fetch_rasters.values():
        r.close()

    def get_fetch_at_point(x, y):
        """Sample effective fetch at a UTM coordinate."""
        try:
            row, col = rasterio.transform.rowcol(fetch_transform, x, y)
            if 0 <= row < fetch_arr.shape[0] and 0 <= col < fetch_arr.shape[1]:
                return float(fetch_arr[row, col])
        except Exception:
            pass
        return 0.0

    # Reproject to UTM for accurate geometry operations
    lake_utm = lake.to_crs(utm_crs)

    # Get the exterior boundary
    boundary = lake_utm.geometry.iloc[0]
    if isinstance(boundary, (Polygon, MultiPolygon)):
        if isinstance(boundary, MultiPolygon):
            rings = [p.exterior for p in boundary.geoms]
            for p in boundary.geoms:
                rings.extend(list(p.interiors))
        else:
            rings = [boundary.exterior] + list(boundary.interiors)
    else:
        rings = [boundary]

    # Segment the shoreline
    segments = []

    for ring in rings:
        total_length = ring.length
        n_segments = max(1, int(total_length / segment_length))

        for i in range(n_segments):
            start_dist = i * total_length / n_segments
            end_dist = (i + 1) * total_length / n_segments
            mid_dist = (start_dist + end_dist) / 2

            mid_point = ring.interpolate(mid_dist)
            start_point = ring.interpolate(start_dist)
            end_point = ring.interpolate(end_dist)

            segment_line = LineString([start_point, end_point])

            dx = end_point.x - start_point.x
            dy = end_point.y - start_point.y
            length = np.sqrt(dx**2 + dy**2)

            if length > 0:
                # Normal pointing INTO the lake (right of CCW traversal)
                normal_x = dy / length
                normal_y = -dx / length
                normal_deg = np.degrees(np.arctan2(normal_x, normal_y)) % 360

                # Angle between wind direction and inward shore normal
                angle_diff = abs(wind_direction - normal_deg)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                impact_factor = np.cos(np.radians(angle_diff))

                if impact_factor > 0:
                    # Sample actual fetch at the shore segment midpoint
                    fetch_m = get_fetch_at_point(mid_point.x, mid_point.y)

                    # Young & Verhagen wave height using real fetch
                    hs = wave_height_young_verhagen(wind_speed_ms, fetch_m, depth_m)

                    # Impact proportional to wave energy flux normal to shore
                    bank_impact = float(hs ** 2 * impact_factor)

                    if bank_impact > 0.001:
                        # Classify using wave height (not energy) for user-facing labels
                        intensity = classify_wave_intensity(hs * impact_factor)
                    else:
                        intensity = 'calm'
                        bank_impact = 0
                else:
                    intensity = 'calm'
                    bank_impact = 0
                    fetch_m = 0
                    hs = 0

                segments.append({
                    'geometry': segment_line,
                    'impact': float(bank_impact),
                    'wave_height_m': float(hs),
                    'intensity': intensity,
                    'fetch_m': float(fetch_m),
                    'angle_diff': float(angle_diff),
                    'shore_normal': float(normal_deg)
                })

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(segments, crs=utm_crs)
    gdf = gdf.to_crs('EPSG:4326')

    gdf.to_file(output_path, driver='GeoJSON')
    logger.info(f"Saved {len(gdf)} bank segments to {output_path}")

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
    
    # Load lake config for depth
    config = load_lake_config(args.lake)
    depth_m = config.avg_depth_m

    logger.info(f"Generating wave impact for {args.lake}")
    logger.info(f"Wind: {args.wind_speed} m/s from {args.wind_dir}°")
    logger.info(f"Using Young & Verhagen (1996) wave model, depth={depth_m:.1f}m")

    # Load lake polygon
    lake = gpd.read_file(lake_polygon_path)

    # Generate wave grid (used internally by styled layers)
    wave_grid = generate_wave_grid(
        fetch_dir, args.wind_speed, args.wind_dir,
        lake, None, args.grid_spacing, depth_m
    )

    # Generate bank impact
    bank_impact_path = output_dir / "bank_impact.geojson"
    generate_bank_impact(
        lake_polygon_path, args.wind_speed, args.wind_dir,
        fetch_dir, bank_impact_path, depth_m=depth_m
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
