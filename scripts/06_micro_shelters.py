#!/usr/bin/env python3
"""
Micro-Shelter Detection

Identifies sheltered coves and bays based on current wind direction and wave height.
Produces clean vector polygons clipped to the lake boundary.

Output: micro_shelters.geojson with labeled shelter polygons
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
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import Point, shape, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid

from lib.lake_config import load_lake_config
from lib.paths import LakePaths
from lib.wave_physics import wave_height_young_verhagen, effective_fetch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _load_known_bays(lake_id: str) -> list:
    """Load known bays from data/lakes/{lake_id}/bays.json if it exists."""
    paths = LakePaths(lake_id)
    if paths.bays.exists():
        with open(paths.bays) as f:
            return json.load(f)
    return []


def detect_micro_shelters(lake_polygon_path: Path, fetch_dir: Path,
                          wind_direction: float, wind_speed_ms: float,
                          output_path: Path,
                          lake_name: str = 'champlain',
                          depth_m: float = 20.0,
                          wave_threshold: float = 0.10,
                          min_shelter_area: float = 50000.0):
    """
    Detect sheltered areas where wave height is below a threshold.

    Produces clean vector polygons by intersecting the raster-derived calm
    zones with the lake polygon, so shoreline boundaries are exact.

    Args:
        lake_polygon_path: Path to lake polygon GeoJSON
        fetch_dir: Directory with fetch rasters
        wind_direction: Wind direction in degrees (FROM)
        wind_speed_ms: Wind speed in m/s
        output_path: Output GeoJSON path
        lake_name: Lake name for known bay lookup
        depth_m: Average lake depth in meters
        wave_threshold: Max significant wave height (m) to be considered sheltered
        min_shelter_area: Minimum shelter area in m²
    """
    logger.info(f"Detecting micro-shelters for wind {wind_speed_ms:.1f} m/s from {wind_direction:.0f}°...")
    logger.info(f"Shelter threshold: Hs < {wave_threshold:.2f}m")

    # Load lake polygon
    lake = gpd.read_file(lake_polygon_path)

    # Load fetch rasters
    index_path = fetch_dir / "fetch_index.json"
    with open(index_path) as f:
        fetch_index = json.load(f)

    utm_crs = fetch_index.get('crs', 'EPSG:32618')
    lake_utm = lake.to_crs(utm_crs)
    lake_geom = lake_utm.geometry.iloc[0]
    lake_geom = make_valid(lake_geom)

    # Open fetch rasters
    fetch_rasters = {}
    for dir_str, filename in fetch_index['files'].items():
        fetch_rasters[float(dir_str)] = rasterio.open(fetch_dir / filename)

    ref_raster = list(fetch_rasters.values())[0]
    fetch_transform = ref_raster.transform
    cell_size = abs(fetch_transform.a)

    # Compute effective fetch and wave height
    eff_fetch = effective_fetch(fetch_rasters, wind_direction)
    wave_ht = wave_height_young_verhagen(wind_speed_ms, eff_fetch, depth_m)

    # Close rasters
    for r in fetch_rasters.values():
        r.close()

    # Determine shelter threshold: use the explicit value OR a relative
    # threshold (25th percentile of wave heights on the lake), whichever is
    # larger.  This ensures shelters are meaningful even in strong wind.
    water_mask = eff_fetch > 0
    water_waves = wave_ht[water_mask]
    if len(water_waves) > 0:
        p25 = float(np.percentile(water_waves[water_waves > 0], 25))
        adaptive_threshold = max(wave_threshold, p25)
    else:
        adaptive_threshold = wave_threshold

    logger.info(f"Adaptive shelter threshold: Hs < {adaptive_threshold:.3f}m "
                f"(fixed={wave_threshold:.2f}m, p25={p25:.3f}m)")

    # Create shelter mask: water cells where wave height is below threshold
    shelter_mask = water_mask & (wave_ht < adaptive_threshold)

    # Morphological cleanup: remove noise, fill small holes
    shelter_mask = ndimage.binary_opening(shelter_mask, iterations=1)
    shelter_mask = ndimage.binary_closing(shelter_mask, iterations=2)

    # Label connected regions
    labeled, num_features = ndimage.label(shelter_mask)
    logger.info(f"Found {num_features} potential shelter regions")

    # Vectorize the labeled regions
    raw_polygons = []
    for geom_dict, value in shapes(labeled.astype('int32'), transform=fetch_transform):
        if value == 0:
            continue
        poly = shape(geom_dict)
        if poly.area >= min_shelter_area * 0.5:  # loose pre-filter before clipping
            raw_polygons.append(poly)

    logger.info(f"Vectorized {len(raw_polygons)} regions above pre-filter area")

    # Intersect each polygon with the lake boundary for clean edges
    shelter_polygons = []
    for raw_poly in raw_polygons:
        try:
            raw_poly = make_valid(raw_poly)
            clipped = raw_poly.intersection(lake_geom)
            if clipped.is_empty:
                continue

            # Handle GeometryCollection — extract only polygons
            if clipped.geom_type == 'GeometryCollection':
                parts = [g for g in clipped.geoms
                         if g.geom_type in ('Polygon', 'MultiPolygon')]
                if not parts:
                    continue
                clipped = unary_union(parts)

            # Simplify slightly to smooth jagged raster edges while
            # keeping the lake-boundary edges crisp (they come from the
            # intersection, not the raster)
            clipped = clipped.simplify(cell_size * 0.5, preserve_topology=True)

            if clipped.is_empty or clipped.area < min_shelter_area:
                continue

            # Sample average fetch and wave height inside the shelter
            centroid = clipped.representative_point()
            try:
                row, col = rasterio.transform.rowcol(fetch_transform, centroid.x, centroid.y)
                if 0 <= row < eff_fetch.shape[0] and 0 <= col < eff_fetch.shape[1]:
                    avg_fetch = float(eff_fetch[row, col])
                    avg_wave = float(wave_ht[row, col])
                else:
                    avg_fetch = 0.0
                    avg_wave = 0.0
            except Exception:
                avg_fetch = 0.0
                avg_wave = 0.0

            shelter_polygons.append({
                'geometry': clipped,
                'area_m2': clipped.area,
                'area_acres': clipped.area / 4047,
                'avg_fetch_m': avg_fetch,
                'avg_wave_height_m': avg_wave,
            })
        except Exception as e:
            logger.debug(f"Skipping region: {e}")
            continue

    logger.info(f"Found {len(shelter_polygons)} shelters above {min_shelter_area:.0f} m²")

    # Create GeoDataFrame
    if shelter_polygons:
        gdf = gpd.GeoDataFrame(shelter_polygons, crs=utm_crs)
    else:
        gdf = gpd.GeoDataFrame(
            columns=['geometry', 'area_m2', 'area_acres', 'avg_fetch_m',
                     'avg_wave_height_m', 'name', 'protection'],
            geometry='geometry', crs=utm_crs
        )

    # Name shelters
    known_bays = _load_known_bays(lake_name)
    if len(gdf) > 0:
        _assign_names(gdf, known_bays, utm_crs)

    # Add protection level and wind info
    if len(gdf) > 0:
        gdf['protection'] = gdf['avg_fetch_m'].apply(
            lambda f: 'excellent' if f < 500 else ('good' if f < 1000 else 'moderate')
        )
        gdf['sheltered_from'] = get_wind_direction_name(wind_direction)
        gdf['wind_direction_deg'] = wind_direction

    # Reproject to WGS84
    gdf = gdf.to_crs('EPSG:4326')

    gdf.to_file(output_path, driver='GeoJSON')
    logger.info(f"Saved {len(gdf)} micro-shelters to {output_path}")

    for _, row in gdf.iterrows():
        logger.info(f"  {row.get('name', 'Unknown')}: {row.get('protection', '?')} protection, "
                     f"{row.get('area_acres', 0):.0f} acres, Hs={row.get('avg_wave_height_m', 0):.2f}m")

    return gdf


def _assign_names(gdf, known_bays, crs):
    """Assign names from known bays or generate generic ones."""
    gdf['name'] = None
    gdf['is_named'] = False

    if known_bays:
        for idx, row in gdf.iterrows():
            centroid_wgs84 = gpd.GeoSeries(
                [row.geometry.representative_point()], crs=crs
            ).to_crs('EPSG:4326').iloc[0]

            best_match = None
            best_dist = float('inf')

            for bay in known_bays:
                dist = np.sqrt((centroid_wgs84.x - bay['center'][0])**2 +
                               (centroid_wgs84.y - bay['center'][1])**2)
                dist_m = dist * 111000

                if dist_m < bay.get('radius', 5000) and dist_m < best_dist:
                    best_match = bay['name']
                    best_dist = dist_m

            if best_match:
                gdf.at[idx, 'name'] = best_match
                gdf.at[idx, 'is_named'] = True

    # Fill unnamed shelters with generic names
    unnamed_count = 0
    for idx, row in gdf.iterrows():
        if row['name'] is None:
            unnamed_count += 1
            gdf.at[idx, 'name'] = f"Shelter Zone {unnamed_count}"


def generate_shelter_labels(shelters_gdf: gpd.GeoDataFrame, output_path: Path):
    """Generate point labels for shelter zones (for map labeling)."""
    if len(shelters_gdf) == 0:
        gpd.GeoDataFrame({'geometry': []}).to_file(output_path, driver='GeoJSON')
        return

    labels = []
    for idx, row in shelters_gdf.iterrows():
        labels.append({
            'geometry': row.geometry.representative_point(),
            'name': row.get('name', f'Shelter {idx}'),
            'protection': row.get('protection', 'moderate'),
            'area_acres': row.get('area_acres', 0),
            'avg_fetch_m': row.get('avg_fetch_m', 0),
            'avg_wave_height_m': row.get('avg_wave_height_m', 0),
            'sheltered_from': row.get('sheltered_from', ''),
        })

    gdf = gpd.GeoDataFrame(labels, crs=shelters_gdf.crs)
    gdf.to_file(output_path, driver='GeoJSON')
    logger.info(f"Saved {len(gdf)} shelter labels to {output_path}")
    return gdf


def get_wind_direction_name(direction_deg: float) -> str:
    """Convert degrees to compass direction name."""
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    idx = int((direction_deg + 11.25) / 22.5) % 16
    return directions[idx]


def main():
    parser = argparse.ArgumentParser(description='Detect micro-shelters')
    parser.add_argument('--lake', type=str, default='champlain',
                        help='Lake name')
    parser.add_argument('--wind-dir', type=float, required=True,
                        help='Wind direction in degrees (FROM)')
    parser.add_argument('--wind-speed', type=float, default=None,
                        help='Wind speed in m/s (reads from metadata if omitted)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory (default: auto-detect)')
    parser.add_argument('--wave-threshold', type=float, default=0.10,
                        help='Max wave height (m) to be considered sheltered (default: 0.10)')
    parser.add_argument('--min-area', type=float, default=50000.0,
                        help='Minimum shelter area in m²')

    args = parser.parse_args()

    # Auto-detect paths from project root
    config = load_lake_config(args.lake)
    paths = LakePaths(config.lake_id)
    lake_polygon_path = paths.polygon
    fetch_dir = paths.fetch_dir
    output_dir = args.output_dir if args.output_dir else paths.output_dir

    if not lake_polygon_path.exists():
        logger.error(f"Lake polygon not found: {lake_polygon_path}")
        return

    if not fetch_dir.exists():
        logger.error(f"Fetch rasters not found: {fetch_dir}")
        return

    # Get wind speed from args or metadata
    wind_speed = args.wind_speed
    if wind_speed is None:
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            wind_speed = meta.get('wind_speed_ms')
            if wind_speed is None and 'hrrr' in meta:
                wind_speed = meta['hrrr'].get('wind_speed_ms')
        if wind_speed is None:
            logger.error("No --wind-speed provided and no metadata.json found")
            return

    # Detect shelters
    shelters_path = output_dir / "micro_shelters.geojson"
    shelters = detect_micro_shelters(
        lake_polygon_path, fetch_dir,
        args.wind_dir, wind_speed,
        shelters_path,
        lake_name=config.lake_id,
        depth_m=config.avg_depth_m,
        wave_threshold=args.wave_threshold,
        min_shelter_area=args.min_area,
    )

    # Generate labels
    labels_path = output_dir / "shelter_labels.geojson"
    generate_shelter_labels(shelters, labels_path)

    logger.info("Micro-shelter detection complete!")


if __name__ == '__main__':
    main()
