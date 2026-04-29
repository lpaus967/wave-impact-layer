#!/usr/bin/env python3
"""
Batch Processing Pipeline for Wave Impact Layers

Queries PostGIS for lakes, fetches HRRR wind, and runs the full wave pipeline.

Usage:
    python scripts/run_pipeline.py --lake "Lake Champlain"
    python scripts/run_pipeline.py --state VT --workers 4
    python scripts/run_pipeline.py --min-area 50 --workers 8
    python scripts/run_pipeline.py --state VT --min-area 10 --workers 2
    python scripts/run_pipeline.py --lake "Lake Champlain" --wind-speed 15 --wind-dir 180  # manual override
"""

import argparse
import json
import logging
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import proj_fix  # noqa: E402,F401
from lib.lake_config import load_lake_config, list_lakes_from_db, list_lakes_local, LakeConfig
from lib.paths import LakePaths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent


def _raster_valid(raster_path: Path) -> bool:
    """Check that a raster has a valid CRS and non-zero data."""
    try:
        import rasterio
        with rasterio.open(raster_path) as r:
            if r.crs is None:
                return False
            data = r.read(1)
            return data.max() > 0
    except Exception:
        return False


def _fetch_data_valid(fetch_index_path: Path, min_directions: int = 36) -> bool:
    """Check that fetch rasters have a valid CRS, enough directions, and actual data."""
    try:
        with open(fetch_index_path) as f:
            index = json.load(f)
        crs = index.get('crs', '')
        if not crs or crs == 'None':
            return False
        # Check we have enough directions (36 for effective fetch accuracy)
        files = index.get('files', {})
        if len(files) < min_directions:
            return False
        # Spot-check the first raster file has non-zero data
        first_file = list(files.values())[0]
        raster_path = fetch_index_path.parent / first_file
        if not raster_path.exists():
            return False
        import rasterio
        with rasterio.open(raster_path) as r:
            if r.crs is None:
                return False
            data = r.read(1)
            if data.max() == 0:
                return False
        return True
    except Exception:
        return False


def auto_resolution(area_km2: float) -> float:
    """Auto-scale raster resolution based on lake area to keep computation manageable."""
    # Small lakes: 100m, large lakes: up to 500m
    return max(100.0, min(500.0, 50.0 * math.sqrt(area_km2 / 10.0)))


def _get_subprocess_env():
    """Get environment with PROJ_LIB set correctly for child processes."""
    import os
    env = os.environ.copy()
    # proj_fix already ran and set PROJ_LIB/PROJ_DATA — propagate them
    for key in ('PROJ_LIB', 'PROJ_DATA'):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


_SUBPROCESS_ENV = None


def run_step(cmd: list, step_name: str, lake_id: str) -> bool:
    """Run a pipeline step as a subprocess."""
    global _SUBPROCESS_ENV
    if _SUBPROCESS_ENV is None:
        _SUBPROCESS_ENV = _get_subprocess_env()

    logger.info(f"[{lake_id}] Running {step_name}...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            env=_SUBPROCESS_ENV
        )
        if result.returncode != 0:
            logger.error(f"[{lake_id}] {step_name} failed:\n{result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"[{lake_id}] {step_name} timed out (10 min)")
        return False
    except Exception as e:
        logger.error(f"[{lake_id}] {step_name} error: {e}")
        return False


def read_wind_from_metadata(output_dir: Path) -> tuple:
    """
    Read wind speed and direction from metadata.json written by script 03/04.

    Returns:
        (wind_speed_ms, wind_direction_deg) or (None, None) if not found
    """
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return None, None

    with open(metadata_path) as f:
        meta = json.load(f)

    speed = meta.get('wind_speed_ms')
    direction = meta.get('wind_direction_deg')

    # Also check nested hrrr block
    if speed is None and 'hrrr' in meta:
        speed = meta['hrrr'].get('wind_speed_ms')
        direction = meta['hrrr'].get('wind_direction_deg')

    return speed, direction


def process_lake(config: LakeConfig, wind_speed: float = None, wind_dir: float = None,
                 steps: list = None):
    """
    Run the full pipeline for a single lake.

    Static data (polygon, raster, fetch rasters) is automatically skipped if
    outputs already exist. Wind-dependent steps (3, 5, 6) always re-run.

    By default, wind data is fetched from HRRR (script 04). If --wind-speed
    and --wind-dir are provided, those are used instead (script 03 directly).

    Args:
        config: LakeConfig instance (from DB or local JSON)
        wind_speed: Manual wind speed override in m/s (default: use HRRR)
        wind_dir: Manual wind direction override in degrees (default: use HRRR)
        steps: List of step numbers to run (default: all)

    Returns:
        Dict with lake_id and status
    """
    if steps is None:
        steps = [1, 2, 3, 5, 6]

    manual_wind = wind_speed is not None and wind_dir is not None
    lake_id = config.lake_id

    paths = LakePaths(lake_id)
    paths.ensure_dirs()

    resolution = auto_resolution(config.area_km2) if config.area_km2 > 0 else 100.0
    python = sys.executable

    # Step 1: Prepare lake (download polygon + rasterize)
    # Auto-skip if outputs already exist and raster is valid (geometry is static)
    if 1 in steps:
        if paths.polygon.exists() and paths.raster.exists() and _raster_valid(paths.raster):
            logger.info(f"[{lake_id}] Skipping prepare (valid polygon + raster exist)")
        else:
            ok = run_step([
                python, str(SCRIPTS_DIR / '01_prepare_lake.py'),
                '--lake', config.name,
                '--resolution', str(resolution),
            ], 'prepare', lake_id)
            if not ok:
                return {'lake_id': lake_id, 'status': 'failed', 'step': 'prepare'}

    # Step 2: Calculate fetch
    # Auto-skip if outputs already exist and are valid (fetch is purely geometric)
    if 2 in steps:
        fetch_index = paths.fetch_dir / "fetch_index.json"
        if fetch_index.exists() and _fetch_data_valid(fetch_index):
            logger.info(f"[{lake_id}] Skipping fetch (valid fetch rasters exist)")
        else:
            ok = run_step([
                python, str(SCRIPTS_DIR / '02_calculate_fetch.py'),
                '--lake', lake_id,
            ], 'fetch', lake_id)
            if not ok:
                return {'lake_id': lake_id, 'status': 'failed', 'step': 'fetch'}

    # Step 3: Generate wave layer
    if 3 in steps:
        if manual_wind:
            ok = run_step([
                python, str(SCRIPTS_DIR / '03_generate_wave_layer.py'),
                '--lake', lake_id,
                '--wind-speed', str(wind_speed),
                '--wind-dir', str(wind_dir),
            ], 'wave_layer', lake_id)
        else:
            # Default: fetch live HRRR wind and generate wave layer
            ok = run_step([
                python, str(SCRIPTS_DIR / '04_hrrr_wave_layer.py'),
                '--lake', lake_id,
            ], 'hrrr_wave', lake_id)

        if not ok:
            return {'lake_id': lake_id, 'status': 'failed', 'step': 'wave_layer'}

    # Read wind values from metadata for downstream steps
    # (HRRR writes these; manual wind was passed in via args)
    if not manual_wind:
        wind_speed, wind_dir = read_wind_from_metadata(paths.output_dir)
        if wind_speed is None:
            logger.warning(f"[{lake_id}] Could not read wind from metadata — "
                           "skipping styled layers and shelters")
            return {'lake_id': lake_id, 'status': 'partial', 'note': 'no wind metadata'}

    # Step 5: Styled layers
    if 5 in steps:
        ok = run_step([
            python, str(SCRIPTS_DIR / '05_generate_styled_layers.py'),
            '--lake', lake_id,
            '--wind-speed', str(wind_speed),
            '--wind-dir', str(wind_dir),
        ], 'styled', lake_id)
        if not ok:
            return {'lake_id': lake_id, 'status': 'failed', 'step': 'styled'}

    # Step 6: Micro-shelters
    if 6 in steps:
        ok = run_step([
            python, str(SCRIPTS_DIR / '06_micro_shelters.py'),
            '--lake', lake_id,
            '--wind-speed', str(wind_speed),
            '--wind-dir', str(wind_dir),
        ], 'shelters', lake_id)
        if not ok:
            return {'lake_id': lake_id, 'status': 'failed', 'step': 'shelters'}

    return {'lake_id': lake_id, 'status': 'success'}


def get_lake_configs(args) -> list:
    """
    Get lake configs, preferring local data over database queries.

    For single-lake runs, loads from local config.json if it exists.
    For multi-lake runs, scans local configs first; only queries the DB
    if no matching local configs are found.
    """
    min_area = args.min_area if hasattr(args, 'min_area') and args.min_area else 5.0
    states = None
    if args.state:
        states = [s.strip().upper() for s in args.state.split(',')]

    if args.lake:
        # Single lake by name — load_lake_config checks local first
        config = load_lake_config(args.lake)
        return [config]

    # Query the database for all matching lakes
    logger.info(f"Querying database for lakes "
                f"(min area: {min_area} km², states: {states or 'all'})...")
    configs = list_lakes_from_db(min_area_km2=min_area, states=states)

    # Merge with local configs (local takes precedence for lakes that exist locally)
    local_configs = list_lakes_local(min_area_km2=min_area, states=states)
    local_ids = {c.lake_id for c in local_configs}
    merged = list(local_configs)
    for cfg in configs:
        if cfg.lake_id not in local_ids:
            merged.append(cfg)

    logger.info(f"Found {len(merged)} lake(s) "
                f"({len(local_ids)} local, {len(merged) - len(local_ids)} from database)")
    return merged


def main():
    parser = argparse.ArgumentParser(description='Run wave impact pipeline for lakes')
    parser.add_argument('--lake', type=str,
                        help='Single lake name (e.g. "Lake Champlain")')
    parser.add_argument('--state', type=str,
                        help='Comma-separated state abbreviations (e.g. VT,NY)')
    parser.add_argument('--min-area', type=float, default=5.0,
                        help='Minimum lake area in km² (default: 5.0)')
    parser.add_argument('--wind-speed', type=float, default=None,
                        help='Manual wind speed override in m/s (default: use HRRR)')
    parser.add_argument('--wind-dir', type=float, default=None,
                        help='Manual wind direction override in degrees (default: use HRRR)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--steps', type=str, default=None,
                        help='Comma-separated step numbers to run (e.g., "1,2" for prepare+fetch)')

    args = parser.parse_args()

    # Parse steps
    steps = None
    if args.steps:
        steps = [int(s) for s in args.steps.split(',')]

    configs = get_lake_configs(args)
    if not configs:
        logger.error("No lakes found to process!")
        return

    wind_source = "manual" if (args.wind_speed and args.wind_dir) else "HRRR"
    logger.info(f"Processing {len(configs)} lake(s) with {args.workers} worker(s) "
                f"(wind: {wind_source})")

    results = []

    if args.workers <= 1:
        for cfg in configs:
            result = process_lake(
                cfg, args.wind_speed, args.wind_dir, steps
            )
            results.append(result)
            logger.info(f"  {result['lake_id']}: {result['status']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_lake, cfg, args.wind_speed, args.wind_dir, steps
                ): cfg.lake_id
                for cfg in configs
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                logger.info(f"  {result['lake_id']}: {result['status']}")

    # Summary
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    errors = sum(1 for r in results if r['status'] == 'error')

    logger.info(f"\nPipeline complete: {success} success, {failed} failed, {errors} errors")

    # Write results index
    index_path = Path('data/output/index.json')
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, 'w') as f:
        json.dump({
            'total': len(results),
            'success': success,
            'failed': failed,
            'results': results,
        }, f, indent=2)


if __name__ == '__main__':
    main()
