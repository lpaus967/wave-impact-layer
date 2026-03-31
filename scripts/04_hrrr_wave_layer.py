#!/usr/bin/env python3
"""
Generate Wave Impact Layer from Live HRRR Wind Data

Fetches current or forecast HRRR wind data and generates wave impact layers.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys
import numpy as np
import json
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.lake_config import load_lake_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_latest_hrrr_time() -> datetime:
    """Get the latest available HRRR forecast time (~3 hours ago)."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    latest = now - timedelta(hours=3)
    latest = latest.replace(minute=0, second=0, microsecond=0)
    return latest


def fetch_hrrr_wind(date: datetime, lat: float, lon: float):
    """
    Fetch current HRRR analysis wind components for a specific location.

    Args:
        date: HRRR cycle datetime
        lat: Latitude
        lon: Longitude

    Returns:
        Tuple of (u_wind, v_wind) in m/s
    """
    from herbie import Herbie

    logger.info(f"Fetching HRRR wind for {date.strftime('%Y-%m-%d %H:00')}")
    logger.info(f"Location: {lat}, {lon}")

    # Try the requested cycle, then fall back to earlier cycles
    max_retries = 4
    for attempt in range(max_retries):
        try_date = date - timedelta(hours=attempt)
        try:
            if attempt > 0:
                logger.info(f"Retrying with HRRR cycle {try_date.strftime('%Y-%m-%d %H:00')}")

            H = Herbie(try_date, model='hrrr', product='sfc', fxx=0)

            ds_u = H.xarray("UGRD:10 m")
            ds_v = H.xarray("VGRD:10 m")

            u_data = ds_u['u10'].values
            v_data = ds_v['v10'].values
            lats = ds_u['latitude'].values
            lons = ds_u['longitude'].values

            if lats.ndim == 2:
                dist = np.sqrt((lats - lat)**2 + (lons - lon)**2)
                idx = np.unravel_index(np.argmin(dist), dist.shape)
                u_wind = float(u_data[idx])
                v_wind = float(v_data[idx])
            else:
                lat_idx = np.argmin(np.abs(lats - lat))
                lon_idx = np.argmin(np.abs(lons - lon))
                u_wind = float(u_data[lat_idx, lon_idx])
                v_wind = float(v_data[lat_idx, lon_idx])

            logger.info(f"U wind: {u_wind:.2f} m/s, V wind: {v_wind:.2f} m/s")

            return u_wind, v_wind

        except (FileNotFoundError, OSError) as e:
            logger.warning(f"HRRR cycle {try_date.strftime('%Y-%m-%d %H:00')} not available: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch HRRR data after {max_retries} cycle attempts")
                raise
        except Exception as e:
            logger.error(f"Failed to fetch HRRR data: {e}")
            raise


def calculate_wind_speed_direction(u: float, v: float) -> tuple:
    """
    Calculate wind speed and direction from U/V components.
    
    Args:
        u: U (eastward) wind component in m/s
        v: V (northward) wind component in m/s
        
    Returns:
        Tuple of (speed_ms, direction_deg)
        Direction is where wind comes FROM (meteorological convention)
    """
    # Speed
    speed = np.sqrt(u**2 + v**2)
    
    # Direction (meteorological: where wind comes FROM)
    # atan2 gives direction wind is going TO, so add 180
    direction = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    
    return speed, direction


def generate_wave_layer_from_hrrr(lake: str, date: datetime = None,
                                  output_dir: Path = Path('data/output')):
    """
    Generate wave impact layer using current HRRR analysis wind data.
    """
    if date is None:
        date = get_latest_hrrr_time()

    # Get lake center from config
    config = load_lake_config(lake)

    # Fetch wind data
    u_wind, v_wind = fetch_hrrr_wind(date, config.lat, config.lon)
    
    # Calculate speed and direction
    wind_speed, wind_direction = calculate_wind_speed_direction(u_wind, v_wind)
    
    logger.info(f"Wind: {wind_speed:.1f} m/s from {wind_direction:.0f}°")
    
    # Convert to common wind descriptions
    wind_mph = wind_speed * 2.237
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    dir_idx = int((wind_direction + 11.25) / 22.5) % 16
    dir_name = directions[dir_idx]
    
    logger.info(f"Wind: {wind_mph:.0f} mph from {dir_name}")
    
    # Call the wave layer generator
    script_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable, str(script_dir / '03_generate_wave_layer.py'),
        '--lake', lake,
        '--wind-speed', str(wind_speed),
        '--wind-dir', str(wind_direction),
        '--output-dir', str(output_dir)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Wave layer generation failed:\n{result.stderr}")
        return None
    
    logger.info(result.stdout)
    
    # Update metadata with HRRR info
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        metadata['hrrr'] = {
            'analysis_time': date.strftime('%Y-%m-%d %H:00 UTC'),
            'u_wind_ms': u_wind,
            'v_wind_ms': v_wind,
            'wind_speed_ms': wind_speed,
            'wind_speed_mph': wind_mph,
            'wind_direction_deg': wind_direction,
            'wind_direction_name': dir_name
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Generate wave layer from current HRRR winds')
    parser.add_argument('--lake', type=str, default='champlain',
                        help='Lake ID (must have config in data/lakes/{lake}/config.json)')
    parser.add_argument('--date', type=str,
                        help='HRRR cycle date (YYYY-MM-DD)')
    parser.add_argument('--cycle', type=int,
                        help='HRRR cycle hour (0-23)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory (default: auto-detect)')

    args = parser.parse_args()

    # Auto-detect output dir if not provided
    if args.output_dir is None:
        from lib.paths import LakePaths
        args.output_dir = LakePaths(args.lake).output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine HRRR cycle time
    if args.date:
        date = datetime.strptime(f"{args.date} {args.cycle or 0}", "%Y-%m-%d %H")
    else:
        date = get_latest_hrrr_time()

    logger.info(f"Using HRRR analysis: {date.strftime('%Y-%m-%d %H:00 UTC')}")

    generate_wave_layer_from_hrrr(args.lake, date, args.output_dir)


if __name__ == '__main__':
    main()
