"""
PostGIS database connection and lake queries.

Replaces JSON config files — the database is the single source of truth
for lake metadata, geometry, and state filtering.
"""

import os
from pathlib import Path
from functools import lru_cache

import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine

from .geo_utils import utm_epsg_from_lonlat
from .depth_estimation import estimate_depth


def _load_dotenv():
    """Load .env file from project root if it exists."""
    # Walk up from this file to find project root
    current = Path(__file__).resolve()
    for parent in current.parents:
        env_path = parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, _, value = line.partition('=')
                        os.environ.setdefault(key.strip(), value.strip())
            break


def get_connection_string() -> str:
    """Build SQLAlchemy connection string from environment variables."""
    _load_dotenv()

    host = os.environ.get('DB_HOST', 'localhost')
    port = os.environ.get('DB_PORT', '5432')
    name = os.environ.get('DB_NAME', 'curation_env_onwater')
    user = os.environ.get('DB_USER', 'dbmasteruser')
    password = os.environ.get('DB_PASSWORD', '')

    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


@lru_cache(maxsize=1)
def get_engine():
    """Get a cached SQLAlchemy engine."""
    return create_engine(get_connection_string())


def query_lakes(min_area_km2: float = 5.0, states: list = None,
                limit: int = None) -> gpd.GeoDataFrame:
    """
    Query lakes from PostGIS with optional area and state filters.

    Args:
        min_area_km2: Minimum lake area in km^2
        states: Optional list of state abbreviations to filter by
        limit: Maximum number of lakes to return

    Returns:
        GeoDataFrame with columns: uuid, name, area_sqkm, geom
    """
    engine = get_engine()

    if states:
        placeholders = ", ".join(f"'{s}'" for s in states)
        where_clauses = [
            f"l.area_sqkm >= {min_area_km2}",
            "l.name IS NOT NULL",
            f"s.stusps IN ({placeholders})",
        ]
        where_sql = " AND ".join(where_clauses)
        sql = f"""
            SELECT l.uuid, l.name, l.area_sqkm, l.geom
            FROM hydrology.lakes l
            JOIN admin_boundaries_usa.us_states s ON ST_Intersects(l.geom, s.geom)
            WHERE {where_sql}
            ORDER BY l.area_sqkm DESC
        """
    else:
        where_clauses = [
            f"area_sqkm >= {min_area_km2}",
            "name IS NOT NULL",
        ]
        where_sql = " AND ".join(where_clauses)
        sql = f"""
            SELECT uuid, name, area_sqkm, geom
            FROM hydrology.lakes
            WHERE {where_sql}
            ORDER BY area_sqkm DESC
        """

    if limit:
        sql += f" LIMIT {limit}"

    conn = engine.raw_connection()
    try:
        gdf = gpd.read_postgis(sql, conn, geom_col='geom')
    finally:
        conn.close()
    return gdf


def get_lake_by_name(name: str) -> gpd.GeoDataFrame:
    """
    Fetch a single lake by name (case-insensitive partial match).

    Handles both real names ("Lake Champlain") and slugs ("lake-champlain")
    by replacing dashes/underscores with spaces for the search.
    """
    engine = get_engine()
    # Normalize slug-style names: "lake-champlain" -> "lake champlain"
    search_name = name.replace('-', ' ').replace('_', ' ')
    sql = f"""
        SELECT uuid, name, area_sqkm, geom
        FROM hydrology.lakes
        WHERE LOWER(name) LIKE LOWER('%{search_name}%')
        ORDER BY area_sqkm DESC
        LIMIT 1
    """
    conn = engine.raw_connection()
    try:
        gdf = gpd.read_postgis(sql, conn, geom_col='geom')
    finally:
        conn.close()
    return gdf


def get_lake_by_uuid(uuid: str) -> gpd.GeoDataFrame:
    """Fetch a single lake by UUID."""
    engine = get_engine()
    sql = f"""
        SELECT uuid, name, area_sqkm, geom
        FROM hydrology.lakes
        WHERE uuid = '{uuid}'
    """
    conn = engine.raw_connection()
    try:
        gdf = gpd.read_postgis(sql, conn, geom_col='geom')
    finally:
        conn.close()
    return gdf


def lake_row_to_config(row) -> dict:
    """
    Convert a GeoDataFrame row from the lakes table into a config dict
    compatible with the rest of the pipeline.
    """
    geom = row['geom']
    centroid = geom.centroid
    bounds = geom.bounds  # (minx, miny, maxx, maxy)
    area = row['area_sqkm']
    name = row['name']

    return {
        'uuid': str(row['uuid']),
        'name': name,
        'area_km2': float(area),
        'center': [round(centroid.x, 4), round(centroid.y, 4)],
        'bbox': [round(v, 4) for v in bounds],
        'avg_depth_m': round(estimate_depth(area, name), 1),
        'utm_epsg': utm_epsg_from_lonlat(centroid.x, centroid.y),
    }
