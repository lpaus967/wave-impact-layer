"""
Wave physics calculations for inland lakes.

Primary formula: Young & Verhagen (1996) — finite-depth, fetch-limited wave growth.
Calibrated on Lake George, Australia, and widely validated for inland water bodies.

Reference:
    Young, I.R. & Verhagen, L.A. (1996). "The growth of fetch limited waves
    in water of finite depth." Coastal Engineering, 29(1-2), 47-78.
"""

import numpy as np

G = 9.81  # gravitational acceleration (m/s²)

# Wave intensity thresholds (significant wave height in meters)
WAVE_THRESHOLDS = {
    'calm': 0.05,       # < 5cm
    'light': 0.15,      # 5-15cm
    'moderate': 0.30,   # 15-30cm
    'rough': 0.50,      # 30-50cm
    'very_rough': 1.0   # > 50cm
}


def wave_height_young_verhagen(wind_speed_ms, fetch_m, depth_m=20.0):
    """
    Significant wave height using Young & Verhagen (1996).

    Handles deep water, shallow water, and the transition naturally via
    the tanh terms. No separate depth clamp is needed.

    Args:
        wind_speed_ms: Wind speed at 10m height (m/s). Scalar or array.
        fetch_m: Fetch distance in meters. Scalar or array.
        depth_m: Water depth in meters. Scalar or array.

    Returns:
        Significant wave height Hs in meters.
    """
    # Handle zero/negative inputs
    wind_speed_ms = np.asarray(wind_speed_ms, dtype=np.float64)
    fetch_m = np.asarray(fetch_m, dtype=np.float64)
    depth_m = np.asarray(depth_m, dtype=np.float64)

    # Avoid division by zero
    U = np.maximum(wind_speed_ms, 0.01)

    d_star = G * depth_m / U**2       # dimensionless depth
    F_star = G * fetch_m / U**2       # dimensionless fetch

    A_H = 0.493 * d_star**0.75
    B_H = 0.00313 * F_star**0.57

    # Prevent tanh(0)/tanh(0) issues
    A_H = np.maximum(A_H, 1e-10)

    H_star = 0.241 * (np.tanh(A_H) * np.tanh(B_H / np.tanh(A_H)))**0.87
    Hs = H_star * U**2 / G

    # Zero out where there's no fetch or no wind
    Hs = np.where((fetch_m > 0) & (wind_speed_ms > 0), Hs, 0.0)

    return float(Hs) if Hs.ndim == 0 else Hs


def wave_period_young_verhagen(wind_speed_ms, fetch_m, depth_m=20.0):
    """
    Peak wave period using Young & Verhagen (1996).

    Args:
        wind_speed_ms: Wind speed at 10m height (m/s).
        fetch_m: Fetch distance in meters.
        depth_m: Water depth in meters.

    Returns:
        Peak wave period Tp in seconds.
    """
    wind_speed_ms = np.asarray(wind_speed_ms, dtype=np.float64)
    fetch_m = np.asarray(fetch_m, dtype=np.float64)
    depth_m = np.asarray(depth_m, dtype=np.float64)

    U = np.maximum(wind_speed_ms, 0.01)

    d_star = G * depth_m / U**2
    F_star = G * fetch_m / U**2

    A_T = 0.331 * d_star**1.01
    B_T = 0.0005215 * F_star**0.73

    A_T = np.maximum(A_T, 1e-10)

    T_star = 7.519 * (np.tanh(A_T) * np.tanh(B_T / np.tanh(A_T)))**0.387
    Tp = T_star * U / G

    Tp = np.where((fetch_m > 0) & (wind_speed_ms > 0), Tp, 0.0)

    return float(Tp) if Tp.ndim == 0 else Tp


def classify_wave_intensity(wave_height):
    """Classify significant wave height into intensity category."""
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


def effective_fetch(fetch_rasters, wind_direction, n_radials=9, radial_spacing=3.0):
    """
    Compute SPM effective fetch using cosine-weighted radials.

    Instead of using a single fetch ray for the wind direction, averages
    fetch across multiple radials spanning +/- (n_radials//2 * radial_spacing)
    degrees around the wind direction, weighted by cos(offset_angle).

    Based on: USACE Shore Protection Manual (1984), Saville (1954).

    Args:
        fetch_rasters: Dict mapping direction (float degrees) -> rasterio dataset
        wind_direction: Wind direction in degrees (FROM convention)
        n_radials: Number of radials (default 9, centered on wind direction)
        radial_spacing: Degrees between radials (default 3.0)

    Returns:
        2D numpy array of effective fetch in meters
    """
    directions = sorted(fetch_rasters.keys())
    half_spread = (n_radials - 1) / 2.0

    weighted_sum = None
    weight_sum = 0.0

    for i in range(n_radials):
        offset = (i - half_spread) * radial_spacing
        query_dir = (wind_direction + offset) % 360
        weight = np.cos(np.radians(offset))

        # Interpolate fetch for this radial from the pre-computed rasters
        fetch = _interpolate_fetch_array(fetch_rasters, directions, query_dir)

        if weighted_sum is None:
            weighted_sum = fetch * weight
        else:
            weighted_sum += fetch * weight
        weight_sum += weight

    return weighted_sum / weight_sum


def _interpolate_fetch_array(fetch_rasters, directions, query_dir):
    """Linearly interpolate a fetch array for an arbitrary direction."""
    query_dir = query_dir % 360

    lower_dir = max([d for d in directions if d <= query_dir], default=directions[-1])
    upper_dir = min([d for d in directions if d > query_dir], default=directions[0])

    if upper_dir < lower_dir:
        upper_dir += 360

    if upper_dir == lower_dir or (upper_dir - lower_dir) > 180:
        return fetch_rasters[lower_dir % 360].read(1).astype(np.float64)

    weight = (query_dir - lower_dir) / (upper_dir - lower_dir)

    lower = fetch_rasters[lower_dir % 360].read(1).astype(np.float64)
    upper = fetch_rasters[upper_dir % 360].read(1).astype(np.float64)

    return lower * (1 - weight) + upper * weight
