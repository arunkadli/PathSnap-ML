"""Coordinate projection helpers for PathSnap's longitude/latitude data."""

import math
from typing import Union

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


def utm_zone_for_location(longitude: float, latitude: float) -> int:
    """Return the WGS 84 UTM zone for a longitude/latitude position."""
    try:
        lon = float(longitude)
        lat = float(latitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("longitude and latitude must be finite numbers.") from exc

    if not math.isfinite(lon) or not math.isfinite(lat):
        raise ValueError("longitude and latitude must be finite numbers.")
    if not -180 <= lon <= 180 or not -80 <= lat <= 84:
        raise ValueError(
            "UTM requires longitude in [-180, 180] and latitude in [-80, 84]."
        )

    zone = min(60, max(1, int(math.floor((lon + 180) / 6)) + 1))
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    elif 72 <= lat <= 84:
        if 0 <= lon < 9:
            zone = 31
        elif 9 <= lon < 21:
            zone = 33
        elif 21 <= lon < 33:
            zone = 35
        elif 33 <= lon < 42:
            zone = 37
    return zone


def utm_crs_for_location(longitude: float, latitude: float) -> CRS:
    """Return the WGS 84 UTM CRS (EPSG:326xx north, EPSG:327xx south)."""
    zone = utm_zone_for_location(longitude, latitude)
    epsg = (32600 if float(latitude) >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def utm_crs_for_trip(track: pd.DataFrame) -> CRS:
    """Choose a trip-local UTM CRS from the median valid GPS position."""
    required = {"longitude", "latitude"}
    missing = required.difference(track.columns)
    if missing:
        raise ValueError(
            "track is missing coordinate column(s): " + ", ".join(sorted(missing))
        )

    longitude = pd.to_numeric(track["longitude"], errors="coerce").to_numpy()
    latitude = pd.to_numeric(track["latitude"], errors="coerce").to_numpy()
    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= -180)
        & (longitude <= 180)
        & (latitude >= -80)
        & (latitude <= 84)
    )
    if not valid.any():
        raise ValueError("track has no valid coordinates within the UTM latitude range.")

    return utm_crs_for_location(
        float(np.median(longitude[valid])),
        float(np.median(latitude[valid])),
    )


def project_coordinates(
    frame: pd.DataFrame,
    crs: Union[CRS, int, str],
    longitude_column: str = "longitude",
    latitude_column: str = "latitude",
) -> pd.DataFrame:
    """Return a copy with projected ``x`` and ``y`` columns; invalid rows become NaN."""
    missing = {longitude_column, latitude_column}.difference(frame.columns)
    if missing:
        raise ValueError(
            "frame is missing coordinate column(s): " + ", ".join(sorted(missing))
        )

    target_crs = CRS.from_user_input(crs)
    if not target_crs.is_projected:
        raise ValueError("crs must be a projected coordinate reference system.")

    longitude = pd.to_numeric(frame[longitude_column], errors="coerce").to_numpy()
    latitude = pd.to_numeric(frame[latitude_column], errors="coerce").to_numpy()
    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= -180)
        & (longitude <= 180)
        & (latitude >= -90)
        & (latitude <= 90)
    )
    x = np.full(len(frame), np.nan, dtype=float)
    y = np.full(len(frame), np.nan, dtype=float)
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    if valid.any():
        x[valid], y[valid] = transformer.transform(longitude[valid], latitude[valid])

    projected = frame.copy()
    projected["x"] = x
    projected["y"] = y
    return projected
