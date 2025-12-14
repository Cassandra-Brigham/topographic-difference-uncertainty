"""Time and epoch conversion utilities.

Provides functions for converting between datetime formats, decimal years,
GPS time, and parsing epoch strings from various formats.
"""
import datetime
import re
from typing import Tuple, Union


def _datetime_to_decimal_year(dt: datetime.datetime) -> float:
    """
    Convert a datetime to a decimal year.

    Uses the fraction of the year represented by the elapsed seconds between
    Jan 1 and dt, divided by the total seconds in the year.
    """
    y0 = datetime.datetime(dt.year, 1, 1)
    y1 = datetime.datetime(dt.year + 1, 1, 1)
    return dt.year + (dt - y0).total_seconds() / (y1 - y0).total_seconds()


def _parse_epoch_string_to_decimal(
    s: str,
) -> Union[float, Tuple[float, float]]:
    """
    Parse a string epoch specification into decimal years.

    Examples:
      "2006-04-06"                    -> 2006.26...
      "04/06/2006"                    -> 2006.26...  (assumed MM/DD/YYYY)
      "04/06/2006 - 05/01/2006"       -> (2006.26..., 2006.33...)
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty epoch string.")

    def _parse_date(text: str) -> datetime.datetime:
        text = text.strip()
        # Try several common formats
        fmts = [
            "%m/%d/%Y",   # 04/06/2006 (assumed MM/DD/YYYY)
            "%Y-%m-%d",   # 2006-04-06
            "%d/%m/%Y",   # 06/04/2006 (DD/MM/YYYY) — fallback
            "%Y%m%d",     # 20060406
        ]
        last_err = None
        for fmt in fmts:
            try:
                return datetime.datetime.strptime(text, fmt)
            except ValueError as e:
                last_err = e
        raise ValueError(f"Unable to parse date '{text}' in epoch string: {last_err}")

    # Range "start - end"
    # We split on '-' with optional spaces around it; dates themselves use '/'
    parts = re.split(r"\s*-\s*", s)
    if len(parts) == 2:
        start_dt = _parse_date(parts[0])
        end_dt = _parse_date(parts[1])
        start_dec = _datetime_to_decimal_year(start_dt)
        end_dec = _datetime_to_decimal_year(end_dt)
        return (min(start_dec, end_dec), max(start_dec, end_dec))

    # Single date string
    dt = _parse_date(s)
    return _datetime_to_decimal_year(dt)


# ---- GPS time helpers ----

GPS_EPOCH = datetime.datetime(1980, 1, 6, 0, 0, 0)

# Leap-second table for UTC alignment (through 2017-01-01 = 18 s)
LEAP_SECONDS = [
    (datetime.datetime(1981, 7, 1), 1),  (datetime.datetime(1982, 7, 1), 2),
    (datetime.datetime(1983, 7, 1), 3),  (datetime.datetime(1985, 7, 1), 4),
    (datetime.datetime(1988, 1, 1), 5),  (datetime.datetime(1990, 1, 1), 6),
    (datetime.datetime(1991, 1, 1), 7),  (datetime.datetime(1992, 7, 1), 8),
    (datetime.datetime(1993, 7, 1), 9),  (datetime.datetime(1994, 7, 1), 10),
    (datetime.datetime(1996, 1, 1), 11), (datetime.datetime(1997, 7, 1), 12),
    (datetime.datetime(1999, 1, 1), 13), (datetime.datetime(2006, 1, 1), 14),
    (datetime.datetime(2009, 1, 1), 15), (datetime.datetime(2012, 7, 1), 16),
    (datetime.datetime(2015, 7, 1), 17), (datetime.datetime(2017, 1, 1), 18),
]


def _gps_leap_seconds(dt_gps_naive: datetime.datetime) -> int:
    """
    Return the number of leap seconds accumulated by the given GPS datetime.
    """
    ls = 0
    for when, n in LEAP_SECONDS:
        if dt_gps_naive >= when:
            ls = n
        else:
            break
    return ls


def gps_seconds_to_decimal_year_utc(gps_seconds: float) -> float:
    """
    Absolute GPS seconds (since 1980-01-06) -> UTC datetime (leap seconds subtracted) -> decimal year.
    """
    dt_gps = GPS_EPOCH + datetime.timedelta(seconds=gps_seconds)
    utc_dt = dt_gps - datetime.timedelta(seconds=_gps_leap_seconds(dt_gps))
    return _datetime_to_decimal_year(utc_dt)


# --- PDAL gpstimeconvert integration helpers ---


def _guess_in_time_from_stats(vmin: float, vmean: float, vmax: float) -> str:
    """
    Heuristic if metadata doesn't tell us the time flavor.
    - 'gws' if magnitudes look like week/day seconds (small, < ~1e6)
    - 'gst' if "adjusted" around +/- 1e9 is suspected
    - 'gt'  if ~1e9..2e9 (absolute GPS seconds)
    """
    m = max(abs(vmin), abs(vmean), abs(vmax))
    if m < 900000:                  # < ~10 days -> almost certainly week/day seconds
        return "gws"
    if -1.5e9 < vmean < 5e8:        # adjusted "standard" (minus 1e9) often lands here
        return "gst"
    if 1.0e9 < vmean < 2.5e9:       # typical absolute GPS seconds
        return "gt"
    return "gt"