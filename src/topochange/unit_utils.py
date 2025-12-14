"""
unit_utils.py - Comprehensive unit parsing and conversion for geospatial data.

This module provides:
- UnitInfo dataclass to encapsulate unit metadata
- Unit registry with EPSG codes and common aliases
- Extraction of units from pyproj CRS objects
- Parsing of unit strings from catalogs and metadata
- Array-aware conversion functions

The design leverages pyproj's built-in unit_conversion_factor when available,
with fallback to a comprehensive lookup table for string parsing.

Key insight: PROJ handles unit conversion automatically during CRS transformations.
This module is needed for:
1. Extracting/displaying unit info to users
2. Manual conversion when working with raw values (e.g., point cloud Z before reprojection)
3. Parsing catalog metadata strings into structured data
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pyproj import CRS as _CRS


# =============================================================================
# UnitInfo Dataclass
# =============================================================================

@dataclass(frozen=True)
class UnitInfo:
    """
    Encapsulates information about a measurement unit.
    
    Think of this like a currency object that knows its exchange rate and metadata.
    
    Attributes
    ----------
    name : str
        Canonical normalized name (lowercase, underscores). 
        Examples: "meter", "us_survey_foot", "degree"
    display_name : str
        Human-readable name as typically shown to users.
        Examples: "metre", "US survey foot", "degree"
    abbreviation : str
        Short form for display. Examples: "m", "ft", "°"
    to_base_factor : float
        Factor to convert TO base unit (meters for linear, radians for angular).
        Multiply values in this unit by this factor to get base units.
    category : str
        One of: "linear", "angular", "scale", "time", "unknown"
    epsg_code : Optional[int]
        EPSG unit code if known (e.g., 9001 for metre)
    
    Examples
    --------
    >>> meter = UnitInfo("meter", "metre", "m", 1.0, "linear", 9001)
    >>> us_ft = UnitInfo("us_survey_foot", "US survey foot", "ft", 0.3048006096, "linear", 9003)
    >>> # Convert 100 US survey feet to meters:
    >>> 100 * us_ft.to_base_factor
    30.48006096
    """
    name: str
    display_name: str
    abbreviation: str
    to_base_factor: float
    category: str
    epsg_code: Optional[int] = None
    
    def convert_to(self, values: np.ndarray, target: "UnitInfo") -> np.ndarray:
        """
        Convert values from this unit to target unit.
        
        Parameters
        ----------
        values : np.ndarray
            Values in this unit
        target : UnitInfo
            Target unit to convert to
            
        Returns
        -------
        np.ndarray
            Values converted to target unit
            
        Raises
        ------
        ValueError
            If units have different categories (e.g., linear vs angular)
        """
        if self.category != target.category:
            raise ValueError(
                f"Cannot convert between {self.category} ({self.name}) "
                f"and {target.category} ({target.name})"
            )
        # Convert: source -> base -> target
        # Factor = source_to_base / target_to_base
        factor = self.to_base_factor / target.to_base_factor
        return np.asarray(values) * factor
    
    def __str__(self) -> str:
        return f"{self.display_name} ({self.abbreviation})"
    
    def __repr__(self) -> str:
        return f"UnitInfo({self.name!r}, factor={self.to_base_factor})"


# =============================================================================
# Unit Registry
# =============================================================================

# EPSG codes for common units (from EPSG registry)
# Linear: base = metre (factor = 1.0)
# Angular: base = radian (factor = 1.0)

_LINEAR_UNITS: Dict[str, UnitInfo] = {
    # Metre (EPSG:9001) - the base unit
    "metre": UnitInfo("meter", "metre", "m", 1.0, "linear", 9001),
    "meter": UnitInfo("meter", "metre", "m", 1.0, "linear", 9001),
    "m": UnitInfo("meter", "metre", "m", 1.0, "linear", 9001),
    "meters": UnitInfo("meter", "metre", "m", 1.0, "linear", 9001),
    "metres": UnitInfo("meter", "metre", "m", 1.0, "linear", 9001),
    
    # Kilometre (EPSG:9036)
    "kilometre": UnitInfo("kilometer", "kilometre", "km", 1000.0, "linear", 9036),
    "kilometer": UnitInfo("kilometer", "kilometre", "km", 1000.0, "linear", 9036),
    "km": UnitInfo("kilometer", "kilometre", "km", 1000.0, "linear", 9036),
    
    # Centimetre 
    "centimetre": UnitInfo("centimeter", "centimetre", "cm", 0.01, "linear", None),
    "centimeter": UnitInfo("centimeter", "centimetre", "cm", 0.01, "linear", None),
    "cm": UnitInfo("centimeter", "centimetre", "cm", 0.01, "linear", None),
    
    # Millimetre
    "millimetre": UnitInfo("millimeter", "millimetre", "mm", 0.001, "linear", None),
    "millimeter": UnitInfo("millimeter", "millimetre", "mm", 0.001, "linear", None),
    "mm": UnitInfo("millimeter", "millimetre", "mm", 0.001, "linear", None),
    
    # International foot (EPSG:9002) - exactly 0.3048 m
    "foot": UnitInfo("foot", "foot", "ft", 0.3048, "linear", 9002),
    "feet": UnitInfo("foot", "foot", "ft", 0.3048, "linear", 9002),
    "ft": UnitInfo("foot", "foot", "ft", 0.3048, "linear", 9002),
    "international foot": UnitInfo("foot", "foot", "ft", 0.3048, "linear", 9002),
    "international_foot": UnitInfo("foot", "foot", "ft", 0.3048, "linear", 9002),
    
    # US survey foot (EPSG:9003) - 1200/3937 m (slightly longer than international foot)
    "us survey foot": UnitInfo("us_survey_foot", "US survey foot", "ftUS", 1200.0/3937.0, "linear", 9003),
    "us_survey_foot": UnitInfo("us_survey_foot", "US survey foot", "ftUS", 1200.0/3937.0, "linear", 9003),
    "us-survey-foot": UnitInfo("us_survey_foot", "US survey foot", "ftUS", 1200.0/3937.0, "linear", 9003),
    "ussurveyfoot": UnitInfo("us_survey_foot", "US survey foot", "ftUS", 1200.0/3937.0, "linear", 9003),
    "ftus": UnitInfo("us_survey_foot", "US survey foot", "ftUS", 1200.0/3937.0, "linear", 9003),
    "us foot": UnitInfo("us_survey_foot", "US survey foot", "ftUS", 1200.0/3937.0, "linear", 9003),
    "us feet": UnitInfo("us_survey_foot", "US survey foot", "ftUS", 1200.0/3937.0, "linear", 9003),
    "survey foot": UnitInfo("us_survey_foot", "US survey foot", "ftUS", 1200.0/3937.0, "linear", 9003),
    "survey feet": UnitInfo("us_survey_foot", "US survey foot", "ftUS", 1200.0/3937.0, "linear", 9003),
    
    # British foot (EPSG:9070) - used in some older UK data
    "british foot": UnitInfo("british_foot", "British foot", "ft(Br)", 0.3047972654, "linear", 9070),
    "british_foot": UnitInfo("british_foot", "British foot", "ft(Br)", 0.3047972654, "linear", 9070),
    
    # Clarke's foot (EPSG:9005) - used in some African/Indian surveys
    "clarke's foot": UnitInfo("clarke_foot", "Clarke's foot", "ft(Cla)", 0.3047972654, "linear", 9005),
    "clarke_foot": UnitInfo("clarke_foot", "Clarke's foot", "ft(Cla)", 0.3047972654, "linear", 9005),
    
    # Inch
    "inch": UnitInfo("inch", "inch", "in", 0.0254, "linear", None),
    "inches": UnitInfo("inch", "inch", "in", 0.0254, "linear", None),
    "in": UnitInfo("inch", "inch", "in", 0.0254, "linear", None),
    
    # Yard
    "yard": UnitInfo("yard", "yard", "yd", 0.9144, "linear", None),
    "yards": UnitInfo("yard", "yard", "yd", 0.9144, "linear", None),
    "yd": UnitInfo("yard", "yard", "yd", 0.9144, "linear", None),
    
    # Mile
    "mile": UnitInfo("mile", "mile", "mi", 1609.344, "linear", None),
    "miles": UnitInfo("mile", "mile", "mi", 1609.344, "linear", None),
    "mi": UnitInfo("mile", "mile", "mi", 1609.344, "linear", None),
    
    # Nautical mile (EPSG:9030)
    "nautical mile": UnitInfo("nautical_mile", "nautical mile", "nmi", 1852.0, "linear", 9030),
    "nautical_mile": UnitInfo("nautical_mile", "nautical mile", "nmi", 1852.0, "linear", 9030),
    "nmi": UnitInfo("nautical_mile", "nautical mile", "nmi", 1852.0, "linear", 9030),
    
    # German legal metre (EPSG:9031)
    "german legal metre": UnitInfo("german_legal_metre", "German legal metre", "GLM", 1.0000135965, "linear", 9031),
}

# Angular units (base = radian)
import math
_ANGULAR_UNITS: Dict[str, UnitInfo] = {
    # Radian (EPSG:9101) - the base unit
    "radian": UnitInfo("radian", "radian", "rad", 1.0, "angular", 9101),
    "radians": UnitInfo("radian", "radian", "rad", 1.0, "angular", 9101),
    "rad": UnitInfo("radian", "radian", "rad", 1.0, "angular", 9101),
    
    # Degree (EPSG:9102) - π/180 radians
    "degree": UnitInfo("degree", "degree", "°", math.pi / 180.0, "angular", 9102),
    "degrees": UnitInfo("degree", "degree", "°", math.pi / 180.0, "angular", 9102),
    "deg": UnitInfo("degree", "degree", "°", math.pi / 180.0, "angular", 9102),
    "°": UnitInfo("degree", "degree", "°", math.pi / 180.0, "angular", 9102),
    
    # Arc-minute (EPSG:9103)
    "arc-minute": UnitInfo("arc_minute", "arc-minute", "'", math.pi / 10800.0, "angular", 9103),
    "arc_minute": UnitInfo("arc_minute", "arc-minute", "'", math.pi / 10800.0, "angular", 9103),
    "arcminute": UnitInfo("arc_minute", "arc-minute", "'", math.pi / 10800.0, "angular", 9103),
    "'": UnitInfo("arc_minute", "arc-minute", "'", math.pi / 10800.0, "angular", 9103),
    
    # Arc-second (EPSG:9104)
    "arc-second": UnitInfo("arc_second", "arc-second", '"', math.pi / 648000.0, "angular", 9104),
    "arc_second": UnitInfo("arc_second", "arc-second", '"', math.pi / 648000.0, "angular", 9104),
    "arcsecond": UnitInfo("arc_second", "arc-second", '"', math.pi / 648000.0, "angular", 9104),
    '"': UnitInfo("arc_second", "arc-second", '"', math.pi / 648000.0, "angular", 9104),
    "arcsec": UnitInfo("arc_second", "arc-second", '"', math.pi / 648000.0, "angular", 9104),
    
    # Grad/gon (EPSG:9105) - π/200 radians
    "grad": UnitInfo("grad", "grad", "gon", math.pi / 200.0, "angular", 9105),
    "gon": UnitInfo("grad", "grad", "gon", math.pi / 200.0, "angular", 9105),
    "grads": UnitInfo("grad", "grad", "gon", math.pi / 200.0, "angular", 9105),
}

# Combined registry
_ALL_UNITS = {**_LINEAR_UNITS, **_ANGULAR_UNITS}

# Canonical unit objects for common access
METER = _LINEAR_UNITS["meter"]
FOOT = _LINEAR_UNITS["foot"]
US_SURVEY_FOOT = _LINEAR_UNITS["us_survey_foot"]
KILOMETER = _LINEAR_UNITS["kilometer"]
DEGREE = _ANGULAR_UNITS["degree"]
RADIAN = _ANGULAR_UNITS["radian"]

# Unknown unit placeholder
UNKNOWN_UNIT = UnitInfo("unknown", "unknown", "?", 1.0, "unknown", None)


# =============================================================================
# Unit Lookup Functions
# =============================================================================

def lookup_unit(name: str) -> Optional[UnitInfo]:
    """
    Look up a unit by name, returning UnitInfo or None if not found.
    
    Parameters
    ----------
    name : str
        Unit name to look up (case-insensitive, various formats accepted)
        
    Returns
    -------
    UnitInfo or None
        The unit info if found, None otherwise
        
    Examples
    --------
    >>> lookup_unit("metre")
    UnitInfo('meter', factor=1.0)
    >>> lookup_unit("US survey foot")
    UnitInfo('us_survey_foot', factor=0.3048006096...)
    >>> lookup_unit("invalid")
    None
    """
    if not name:
        return None
    key = name.lower().strip()
    return _ALL_UNITS.get(key)


def lookup_unit_strict(name: str) -> UnitInfo:
    """
    Look up a unit by name, raising ValueError if not found.
    
    Parameters
    ----------
    name : str
        Unit name to look up
        
    Returns
    -------
    UnitInfo
        The unit info
        
    Raises
    ------
    ValueError
        If unit is not found in registry
    """
    unit = lookup_unit(name)
    if unit is None:
        raise ValueError(f"Unknown unit: {name!r}")
    return unit


def parse_unit_string(s: str) -> UnitInfo:
    """
    Parse a unit from a string that may contain extra information.
    
    This handles strings like those from OpenTopography catalog:
    - "(metre)" 
    - "(ftUS)"
    - "NAVD88 height - Geoid12B (metre)"
    - "meter"
    
    Parameters
    ----------
    s : str
        String that may contain unit information
        
    Returns
    -------
    UnitInfo
        Parsed unit, or UNKNOWN_UNIT if not parseable
        
    Examples
    --------
    >>> parse_unit_string("(metre)")
    UnitInfo('meter', factor=1.0)
    >>> parse_unit_string("NAVD88 height (ftUS)")
    UnitInfo('us_survey_foot', factor=0.3048006096...)
    """
    if not s:
        return UNKNOWN_UNIT
    
    s_lower = s.lower().strip()
    
    # First try direct lookup (for simple cases like "meter")
    direct = lookup_unit(s_lower)
    if direct is not None:
        return direct
    
    # Try to extract unit from parentheses: "(metre)", "(ftUS)", etc.
    paren_match = re.search(r'\(([^)]+)\)', s)
    if paren_match:
        paren_content = paren_match.group(1).strip()
        unit = lookup_unit(paren_content)
        if unit is not None:
            return unit
    
    # Try common patterns in the string
    patterns = [
        (r'\bft\s*us\b', "us_survey_foot"),
        (r'\bus\s*survey\s*foot\b', "us_survey_foot"),
        (r'\bus\s*survey\s*feet\b', "us_survey_foot"),
        (r'\bsurvey\s*foot\b', "us_survey_foot"),
        (r'\bsurvey\s*feet\b', "us_survey_foot"),
        (r'\bfeet\b', "foot"),
        (r'\bfoot\b', "foot"),
        (r'\bmetres?\b', "meter"),
        (r'\bmeters?\b', "meter"),
        (r'\bdegrees?\b', "degree"),
    ]
    
    for pattern, unit_name in patterns:
        if re.search(pattern, s_lower):
            unit = lookup_unit(unit_name)
            if unit is not None:
                return unit
    
    return UNKNOWN_UNIT


# =============================================================================
# CRS Unit Extraction
# =============================================================================

def _ensure_crs_obj(crs: Union[str, _CRS, Dict[str, Any]]) -> _CRS:
    """
    Accept WKT, PROJJSON (dict), proj string, EPSG code, or CRS object.
    Return a pyproj.CRS instance, raising on failure.
    """
    if isinstance(crs, _CRS):
        return crs
    if isinstance(crs, dict):
        return _CRS.from_json_dict(crs)
    return _CRS.from_user_input(crs)


def _unit_from_axis(axis) -> UnitInfo:
    """
    Extract UnitInfo from a pyproj Axis object.
    
    Uses both the unit_name for lookup and unit_conversion_factor as authoritative.
    """
    unit_name = axis.unit_name
    factor = axis.unit_conversion_factor
    
    # Determine category from factor magnitude
    # Angular units have factor < 0.1 (degrees = 0.0174...)
    # Linear units have factor >= 0.001
    if factor is None:
        return UNKNOWN_UNIT
    
    # Try to find matching unit in registry
    matched_unit = lookup_unit(unit_name) if unit_name else None
    
    if matched_unit is not None:
        # If registry factor is close to pyproj factor, use registry (has more metadata)
        if abs(matched_unit.to_base_factor - factor) < 1e-9:
            return matched_unit
        # Otherwise create new UnitInfo with pyproj's authoritative factor
        return UnitInfo(
            name=matched_unit.name,
            display_name=matched_unit.display_name,
            abbreviation=matched_unit.abbreviation,
            to_base_factor=factor,
            category=matched_unit.category,
            epsg_code=matched_unit.epsg_code,
        )
    
    # Determine category heuristically
    if factor < 0.1:
        category = "angular"
        abbrev = "?"
    else:
        category = "linear"
        abbrev = "?"
    
    # Create from pyproj info
    display_name = unit_name if unit_name else "unknown"
    return UnitInfo(
        name=display_name.lower().replace(" ", "_") if display_name else "unknown",
        display_name=display_name,
        abbreviation=abbrev,
        to_base_factor=factor,
        category=category,
        epsg_code=None,
    )


def get_horizontal_unit(crs: Any) -> UnitInfo:
    """
    Extract the horizontal unit from a CRS.
    
    For projected CRS (UTM, State Plane, etc.), returns the linear unit (metre, foot, etc.)
    For geographic CRS, returns the angular unit (degree).
    For compound CRS, extracts from the horizontal component.
    
    Parameters
    ----------
    crs : Any
        CRS in any format accepted by pyproj
        
    Returns
    -------
    UnitInfo
        The horizontal unit
        
    Examples
    --------
    >>> get_horizontal_unit("EPSG:32610")  # UTM
    UnitInfo('meter', factor=1.0)
    >>> get_horizontal_unit("EPSG:2230")  # State Plane (US feet)
    UnitInfo('us_survey_foot', factor=0.3048006096...)
    >>> get_horizontal_unit("EPSG:4326")  # Geographic
    UnitInfo('degree', factor=0.0174532...)
    """
    try:
        crs_obj = _ensure_crs_obj(crs)
    except Exception:
        return UNKNOWN_UNIT
    
    # For compound CRS, get the horizontal component
    if crs_obj.is_compound:
        sub_crs_list = getattr(crs_obj, 'sub_crs_list', None) or []
        if sub_crs_list:
            crs_obj = sub_crs_list[0]  # First is horizontal
    
    # Get first axis (should be horizontal for most CRS)
    if crs_obj.coordinate_system and crs_obj.coordinate_system.axis_list:
        axis = crs_obj.coordinate_system.axis_list[0]
        return _unit_from_axis(axis)
    
    return UNKNOWN_UNIT


def get_vertical_unit(crs: Any) -> UnitInfo:
    """
    Extract the vertical unit from a CRS.
    
    For compound CRS, extracts from the vertical component.
    For standalone vertical CRS, returns its unit.
    For 2D CRS, returns UNKNOWN_UNIT.
    
    Parameters
    ----------
    crs : Any
        CRS in any format accepted by pyproj
        
    Returns
    -------
    UnitInfo
        The vertical unit
        
    Examples
    --------
    >>> get_vertical_unit("EPSG:5703")  # NAVD88
    UnitInfo('meter', factor=1.0)
    >>> get_vertical_unit("EPSG:6349")  # NAD83(2011) + NAVD88
    UnitInfo('meter', factor=1.0)
    """
    try:
        crs_obj = _ensure_crs_obj(crs)
    except Exception:
        return UNKNOWN_UNIT
    
    # For compound CRS, get the vertical component
    if crs_obj.is_compound:
        sub_crs_list = getattr(crs_obj, 'sub_crs_list', None) or []
        if len(sub_crs_list) >= 2:
            vert_crs = sub_crs_list[1]  # Second is vertical
            if vert_crs.coordinate_system and vert_crs.coordinate_system.axis_list:
                return _unit_from_axis(vert_crs.coordinate_system.axis_list[0])
    
    # For standalone vertical CRS
    if getattr(crs_obj, 'is_vertical', False):
        if crs_obj.coordinate_system and crs_obj.coordinate_system.axis_list:
            return _unit_from_axis(crs_obj.coordinate_system.axis_list[0])
    
    # For 3D geographic CRS, the last axis is vertical (height)
    if crs_obj.coordinate_system and crs_obj.coordinate_system.axis_list:
        axis_list = crs_obj.coordinate_system.axis_list
        if len(axis_list) >= 3:
            # Third axis is typically height
            return _unit_from_axis(axis_list[2])
    
    return UNKNOWN_UNIT


def get_crs_units(crs: Any) -> Tuple[UnitInfo, Optional[UnitInfo]]:
    """
    Get both horizontal and vertical units from a CRS.
    
    Parameters
    ----------
    crs : Any
        CRS in any format accepted by pyproj
        
    Returns
    -------
    tuple[UnitInfo, Optional[UnitInfo]]
        (horizontal_unit, vertical_unit)
        vertical_unit is None if CRS has no vertical component
    """
    h_unit = get_horizontal_unit(crs)
    v_unit = get_vertical_unit(crs)
    
    # Return None for vertical if unknown/not present
    if v_unit.name == "unknown":
        return h_unit, None
    return h_unit, v_unit


# =============================================================================
# Unit Conversion Functions
# =============================================================================

def convert_length(
    values: Union[np.ndarray, float, List[float]],
    from_unit: Union[str, UnitInfo],
    to_unit: Union[str, UnitInfo],
) -> np.ndarray:
    """
    Convert linear values between units.
    
    Parameters
    ----------
    values : array-like
        Values to convert
    from_unit : str or UnitInfo
        Source unit
    to_unit : str or UnitInfo
        Target unit
        
    Returns
    -------
    np.ndarray
        Converted values
        
    Examples
    --------
    >>> convert_length([100, 200], "foot", "meter")
    array([30.48, 60.96])
    >>> convert_length(1000, METER, US_SURVEY_FOOT)
    array(3280.8333...)
    """
    # Resolve units
    if isinstance(from_unit, str):
        from_unit = lookup_unit_strict(from_unit)
    if isinstance(to_unit, str):
        to_unit = lookup_unit_strict(to_unit)
    
    values = np.asarray(values)
    return from_unit.convert_to(values, to_unit)


def convert_to_meters(
    values: Union[np.ndarray, float, List[float]],
    from_unit: Union[str, UnitInfo],
) -> np.ndarray:
    """
    Convert linear values to meters.
    
    Parameters
    ----------
    values : array-like
        Values to convert
    from_unit : str or UnitInfo
        Source unit
        
    Returns
    -------
    np.ndarray
        Values in meters
    """
    return convert_length(values, from_unit, METER)


def convert_from_meters(
    values: Union[np.ndarray, float, List[float]],
    to_unit: Union[str, UnitInfo],
) -> np.ndarray:
    """
    Convert values from meters to another unit.
    
    Parameters
    ----------
    values : array-like
        Values in meters
    to_unit : str or UnitInfo
        Target unit
        
    Returns
    -------
    np.ndarray
        Converted values
    """
    return convert_length(values, METER, to_unit)


def get_conversion_factor(from_unit: Union[str, UnitInfo], to_unit: Union[str, UnitInfo]) -> float:
    """
    Get the conversion factor between two units.
    
    Multiply values in from_unit by this factor to get values in to_unit.
    
    Parameters
    ----------
    from_unit : str or UnitInfo
        Source unit
    to_unit : str or UnitInfo
        Target unit
        
    Returns
    -------
    float
        Conversion factor
        
    Examples
    --------
    >>> get_conversion_factor("foot", "meter")
    0.3048
    >>> get_conversion_factor("meter", "foot")
    3.280839895...
    """
    if isinstance(from_unit, str):
        from_unit = lookup_unit_strict(from_unit)
    if isinstance(to_unit, str):
        to_unit = lookup_unit_strict(to_unit)
    
    return from_unit.to_base_factor / to_unit.to_base_factor


# =============================================================================
# Convenience Wrappers (for backward compatibility with existing crs_utils.py)
# =============================================================================

def horizontal_unit_scale(src_crs: Any, target_unit: str) -> Optional[float]:
    """
    Return scale factor to go from source horizontal units to target_unit.
    (e.g. metres -> feet => factor ~ 3.28084)
    
    This is a drop-in replacement for the existing function in crs_utils.py.
    
    Parameters
    ----------
    src_crs : Any
        Source CRS
    target_unit : str
        Target unit name
        
    Returns
    -------
    float or None
        Scale factor, or None if units are unknown
    """
    src_unit = get_horizontal_unit(src_crs)
    tgt_unit = lookup_unit(target_unit)
    
    if src_unit.name == "unknown" or tgt_unit is None:
        return None
    
    try:
        return get_conversion_factor(src_unit, tgt_unit)
    except ValueError:
        return None


def vertical_unit_scale(src_crs: Any, target_unit: str) -> Optional[float]:
    """
    Return scale factor to go from source vertical units to target_unit.
    
    This is a drop-in replacement for the existing function in crs_utils.py.
    
    Parameters
    ----------
    src_crs : Any
        Source CRS
    target_unit : str
        Target unit name
        
    Returns
    -------
    float or None
        Scale factor, or None if units are unknown
    """
    src_unit = get_vertical_unit(src_crs)
    tgt_unit = lookup_unit(target_unit)
    
    if src_unit.name == "unknown" or tgt_unit is None:
        return None
    
    try:
        return get_conversion_factor(src_unit, tgt_unit)
    except ValueError:
        return None


# =============================================================================
# PDAL Metadata Parsing
# =============================================================================

def parse_pdal_units(srs_metadata: Dict[str, Any]) -> Tuple[UnitInfo, UnitInfo]:
    """
    Parse horizontal and vertical units from PDAL SRS metadata.
    
    PDAL's readers.las provides unit info in:
    srs["units"]["horizontal"] and srs["units"]["vertical"]
    
    Parameters
    ----------
    srs_metadata : dict
        The 'srs' dict from PDAL pipeline metadata
        
    Returns
    -------
    tuple[UnitInfo, UnitInfo]
        (horizontal_unit, vertical_unit)
        
    Examples
    --------
    >>> srs_md = {"units": {"horizontal": "metre", "vertical": "US survey foot"}}
    >>> parse_pdal_units(srs_md)
    (UnitInfo('meter', ...), UnitInfo('us_survey_foot', ...))
    """
    units_dict = srs_metadata.get("units", {}) or {}
    
    h_name = units_dict.get("horizontal", "unknown")
    v_name = units_dict.get("vertical", "unknown")
    
    h_unit = parse_unit_string(h_name) if h_name else UNKNOWN_UNIT
    v_unit = parse_unit_string(v_name) if v_name else UNKNOWN_UNIT
    
    return h_unit, v_unit


# =============================================================================
# Catalog String Parsing
# =============================================================================

def parse_catalog_vertical_units(vertical_str: str) -> UnitInfo:
    """
    Parse vertical units from OpenTopography catalog strings.
    
    Handles strings like:
    - "NAVD88 (Geoid 12B)"
    - "NAVD88 height - Geoid12B (metre)"
    - "NAVD88 height - Geoid12B (ftUS)"
    
    Parameters
    ----------
    vertical_str : str
        Vertical coordinates string from catalog
        
    Returns
    -------
    UnitInfo
        Parsed unit (defaults to meter if not specified)
    """
    if not vertical_str:
        return UNKNOWN_UNIT
    
    # Use the general parser
    unit = parse_unit_string(vertical_str)
    
    # If unknown but looks like an orthometric datum, assume meters
    if unit.name == "unknown":
        v_lower = vertical_str.lower()
        if any(kw in v_lower for kw in ["navd88", "navd 88", "ngvd29", "egm96", "egm2008", "geoid"]):
            return METER
    
    return unit


# =============================================================================
# Display/Formatting Utilities
# =============================================================================

def format_value_with_unit(value: float, unit: UnitInfo, precision: int = 2) -> str:
    """
    Format a value with its unit for display.
    
    Parameters
    ----------
    value : float
        The value to format
    unit : UnitInfo
        The unit
    precision : int
        Decimal places
        
    Returns
    -------
    str
        Formatted string like "123.45 m" or "405.23 ft"
    """
    return f"{value:.{precision}f} {unit.abbreviation}"


def describe_unit(unit: UnitInfo) -> str:
    """
    Get a human-readable description of a unit.
    
    Parameters
    ----------
    unit : UnitInfo
        The unit to describe
        
    Returns
    -------
    str
        Description like "metre (m) - 1.0 meters"
    """
    if unit.category == "linear":
        base_desc = f"{unit.to_base_factor} meters"
    elif unit.category == "angular":
        base_desc = f"{unit.to_base_factor} radians"
    else:
        base_desc = f"factor {unit.to_base_factor}"
    
    epsg_str = f" [EPSG:{unit.epsg_code}]" if unit.epsg_code else ""
    return f"{unit.display_name} ({unit.abbreviation}) = {base_desc}{epsg_str}"