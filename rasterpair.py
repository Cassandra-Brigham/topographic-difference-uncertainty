# rasterpair.py
"""
RasterPair class for comparing, transforming, and differencing two rasters.

This module provides tools for:
- Comparing CRS, epoch, geoid, and grid parameters between two rasters
- Transforming raster1 to match raster2's reference frame
- Computing elevation differences with proper metadata tracking

The transformation pipeline follows the order:
1. Dynamic epoch transformation (if epochs differ)
2. Horizontal CRS reprojection (if horizontal CRS differs)
3. Vertical datum transformation (if vertical kind or geoid differs)
4. Grid alignment (resample to match exact pixel grid)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from pyproj import CRS as _CRS
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

# Handle both package and standalone imports
try:
    from .raster import Raster
    from .crs_utils import _ensure_crs_obj, parse_crs_components
    from .crs_history import CRSHistory
    from .unit_utils import (
        UnitInfo,
        UNKNOWN_UNIT,
        METER,
        lookup_unit,
        get_conversion_factor,
        format_value_with_unit,
    )
    _UNIT_UTILS_AVAILABLE = True
except ImportError:
    try:
        from raster import Raster
        from crs_utils import _ensure_crs_obj, parse_crs_components
        from crs_history import CRSHistory
        from unit_utils import (
            UnitInfo,
            UNKNOWN_UNIT,
            METER,
            lookup_unit,
            get_conversion_factor,
            format_value_with_unit,
        )
        _UNIT_UTILS_AVAILABLE = True
    except ImportError:
        from raster import Raster
        from crs_utils import _ensure_crs_obj, parse_crs_components
        from crs_history import CRSHistory
        _UNIT_UTILS_AVAILABLE = False
        UnitInfo = None
        UNKNOWN_UNIT = None
        METER = None




# =============================================================================
# Utility Functions
# =============================================================================

def _get_epsg_or_wkt(crs_input: Any) -> Tuple[Optional[int], Optional[str]]:
    """
    Extract EPSG code and/or WKT from a CRS input.
    
    Returns (epsg_code, wkt_string) where either may be None.
    """
    if crs_input is None:
        return None, None
    
    try:
        crs_obj = _ensure_crs_obj(crs_input)
        epsg = crs_obj.to_epsg()
        wkt = crs_obj.to_wkt()
        return epsg, wkt
    except Exception:
        return None, str(crs_input) if crs_input else None


def _crs_equivalent(crs1: Any, crs2: Any, tolerance: float = 1e-6) -> bool:
    """
    Check if two CRS are equivalent.
    
    Uses EPSG comparison first, then falls back to WKT comparison.
    
    Parameters
    ----------
    crs1, crs2 : Any
        CRS inputs to compare
    tolerance : float
        Not currently used, but reserved for coordinate tolerance checks
        
    Returns
    -------
    bool
        True if CRS are equivalent
    """
    if crs1 is None and crs2 is None:
        return True
    if crs1 is None or crs2 is None:
        return False
    
    try:
        obj1 = _ensure_crs_obj(crs1)
        obj2 = _ensure_crs_obj(crs2)
        
        # Try EPSG comparison first (fastest)
        epsg1 = obj1.to_epsg()
        epsg2 = obj2.to_epsg()
        if epsg1 is not None and epsg2 is not None:
            return epsg1 == epsg2
        
        # Fall back to equals method
        return obj1.equals(obj2)
    except Exception:
        return str(crs1) == str(crs2)


def _geoid_equivalent(geoid1: Optional[str], geoid2: Optional[str]) -> bool:
    """
    Check if two geoid model names are equivalent.
    
    Handles case-insensitive comparison and common aliases.
    """
    if geoid1 is None and geoid2 is None:
        return True
    if geoid1 is None or geoid2 is None:
        return False
    
    # Normalize: lowercase, strip whitespace, remove common prefixes
    def normalize(g):
        g = g.lower().strip()
        # Remove common prefixes
        for prefix in ['us_noaa_', 'noaa_', 'ngs_', 'egm', 'geoid']:
            if g.startswith(prefix):
                g = g[len(prefix):]
        # Remove file extensions
        for ext in ['.tif', '.tiff', '.gtx', '.bin']:
            if g.endswith(ext):
                g = g[:-len(ext)]
        return g
    
    return normalize(geoid1) == normalize(geoid2)


def _units_equivalent(unit1: Any, unit2: Any) -> Tuple[bool, Optional[float]]:
    """
    Check if two units are equivalent and compute conversion factor if not.
    
    Parameters
    ----------
    unit1, unit2 : UnitInfo, str, or None
        Units to compare. Can be UnitInfo objects or unit name strings.
        
    Returns
    -------
    tuple[bool, float or None]
        (are_equivalent, conversion_factor)
        - are_equivalent: True if units match
        - conversion_factor: Factor to multiply unit1 values by to get unit2 values,
          or None if units match or cannot be determined
    """
    # Handle None cases
    if unit1 is None and unit2 is None:
        return True, None
    if unit1 is None or unit2 is None:
        return False, None
    
    # Get UnitInfo objects if available
    if _UNIT_UTILS_AVAILABLE:
        # Convert strings to UnitInfo if needed
        if isinstance(unit1, str):
            unit1_info = lookup_unit(unit1)
        elif hasattr(unit1, 'name'):  # UnitInfo-like object
            unit1_info = unit1
        else:
            unit1_info = None
            
        if isinstance(unit2, str):
            unit2_info = lookup_unit(unit2)
        elif hasattr(unit2, 'name'):  # UnitInfo-like object
            unit2_info = unit2
        else:
            unit2_info = None
        
        # Compare using UnitInfo
        if unit1_info is not None and unit2_info is not None:
            # Handle unknown units properly
            unit1_unknown = unit1_info.name == "unknown"
            unit2_unknown = unit2_info.name == "unknown"
            
            if unit1_unknown and unit2_unknown:
                # Both unknown - can't determine, assume match
                return True, None
            elif unit1_unknown or unit2_unknown:
                # One is unknown, other is known - DON'T assume match
                # This flags a potential mismatch that needs user attention
                return False, None
            
            # Check if names match
            if unit1_info.name == unit2_info.name:
                return True, None
            
            # Different units - get conversion factor
            try:
                factor = get_conversion_factor(unit1_info, unit2_info)
                return False, factor
            except Exception:
                return False, None
    
    # Fallback to string comparison
    str1 = str(unit1).lower().strip() if unit1 else "meter"
    str2 = str(unit2).lower().strip() if unit2 else "meter"
    
    # Normalize common variations
    def normalize_unit_str(s):
        s = s.lower().replace(' ', '_').replace('-', '_')
        aliases = {
            'm': 'meter', 'meters': 'meter', 'metre': 'meter', 'metres': 'meter',
            'ft': 'foot', 'feet': 'foot',
            'ftus': 'us_survey_foot', 'us_ft': 'us_survey_foot', 
            'usft': 'us_survey_foot', 'us_survey_feet': 'us_survey_foot',
        }
        return aliases.get(s, s)
    
    norm1 = normalize_unit_str(str1)
    norm2 = normalize_unit_str(str2)
    
    if norm1 == norm2:
        return True, None
    
    # Known conversion factors (unit1 * factor = unit2 in meters, then divide)
    # This is a simplified fallback
    to_meters = {
        'meter': 1.0,
        'foot': 0.3048,
        'us_survey_foot': 1200.0/3937.0,  # Exact definition
    }
    
    if norm1 in to_meters and norm2 in to_meters:
        factor = to_meters[norm1] / to_meters[norm2]
        return False, factor
    
    return False, None


def _get_extent_polygon(raster: Raster, valid_data_only: bool = True, simplify_tolerance: float = None):
    """
    Get the extent of a raster as a shapely Polygon or MultiPolygon.
    
    Parameters
    ----------
    raster : Raster
        The raster to get extent from
    valid_data_only : bool, default True
        If True, return polygon(s) covering only valid (non-nodata) pixels.
        If False, return the simple bounding box.
    simplify_tolerance : float, optional
        If provided, simplify the polygon geometry to reduce vertex count.
        Units are in the raster's CRS (e.g., meters for projected CRS).
        Useful for large rasters where pixel-level precision isn't needed.
        
    Returns
    -------
    shapely.geometry.Polygon or MultiPolygon
        The extent polygon in the raster's CRS. May be MultiPolygon if there
        are disconnected valid data regions or internal holes.
    """
    try:
        from shapely.geometry import box, shape, MultiPolygon
        from shapely.ops import unary_union
    except ImportError:
        raise ImportError("shapely is required for extent polygon operations")
    
    if not valid_data_only:
        # Simple bounding box
        bounds = raster.bounds
        if bounds is None:
            with rasterio.open(raster.filename) as src:
                bounds = src.bounds
        return box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    # Get valid data mask and vectorize it
    try:
        from rasterio.features import shapes as rio_shapes
    except ImportError:
        raise ImportError("rasterio.features is required for valid data extent")
    
    with rasterio.open(raster.filename) as src:
        data = src.read(1)
        nodata = src.nodata
        transform = src.transform
        
        # Create valid data mask (1 = valid, 0 = invalid)
        valid_mask = np.ones(data.shape, dtype=np.uint8)
        
        # Mask nodata values
        if nodata is not None:
            if np.isnan(nodata):
                valid_mask[np.isnan(data)] = 0
            else:
                valid_mask[data == nodata] = 0
        
        # Also mask NaN and Inf values
        valid_mask[np.isnan(data)] = 0
        valid_mask[np.isinf(data)] = 0
        
        # Check if there's any valid data
        if not np.any(valid_mask):
            return None
        
        # Vectorize the valid mask
        # shapes() yields (geometry_dict, value) pairs
        polygons = []
        for geom_dict, value in rio_shapes(valid_mask, mask=(valid_mask == 1), transform=transform):
            if value == 1:
                polygons.append(shape(geom_dict))
        
        if not polygons:
            return None
        
        # Union all polygons (handles overlaps and creates proper MultiPolygon if needed)
        result = unary_union(polygons)
        
        # Simplify if requested (useful for large rasters)
        if simplify_tolerance is not None and simplify_tolerance > 0:
            result = result.simplify(simplify_tolerance, preserve_topology=True)
        
        return result


def _get_bounding_box_polygon(raster: Raster):
    """
    Get the simple bounding box of a raster (ignoring nodata).
    
    This is faster than _get_extent_polygon with valid_data_only=True
    when you just need the rectangular extent.
    
    Parameters
    ----------
    raster : Raster
        The raster to get bounds from
        
    Returns
    -------
    shapely.geometry.Polygon
        The bounding box polygon
    """
    try:
        from shapely.geometry import box
    except ImportError:
        raise ImportError("shapely is required for extent polygon operations")
    
    bounds = raster.bounds
    if bounds is None:
        with rasterio.open(raster.filename) as src:
            bounds = src.bounds
    
    return box(bounds.left, bounds.bottom, bounds.right, bounds.top)


def _compute_overlap_polygon(
    raster1: Raster, 
    raster2: Raster, 
    valid_data_only: bool = True,
    simplify_tolerance: float = None,
):
    """
    Compute the intersection polygon of two rasters.
    
    Both rasters must be in the same CRS for meaningful results.
    
    Parameters
    ----------
    raster1, raster2 : Raster
        The rasters to compute overlap for
    valid_data_only : bool, default True
        If True, compute overlap of valid data regions only.
        If False, use simple bounding box intersection.
    simplify_tolerance : float, optional
        Simplification tolerance for the polygons (in CRS units)
        
    Returns
    -------
    shapely.geometry.Polygon, MultiPolygon, or None
        The intersection polygon, or None if no overlap
    """
    poly1 = _get_extent_polygon(raster1, valid_data_only=valid_data_only, 
                                 simplify_tolerance=simplify_tolerance)
    poly2 = _get_extent_polygon(raster2, valid_data_only=valid_data_only,
                                 simplify_tolerance=simplify_tolerance)
    
    if poly1 is None or poly2 is None:
        return None
    
    intersection = poly1.intersection(poly2)
    
    if intersection.is_empty:
        return None
    
    return intersection


def _create_valid_data_mask(data: np.ndarray, nodata: Any) -> np.ndarray:
    """
    Create a boolean mask where True = valid data, False = invalid.
    
    Handles various nodata representations including explicit nodata values,
    NaN, and Inf values.
    
    Parameters
    ----------
    data : np.ndarray
        The raster data array
    nodata : Any
        The nodata value (can be None, NaN, or a numeric value)
        
    Returns
    -------
    np.ndarray
        Boolean mask (True = valid, False = invalid)
    """
    # Start with all valid
    valid = np.ones(data.shape, dtype=bool)
    
    # Mask explicit nodata value
    if nodata is not None:
        if np.isnan(nodata) if isinstance(nodata, float) else False:
            # nodata is NaN - will be handled below
            pass
        else:
            valid &= (data != nodata)
    
    # Always mask NaN and Inf
    valid &= ~np.isnan(data)
    valid &= ~np.isinf(data)
    
    return valid


# =============================================================================
# RasterPair Class
# =============================================================================

@dataclass
class RasterPair:
    """
    Pair of rasters with comparison, transformation, and differencing tools.
    
    This class facilitates DEM differencing workflows by providing methods to:
    - Compare CRS, epoch, geoid, and grid parameters
    - Transform raster1 to match raster2's reference frame
    - Compute elevation differences with proper error handling
    
    The transformation follows the conceptual order (think of it like a 
    coordinate system "pipeline"):
    1. Time adjustment (dynamic epoch transformation)
    2. Horizontal reference (CRS reprojection)  
    3. Vertical reference (datum/geoid transformation)
    4. Grid alignment (pixel-perfect resampling)
    
    Attributes
    ----------
    raster1 : Raster
        First raster (typically the one to be transformed)
    raster2 : Raster
        Second raster (typically the reference)
    """
    
    raster1: Raster
    raster2: Raster

    # Internal state
    _transformation_history: List[Dict[str, Any]] = field(default_factory=list)
    _raster1_transformed: Optional[Raster] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize internal state."""
        self._transformation_history = []
        self._raster1_transformed = None

    # =========================================================================
    # CRS/Epoch/Geoid Comparison Methods
    # =========================================================================
    
    def check_horizontal_crs_match(self) -> Dict[str, Any]:
        """
        Check if horizontal CRS of both rasters match.
        
        Returns
        -------
        dict
            {
                'match': bool,
                'raster1_crs': {'epsg': int|None, 'name': str},
                'raster2_crs': {'epsg': int|None, 'name': str},
                'details': str
            }
        """
        crs1 = (getattr(self.raster1, 'current_horizontal_crs', None) or 
                getattr(self.raster1, 'original_horizontal_crs', None) or
                getattr(self.raster1, 'crs', None))
        crs2 = (getattr(self.raster2, 'current_horizontal_crs', None) or 
                getattr(self.raster2, 'original_horizontal_crs', None) or
                getattr(self.raster2, 'crs', None))
        
        epsg1, wkt1 = _get_epsg_or_wkt(crs1)
        epsg2, wkt2 = _get_epsg_or_wkt(crs2)
        
        match = _crs_equivalent(crs1, crs2)
        
        def get_name(crs):
            if crs is None:
                return "None"
            try:
                return _ensure_crs_obj(crs).name
            except Exception:
                return str(crs)[:50]
        
        return {
            'match': match,
            'raster1_crs': {'epsg': epsg1, 'name': get_name(crs1)},
            'raster2_crs': {'epsg': epsg2, 'name': get_name(crs2)},
            'details': "Horizontal CRS match" if match else "Horizontal CRS differ - transformation required"
        }
    
    def check_vertical_crs_match(self) -> Dict[str, Any]:
        """
        Check if vertical CRS/datum type of both rasters match.
        
        Returns
        -------
        dict
            {
                'match': bool,
                'raster1_vertical': {'crs': str|None, 'is_orthometric': bool|None},
                'raster2_vertical': {'crs': str|None, 'is_orthometric': bool|None},
                'details': str
            }
        """
        vcrs1 = (getattr(self.raster1, 'current_vertical_crs', None) or 
                 getattr(self.raster1, 'original_vertical_crs', None))
        vcrs2 = (getattr(self.raster2, 'current_vertical_crs', None) or 
                 getattr(self.raster2, 'original_vertical_crs', None))
        
        ortho1 = getattr(self.raster1, 'is_orthometric', None)
        ortho2 = getattr(self.raster2, 'is_orthometric', None)
        
        # CRS match check
        crs_match = _crs_equivalent(vcrs1, vcrs2)
        
        # Type match check (orthometric vs ellipsoidal)
        type_match = ortho1 == ortho2 if (ortho1 is not None and ortho2 is not None) else None
        
        overall_match = crs_match and (type_match is None or type_match)
        
        def get_vcrs_name(vcrs):
            if vcrs is None:
                return "None"
            try:
                return _ensure_crs_obj(vcrs).name
            except Exception:
                return str(vcrs)[:50]
        
        details = []
        if not crs_match:
            details.append("Vertical CRS differ")
        if type_match is False:
            details.append(f"Vertical type differ (ortho1={ortho1}, ortho2={ortho2})")
        if not details:
            details.append("Vertical CRS match")
        
        return {
            'match': overall_match,
            'raster1_vertical': {'crs': get_vcrs_name(vcrs1), 'is_orthometric': ortho1},
            'raster2_vertical': {'crs': get_vcrs_name(vcrs2), 'is_orthometric': ortho2},
            'details': " | ".join(details)
        }
    
    def check_compound_crs_match(self) -> Dict[str, Any]:
        """
        Check if compound CRS (horizontal + vertical) of both rasters match.
        
        Returns
        -------
        dict
            {
                'match': bool,
                'horizontal_match': bool,
                'vertical_match': bool,
                'raster1_compound': str|None,
                'raster2_compound': str|None,
                'details': str
            }
        """
        horiz_result = self.check_horizontal_crs_match()
        vert_result = self.check_vertical_crs_match()
        
        ccrs1 = (getattr(self.raster1, 'current_compound_crs', None) or 
                 getattr(self.raster1, 'original_compound_crs', None))
        ccrs2 = (getattr(self.raster2, 'current_compound_crs', None) or 
                 getattr(self.raster2, 'original_compound_crs', None))
        
        def get_ccrs_name(ccrs):
            if ccrs is None:
                return "None"
            try:
                return _ensure_crs_obj(ccrs).name
            except Exception:
                return str(ccrs)[:80]
        
        overall_match = horiz_result['match'] and vert_result['match']
        
        return {
            'match': overall_match,
            'horizontal_match': horiz_result['match'],
            'vertical_match': vert_result['match'],
            'raster1_compound': get_ccrs_name(ccrs1),
            'raster2_compound': get_ccrs_name(ccrs2),
            'details': f"Horizontal: {horiz_result['details']} | Vertical: {vert_result['details']}"
        }
    
    def check_geoid_match(self) -> Dict[str, Any]:
        """
        Check if geoid models of both rasters match.
        
        Returns
        -------
        dict
            {
                'match': bool,
                'raster1_geoid': str|None,
                'raster2_geoid': str|None,
                'details': str
            }
        """
        geoid1 = (getattr(self.raster1, 'current_geoid_model', None) or 
                  getattr(self.raster1, 'original_geoid_model', None))
        geoid2 = (getattr(self.raster2, 'current_geoid_model', None) or 
                  getattr(self.raster2, 'original_geoid_model', None))
        
        match = _geoid_equivalent(geoid1, geoid2)
        
        return {
            'match': match,
            'raster1_geoid': geoid1,
            'raster2_geoid': geoid2,
            'details': "Geoid models match" if match else f"Geoid models differ: {geoid1} vs {geoid2}"
        }
    
    def check_epoch_match(self, tolerance: float = 0.01) -> Dict[str, Any]:
        """
        Check if epochs of both rasters match within tolerance.
        
        Parameters
        ----------
        tolerance : float
            Allowed difference in decimal years (default 0.01 ≈ 3.65 days)
            
        Returns
        -------
        dict
            {
                'match': bool,
                'raster1_epoch': float|None,
                'raster2_epoch': float|None,
                'difference': float|None,
                'details': str
            }
        """
        epoch1 = getattr(self.raster1, 'epoch', None)
        epoch2 = getattr(self.raster2, 'epoch', None)
        
        if epoch1 is None and epoch2 is None:
            return {
                'match': True,
                'raster1_epoch': None,
                'raster2_epoch': None,
                'difference': None,
                'details': "Both epochs unknown - assuming match"
            }
        
        if epoch1 is None or epoch2 is None:
            return {
                'match': False,
                'raster1_epoch': epoch1,
                'raster2_epoch': epoch2,
                'difference': None,
                'details': f"One epoch unknown: {epoch1} vs {epoch2}"
            }
        
        diff = abs(epoch1 - epoch2)
        match = diff <= tolerance
        
        return {
            'match': match,
            'raster1_epoch': epoch1,
            'raster2_epoch': epoch2,
            'difference': diff,
            'details': f"Epochs {'match' if match else 'differ'}: {epoch1:.4f} vs {epoch2:.4f} (Δ={diff:.4f} years)"
        }
    
    def check_vertical_units_match(self) -> Dict[str, Any]:
        """
        Check if vertical units of both rasters match.
        
        Uses UnitInfo objects when available for robust comparison, including
        proper handling of unit aliases and automatic conversion factor calculation.
        
        Returns
        -------
        dict
            {
                'match': bool,
                'raster1_units': str|None,
                'raster2_units': str|None,
                'raster1_unit_info': UnitInfo|None,
                'raster2_unit_info': UnitInfo|None,
                'conversion_factor': float|None,
                'conversion_needed': str|None,
                'details': str
            }
        """
        # Try to get UnitInfo objects first (from updated raster.py)
        unit_info1 = getattr(self.raster1, 'vertical_unit', None)
        if unit_info1 is None:
            unit_info1 = getattr(self.raster1, 'current_vertical_unit', None)

        unit_info2 = getattr(self.raster2, 'vertical_unit', None)
        if unit_info2 is None:
            unit_info2 = getattr(self.raster2, 'current_vertical_unit', None)

        # Get string representation for fallback
        if hasattr(unit_info1, 'name'):
            units1 = unit_info1.name
        else:
            units1 = (getattr(self.raster1, 'current_vertical_units', None) or
                      getattr(self.raster1, 'vertical_units', None) or
                      'meter')  # Default to meters if unknown

        if hasattr(unit_info2, 'name'):
            units2 = unit_info2.name
        else:
            units2 = (getattr(self.raster2, 'current_vertical_units', None) or
                      getattr(self.raster2, 'vertical_units', None) or
                      'meter')  # Default to meters if unknown

        # Use the helper function for comparison
        if unit_info1 is not None and unit_info2 is not None:
            match, conversion_factor = _units_equivalent(unit_info1, unit_info2)
            display1 = getattr(unit_info1, 'display_name', str(unit_info1))
            display2 = getattr(unit_info2, 'display_name', str(unit_info2))
        else:
            match, conversion_factor = _units_equivalent(units1, units2)
            display1 = units1
            display2 = units2
        
        # Build conversion info string
        conversion_needed = None
        if not match:
            if conversion_factor is not None:
                conversion_needed = f"{display1} → {display2} (factor: {conversion_factor:.10g})"
            else:
                conversion_needed = f"{display1} → {display2}"
        
        # Build details string
        if match:
            details = f"Vertical units match: {display1}"
        else:
            # Check if one is unknown
            unit1_unknown = (unit_info1 is not None and hasattr(unit_info1, 'name') and unit_info1.name == "unknown") or display1 == "unknown"
            unit2_unknown = (unit_info2 is not None and hasattr(unit_info2, 'name') and unit_info2.name == "unknown") or display2 == "unknown"
            
            if unit1_unknown or unit2_unknown:
                unknown_which = "raster1" if unit1_unknown else "raster2"
                known_which = "raster2" if unit1_unknown else "raster1"
                known_unit = display2 if unit1_unknown else display1
                details = f"Unit mismatch: {unknown_which} unit UNKNOWN, {known_which} is {known_unit} - use set_units() to specify"
            elif conversion_factor is not None:
                details = f"Unit mismatch: {display1} vs {display2} (conversion factor: {conversion_factor:.10g})"
            else:
                details = f"Unit mismatch: {display1} vs {display2} (conversion factor unknown)"
        
        return {
            'match': match,
            'raster1_units': units1,
            'raster2_units': units2,
            'raster1_unit_info': unit_info1,
            'raster2_unit_info': unit_info2,
            'conversion_factor': conversion_factor,
            'conversion_needed': conversion_needed,
            'details': details
        }
    
    def check_grid_match(self) -> Dict[str, Any]:
        """
        Check if raster grids (resolution, dimensions, alignment) match.
        
        Returns
        -------
        dict
            {
                'match': bool,
                'resolution_match': bool,
                'dimensions_match': bool,
                'transform_match': bool,
                'raster1_grid': dict,
                'raster2_grid': dict,
                'details': str
            }
        """
        def get_grid_info(r):
            try:
                with rasterio.open(r.filename) as src:
                    return {
                        'width': src.width,
                        'height': src.height,
                        'transform': src.transform,
                        'res_x': abs(src.transform.a),
                        'res_y': abs(src.transform.e),
                        'origin_x': src.transform.c,
                        'origin_y': src.transform.f,
                    }
            except Exception:
                return {
                    'width': getattr(r, 'width', None),
                    'height': getattr(r, 'height', None),
                    'transform': getattr(r, 'transform', None),
                    'res_x': r.resolution if hasattr(r, 'resolution') else None,
                    'res_y': r.resolution if hasattr(r, 'resolution') else None,
                    'origin_x': None,
                    'origin_y': None,
                }
        
        g1 = get_grid_info(self.raster1)
        g2 = get_grid_info(self.raster2)
        
        # Check resolution match (within 1e-6)
        res_tol = 1e-6
        res_match = (
            g1['res_x'] is not None and g2['res_x'] is not None and
            abs(g1['res_x'] - g2['res_x']) < res_tol and
            abs(g1['res_y'] - g2['res_y']) < res_tol
        )
        
        # Check dimensions match
        dim_match = (g1['width'] == g2['width'] and g1['height'] == g2['height'])
        
        # Check transform match (pixel-perfect alignment)
        transform_match = (g1['transform'] == g2['transform'])
        
        overall_match = res_match and dim_match and transform_match
        
        details = []
        if not res_match:
            details.append(f"Resolution differ: ({g1['res_x']}, {g1['res_y']}) vs ({g2['res_x']}, {g2['res_y']})")
        if not dim_match:
            details.append(f"Dimensions differ: ({g1['width']}, {g1['height']}) vs ({g2['width']}, {g2['height']})")
        if not transform_match:
            details.append("Transform/alignment differ")
        if overall_match:
            details.append("Grids match exactly")
        
        return {
            'match': overall_match,
            'resolution_match': res_match,
            'dimensions_match': dim_match,
            'transform_match': transform_match,
            'raster1_grid': g1,
            'raster2_grid': g2,
            'details': " | ".join(details)
        }
    
    def check_all_match(self, epoch_tolerance: float = 0.01) -> Dict[str, Any]:
        """
        Comprehensive check of all CRS, epoch, geoid, and grid parameters.
        
        Parameters
        ----------
        epoch_tolerance : float
            Allowed epoch difference in decimal years
            
        Returns
        -------
        dict
            Complete comparison report with all checks
        """
        horiz = self.check_horizontal_crs_match()
        vert = self.check_vertical_crs_match()
        compound = self.check_compound_crs_match()
        geoid = self.check_geoid_match()
        epoch = self.check_epoch_match(tolerance=epoch_tolerance)
        grid = self.check_grid_match()
        units = self.check_vertical_units_match()
        
        all_match = (
            horiz['match'] and
            vert['match'] and
            geoid['match'] and
            epoch['match'] and
            grid['match'] and
            units['match']
        )
        
        transformations_needed = []
        if not epoch['match']:
            transformations_needed.append("epoch")
        if not horiz['match']:
            transformations_needed.append("horizontal_crs")
        if not vert['match'] or not geoid['match']:
            transformations_needed.append("vertical_datum")
        if not units['match']:
            transformations_needed.append("vertical_units")
        if not grid['match']:
            transformations_needed.append("grid_alignment")
        
        return {
            'all_match': all_match,
            'transformations_needed': transformations_needed,
            'horizontal_crs': horiz,
            'vertical_crs': vert,
            'compound_crs': compound,
            'geoid': geoid,
            'epoch': epoch,
            'grid': grid,
            'vertical_units': units,
        }

    # =========================================================================
    # Extent and Overlap Methods
    # =========================================================================
    
    def get_extent_polygons(
        self, 
        valid_data_only: bool = True,
        simplify_tolerance: float = None,
    ) -> Dict[str, Any]:
        """
        Get extent polygons for both rasters.
        
        Parameters
        ----------
        valid_data_only : bool, default True
            If True, return polygons covering only valid (non-nodata) pixels.
            This may return MultiPolygon for rasters with holes or disconnected regions.
            If False, return simple bounding boxes.
        simplify_tolerance : float, optional
            Simplification tolerance in CRS units. Reduces vertex count for 
            large/complex polygons. Useful when pixel-level precision isn't needed.
        
        Returns
        -------
        dict
            {
                'raster1_extent': shapely.Polygon or MultiPolygon,
                'raster2_extent': shapely.Polygon or MultiPolygon,
                'raster1_bounds': tuple (minx, miny, maxx, maxy),
                'raster2_bounds': tuple (minx, miny, maxx, maxy),
                'raster1_valid_area': float,
                'raster2_valid_area': float,
            }
        """
        poly1 = _get_extent_polygon(self.raster1, valid_data_only=valid_data_only,
                                    simplify_tolerance=simplify_tolerance)
        poly2 = _get_extent_polygon(self.raster2, valid_data_only=valid_data_only,
                                    simplify_tolerance=simplify_tolerance)
        
        return {
            'raster1_extent': poly1,
            'raster2_extent': poly2,
            'raster1_bounds': poly1.bounds if poly1 is not None else None,
            'raster2_bounds': poly2.bounds if poly2 is not None else None,
            'raster1_valid_area': poly1.area if poly1 is not None else 0,
            'raster2_valid_area': poly2.area if poly2 is not None else 0,
        }
    
    def get_overlap_polygon(
        self,
        valid_data_only: bool = True,
        simplify_tolerance: float = None,
    ) -> Dict[str, Any]:
        """
        Compute the overlap area between the two rasters.
        
        Note: Rasters should be in the same CRS for meaningful results.
        If they're in different CRS, the bounding box overlap will still
        work but valid_data_only=True results may be inaccurate.
        
        Parameters
        ----------
        valid_data_only : bool, default True
            If True, compute overlap of valid data regions only (excludes nodata).
            If False, use simple bounding box intersection.
        simplify_tolerance : float, optional
            Simplification tolerance in CRS units for the polygon geometries.
        
        Returns
        -------
        dict
            {
                'overlap_polygon': shapely.Polygon/MultiPolygon or None,
                'overlap_bounds': tuple or None,
                'overlap_area': float,
                'raster1_area': float,
                'raster2_area': float,
                'overlap_fraction_r1': float (overlap_area / raster1_area),
                'overlap_fraction_r2': float (overlap_area / raster2_area),
                'has_overlap': bool
            }
        """
        poly1 = _get_extent_polygon(self.raster1, valid_data_only=valid_data_only,
                                    simplify_tolerance=simplify_tolerance)
        poly2 = _get_extent_polygon(self.raster2, valid_data_only=valid_data_only,
                                    simplify_tolerance=simplify_tolerance)
        
        if poly1 is None or poly2 is None:
            return {
                'overlap_polygon': None,
                'overlap_bounds': None,
                'overlap_area': 0.0,
                'raster1_area': poly1.area if poly1 is not None else 0.0,
                'raster2_area': poly2.area if poly2 is not None else 0.0,
                'overlap_fraction_r1': 0.0,
                'overlap_fraction_r2': 0.0,
                'has_overlap': False,
            }
        
        overlap = poly1.intersection(poly2)
        
        if overlap.is_empty:
            return {
                'overlap_polygon': None,
                'overlap_bounds': None,
                'overlap_area': 0.0,
                'raster1_area': poly1.area,
                'raster2_area': poly2.area,
                'overlap_fraction_r1': 0.0,
                'overlap_fraction_r2': 0.0,
                'has_overlap': False,
            }
        
        return {
            'overlap_polygon': overlap,
            'overlap_bounds': overlap.bounds,
            'overlap_area': overlap.area,
            'raster1_area': poly1.area,
            'raster2_area': poly2.area,
            'overlap_fraction_r1': overlap.area / poly1.area if poly1.area > 0 else 0,
            'overlap_fraction_r2': overlap.area / poly2.area if poly2.area > 0 else 0,
            'has_overlap': True,
        }
    
    def save_overlap_polygon(
        self,
        output_path: str,
        valid_data_only: bool = True,
        simplify_tolerance: float = None,
        include_individual: bool = False,
    ) -> str:
        """
        Save the overlap polygon to a file (GeoJSON, Shapefile, or GeoPackage).
        
        Parameters
        ----------
        output_path : str
            Output file path. Format determined by extension:
            - .geojson, .json: GeoJSON
            - .shp: Shapefile
            - .gpkg: GeoPackage
        valid_data_only : bool, default True
            If True, use valid data footprints instead of bounding boxes.
        simplify_tolerance : float, optional
            Simplification tolerance for polygon geometries.
        include_individual : bool, default False
            If True, also save individual raster footprints as separate features.
            
        Returns
        -------
        str
            Path to the saved file
        """
        try:
            import geopandas as gpd
            from shapely.geometry import mapping
        except ImportError:
            raise ImportError("geopandas is required for saving polygons")
        
        # Get polygons
        poly1 = _get_extent_polygon(self.raster1, valid_data_only=valid_data_only,
                                    simplify_tolerance=simplify_tolerance)
        poly2 = _get_extent_polygon(self.raster2, valid_data_only=valid_data_only,
                                    simplify_tolerance=simplify_tolerance)
        
        overlap = None
        if poly1 is not None and poly2 is not None:
            overlap = poly1.intersection(poly2)
            if overlap.is_empty:
                overlap = None
        
        # Get CRS from raster2 (the reference)
        crs = None
        try:
            with rasterio.open(self.raster2.filename) as src:
                crs = src.crs
        except Exception:
            pass
        
        # Build GeoDataFrame
        records = []
        
        if overlap is not None:
            records.append({
                'geometry': overlap,
                'type': 'overlap',
                'area': overlap.area,
                'source': 'intersection',
            })
        
        if include_individual:
            if poly1 is not None:
                records.append({
                    'geometry': poly1,
                    'type': 'raster1_extent',
                    'area': poly1.area,
                    'source': self.raster1.filename,
                })
            if poly2 is not None:
                records.append({
                    'geometry': poly2,
                    'type': 'raster2_extent',
                    'area': poly2.area,
                    'source': self.raster2.filename,
                })
        
        if not records:
            raise ValueError("No valid polygons to save")
        
        gdf = gpd.GeoDataFrame(records, crs=crs)
        
        # Determine driver from extension
        ext = Path(output_path).suffix.lower()
        if ext in ('.geojson', '.json'):
            gdf.to_file(output_path, driver='GeoJSON')
        elif ext == '.shp':
            gdf.to_file(output_path, driver='ESRI Shapefile')
        elif ext == '.gpkg':
            gdf.to_file(output_path, driver='GPKG')
        else:
            # Default to GeoJSON
            gdf.to_file(output_path, driver='GeoJSON')
        
        return output_path
    
    def get_valid_data_mask(self, which: str = "both") -> np.ndarray:
        """
        Get a boolean mask of valid data pixels.
        
        Parameters
        ----------
        which : str, {'raster1', 'raster2', 'both'}
            Which raster(s) to get the mask for:
            - 'raster1': Valid in raster1 only
            - 'raster2': Valid in raster2 only  
            - 'both': Valid in BOTH rasters (intersection)
            
        Returns
        -------
        np.ndarray
            Boolean mask (True = valid data)
        """
        if which not in ('raster1', 'raster2', 'both'):
            raise ValueError("which must be 'raster1', 'raster2', or 'both'")
        
        with rasterio.open(self.raster1.filename) as src:
            data1 = src.read(1)
            nodata1 = src.nodata
        
        with rasterio.open(self.raster2.filename) as src:
            data2 = src.read(1)
            nodata2 = src.nodata
        
        valid1 = _create_valid_data_mask(data1, nodata1)
        valid2 = _create_valid_data_mask(data2, nodata2)
        
        if which == 'raster1':
            return valid1
        elif which == 'raster2':
            return valid2
        else:
            return valid1 & valid2

    # =========================================================================
    # Transformation Methods
    # =========================================================================
    
    def transform_raster1_to_match_raster2(
        self,
        interpolation_method: str = "bilinear",
        overwrite: bool = True,
        skip_epoch: bool = False,
        skip_horizontal: bool = False,
        skip_vertical: bool = False,
        skip_units: bool = False,
        skip_alignment: bool = False,
        verbose: bool = True,
        record_history: bool = True,
    ) -> Raster:
        """
        Transform raster1 to match raster2's reference frame.
        
        The transformation pipeline follows this order:
        1. Dynamic epoch transformation (if epochs differ)
        2. Horizontal CRS reprojection (if horizontal CRS differs)
        3. Vertical datum transformation (if vertical kind or geoid differs)
        3b. Vertical unit conversion (if units differ, e.g., feet to meters)
        4. Grid alignment (resample to match exact pixel grid)
        
        Think of this like converting coordinates through a series of 
        reference frame changes - similar to how GPS coordinates flow through
        the transformation pipeline from receiver to final map projection.
        
        Parameters
        ----------
        interpolation_method : str
            Resampling method ('nearest', 'bilinear', 'cubic', etc.)
        overwrite : bool
            Whether to overwrite intermediate files
        skip_epoch : bool
            Skip epoch transformation even if needed
        skip_horizontal : bool
            Skip horizontal CRS transformation
        skip_vertical : bool
            Skip vertical datum transformation
        skip_units : bool
            Skip vertical unit conversion
        skip_alignment : bool
            Skip final grid alignment
        verbose : bool
            Print transformation progress
        record_history : bool
            If True, record each transformation step in CRSHistory
            
        Returns
        -------
        Raster
            Transformed raster1 matching raster2's reference frame
        """
        import sys
        
        # Get comparison results to determine what transformations are needed
        comparison = self.check_all_match()
        
        if verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print("RasterPair: Transform raster1 to match raster2", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(f"Transformations needed: {comparison['transformations_needed']}", file=sys.stderr)
        
        # Start with raster1
        current = self.raster1
        self._transformation_history = []
        
        # Get target parameters from raster2
        target_epoch = getattr(self.raster2, 'epoch', None)
        target_horiz_crs = (
            getattr(self.raster2, 'current_horizontal_crs', None) or
            getattr(self.raster2, 'original_horizontal_crs', None) or
            getattr(self.raster2, 'crs', None)
        )
        target_geoid = (
            getattr(self.raster2, 'current_geoid_model', None) or
            getattr(self.raster2, 'original_geoid_model', None)
        )
        
        # Determine source/target vertical kinds
        source_is_ortho = getattr(self.raster1, 'is_orthometric', None)
        target_is_ortho = getattr(self.raster2, 'is_orthometric', None)
        
        source_vertical_kind = "orthometric" if source_is_ortho else "ellipsoidal" if source_is_ortho is False else None
        target_vertical_kind = "orthometric" if target_is_ortho else "ellipsoidal" if target_is_ortho is False else None
        
        source_geoid = (
            getattr(current, 'current_geoid_model', None) or
            getattr(current, 'original_geoid_model', None)
        )
        
        # 1. Dynamic epoch transformation
        if not skip_epoch and 'epoch' in comparison['transformations_needed']:
            if target_epoch is not None:
                src_epoch = getattr(current, 'epoch', None)
                if verbose:
                    print(f"\n[Step 1] Epoch transformation: {src_epoch} → {target_epoch}", file=sys.stderr)
                
                current = current.warp_raster(
                    dynamic_target_epoch=target_epoch,
                    interpolation_method=interpolation_method,
                    overwrite=overwrite,
                )
                
                step_info = {
                    'step': 'epoch_transform',
                    'source_epoch': src_epoch,
                    'target_epoch': target_epoch,
                    'source_file': self.raster1.filename,
                    'target_file': current.filename,
                }
                self._transformation_history.append(step_info)
                
                # Record in CRSHistory if available
                # Note: raster.py now handles interpolation tracking internally,
                # so we just record the RasterPair-level context
                if record_history and getattr(current, 'crs_history', None) is not None:
                    # The transformation entry was already recorded by raster.py
                    # Just add RasterPair context note
                    pass  # Interpolation tracking now handled by raster.py
                
                if verbose:
                    print(f"    ✓ Epoch transformation complete", file=sys.stderr)
        
        # 2. Horizontal CRS reprojection
        if not skip_horizontal and 'horizontal_crs' in comparison['transformations_needed']:
            if target_horiz_crs is not None:
                src_crs = getattr(current, 'crs', None)
                if verbose:
                    print(f"\n[Step 2] Horizontal reprojection", file=sys.stderr)
                    try:
                        src_epsg = _ensure_crs_obj(src_crs).to_epsg()
                        tgt_epsg = _ensure_crs_obj(target_horiz_crs).to_epsg()
                        print(f"    EPSG:{src_epsg} → EPSG:{tgt_epsg}", file=sys.stderr)
                    except Exception:
                        print(f"    {src_crs} → {target_horiz_crs}", file=sys.stderr)
                
                prev_file = current.filename
                current = current.warp_raster(
                    target_crs=target_horiz_crs,
                    interpolation_method=interpolation_method,
                    overwrite=overwrite,
                )
                
                step_info = {
                    'step': 'horizontal_reprojection',
                    'source_crs': str(src_crs)[:100] if src_crs else None,
                    'target_crs': str(target_horiz_crs)[:100],
                    'source_file': prev_file,
                    'target_file': current.filename,
                }
                self._transformation_history.append(step_info)
                
                # Record in CRSHistory
                # Note: raster.py now handles interpolation tracking internally
                if record_history and getattr(current, 'crs_history', None) is not None:
                    # The transformation entry was already recorded by raster.py
                    pass  # Interpolation tracking now handled by raster.py
                
                if verbose:
                    print(f"    ✓ Horizontal reprojection complete", file=sys.stderr)
        
        # 3. Vertical datum transformation
        needs_vertical = 'vertical_datum' in comparison['transformations_needed']
        if not skip_vertical and needs_vertical:
            if source_vertical_kind is not None and target_vertical_kind is not None:
                if verbose:
                    print(f"\n[Step 3] Vertical datum transformation", file=sys.stderr)
                    print(f"    {source_vertical_kind} → {target_vertical_kind}", file=sys.stderr)
                    print(f"    Geoid: {source_geoid} → {target_geoid}", file=sys.stderr)
                
                prev_file = current.filename
                current = current.warp_raster(
                    source_vertical_kind=source_vertical_kind,
                    target_vertical_kind=target_vertical_kind,
                    source_geoid_model=source_geoid,
                    target_geoid_model=target_geoid,
                    interpolation_method=interpolation_method,
                    overwrite=overwrite,
                )
                
                step_info = {
                    'step': 'vertical_datum',
                    'source_kind': source_vertical_kind,
                    'target_kind': target_vertical_kind,
                    'source_geoid': source_geoid,
                    'target_geoid': target_geoid,
                    'source_file': prev_file,
                    'target_file': current.filename,
                }
                self._transformation_history.append(step_info)
                
                # Record in CRSHistory
                # Note: raster.py now handles interpolation tracking internally
                if record_history and getattr(current, 'crs_history', None) is not None:
                    # The transformation entry was already recorded by raster.py
                    pass  # Interpolation tracking now handled by raster.py
                
                if verbose:
                    print(f"    ✓ Vertical datum transformation complete", file=sys.stderr)
        
        # 3b. Vertical unit conversion
        needs_unit_conversion = 'vertical_units' in comparison['transformations_needed']
        if not skip_units and needs_unit_conversion:
            # Get target units - handle UnitInfo objects
            target_unit_obj = getattr(self.raster2, 'vertical_unit', None)
            if hasattr(target_unit_obj, 'name'):
                target_units = target_unit_obj.name
            else:
                target_units = (
                    getattr(self.raster2, 'current_vertical_units', None) or
                    getattr(self.raster2, 'vertical_units', None) or
                    'meter'
                )

            # Get source units - handle UnitInfo objects
            source_unit_obj = getattr(current, 'vertical_unit', None)
            if hasattr(source_unit_obj, 'name'):
                source_units = source_unit_obj.name
            else:
                source_units = (
                    getattr(current, 'current_vertical_units', None) or
                    getattr(current, 'vertical_units', None) or
                    'meter'
                )

            if source_units != target_units:
                if verbose:
                    print(f"\n[Step 3b] Vertical unit conversion", file=sys.stderr)
                    print(f"    {source_units} → {target_units}", file=sys.stderr)
                
                prev_file = current.filename
                current = current.convert_vertical_units(
                    target_units=target_units,
                    overwrite=overwrite,
                )
                
                step_info = {
                    'step': 'unit_conversion',
                    'source_units': source_units,
                    'target_units': target_units,
                    'source_file': prev_file,
                    'target_file': current.filename,
                }
                self._transformation_history.append(step_info)
                
                if verbose:
                    print(f"    ✓ Unit conversion complete", file=sys.stderr)
        
        # 4. Grid alignment
        if not skip_alignment:
            if verbose:
                print(f"\n[Step 4] Grid alignment to reference", file=sys.stderr)
            
            prev_file = current.filename
            current = current.warp_raster(
                align_to=self.raster2,
                interpolation_method=interpolation_method,
                overwrite=overwrite,
            )
            
            step_info = {
                'step': 'grid_alignment',
                'reference': self.raster2.filename,
                'source_file': prev_file,
                'target_file': current.filename,
            }
            self._transformation_history.append(step_info)
            
            # Record in CRSHistory
            # Note: raster.py now handles interpolation tracking internally
            if record_history and getattr(current, 'crs_history', None) is not None:
                # The transformation entry was already recorded by raster.py
                pass  # Interpolation tracking now handled by raster.py
            
            if verbose:
                print(f"    ✓ Grid alignment complete", file=sys.stderr)
        
        # Cache the result
        self._raster1_transformed = current
        
        if verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print("Transformation pipeline complete", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
        
        return current

    # =========================================================================
    # Differencing Methods
    # =========================================================================
    
    def compute_difference(
        self,
        transform_first: bool = True,
        skip_epoch: bool = False,
        interpolation_method: str = "bilinear",
        clip_to_overlap: bool = True,
        output_path: Optional[str] = None,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute raster difference (raster2 - raster1).
        
        This is the primary method for DEM differencing. By default, it:
        1. Transforms raster1 to match raster2's reference frame
        2. Clips to the overlap region (if requested)
        3. Computes the difference (raster2 - raster1)
        4. Returns the difference raster with statistics
        
        The difference is computed as raster2 - raster1, which for DEMs 
        means positive values indicate elevation increase over time.
        
        NoData Handling
        ---------------
        Pixels are masked (set to NaN in output) if ANY of these conditions are true:
        - Pixel equals the nodata value in either raster
        - Pixel is NaN in either raster
        - Pixel is Inf in either raster
        - Pixel is outside the overlap region (if clip_to_overlap=True)
        
        Parameters
        ----------
        transform_first : bool
            If True, transform raster1 to match raster2 before differencing.
            If False, assume rasters are already aligned.
        skip_epoch : bool
            If True, skip epoch transformation (faster but less accurate if epochs differ).
            Only used if transform_first=True.
        interpolation_method : str
            Resampling method for transformation
        clip_to_overlap : bool
            If True, mask pixels outside the valid data overlap region.
            This uses the actual valid data footprints, not just bounding boxes.
        output_path : str, optional
            Output file path for difference raster
        overwrite : bool
            Whether to overwrite existing files
        verbose : bool
            Print progress information
            
        Returns
        -------
        dict
            {
                'difference_raster': Raster,
                'difference_raster_path': str,
                'stats': {min, max, mean, std, median, count_valid, count_total},
                'histogram': {hist, bin_edges},
                'transformation_history': list,
                'metadata': dict
            }
        """
        import sys
        
        if verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print("RasterPair: Computing Difference (raster2 - raster1)", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
        
        # Transform raster1 if needed
        if transform_first:
            raster1_aligned = self.transform_raster1_to_match_raster2(
                skip_epoch=skip_epoch,
                interpolation_method=interpolation_method,
                overwrite=overwrite,
                verbose=verbose,
            )
        else:
            raster1_aligned = self._raster1_transformed or self.raster1
        
        # Read both rasters
        with rasterio.open(raster1_aligned.filename) as src1:
            data1 = src1.read(1).astype(np.float64)
            nodata1 = src1.nodata
            profile1 = src1.profile
            transform1 = src1.transform
        
        with rasterio.open(self.raster2.filename) as src2:
            data2 = src2.read(1).astype(np.float64)
            nodata2 = src2.nodata
            profile2 = src2.profile
            transform2 = src2.transform
            crs2 = src2.crs
            height2 = src2.height
            width2 = src2.width
        
        # Create robust valid data masks using the utility function
        valid1 = _create_valid_data_mask(data1, nodata1)
        valid2 = _create_valid_data_mask(data2, nodata2)
        
        # Combined valid mask (must be valid in BOTH rasters)
        valid_both = valid1 & valid2
        
        if verbose:
            n_valid1 = np.sum(valid1)
            n_valid2 = np.sum(valid2)
            n_valid_both = np.sum(valid_both)
            total = data1.size
            print(f"\nValid pixels:", file=sys.stderr)
            print(f"    Raster1: {n_valid1:,} / {total:,} ({100*n_valid1/total:.1f}%)", file=sys.stderr)
            print(f"    Raster2: {n_valid2:,} / {total:,} ({100*n_valid2/total:.1f}%)", file=sys.stderr)
            print(f"    Both:    {n_valid_both:,} / {total:,} ({100*n_valid_both/total:.1f}%)", file=sys.stderr)
        
        # Compute difference: raster2 - raster1
        # Initialize with NaN, then fill valid pixels
        diff = np.full_like(data1, np.nan, dtype=np.float64)
        diff[valid_both] = data2[valid_both] - data1[valid_both]
        
        # Compute statistics on valid pixels only
        valid_diff = diff[~np.isnan(diff)]
        if len(valid_diff) > 0:
            stats = {
                'min': float(np.min(valid_diff)),
                'max': float(np.max(valid_diff)),
                'mean': float(np.mean(valid_diff)),
                'std': float(np.std(valid_diff)),
                'median': float(np.median(valid_diff)),
                'count_valid': int(len(valid_diff)),
                'count_total': int(data1.size),
                'rmse': float(np.sqrt(np.mean(valid_diff**2))),
                'mae': float(np.mean(np.abs(valid_diff))),
            }
            
            # Compute histogram
            hist, bin_edges = np.histogram(valid_diff, bins=50)
            histogram = {
                'hist': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
            }
            
            # Compute percentiles for robust statistics
            percentiles = np.percentile(valid_diff, [2.5, 16, 50, 84, 97.5])
            stats['percentiles'] = {
                'p2.5': float(percentiles[0]),
                'p16': float(percentiles[1]),
                'p50': float(percentiles[2]),
                'p84': float(percentiles[3]),
                'p97.5': float(percentiles[4]),
            }
        else:
            stats = {
                'min': None, 'max': None, 'mean': None, 'std': None,
                'median': None, 'count_valid': 0, 'count_total': int(data1.size),
                'rmse': None, 'mae': None, 'percentiles': None
            }
            histogram = {'hist': [], 'bin_edges': []}
        
        if verbose:
            print(f"\nDifference statistics:", file=sys.stderr)
            if stats['mean'] is not None:
                print(f"    Min:    {stats['min']:.4f} m", file=sys.stderr)
                print(f"    Max:    {stats['max']:.4f} m", file=sys.stderr)
                print(f"    Mean:   {stats['mean']:.4f} m", file=sys.stderr)
                print(f"    Std:    {stats['std']:.4f} m", file=sys.stderr)
                print(f"    Median: {stats['median']:.4f} m", file=sys.stderr)
                print(f"    RMSE:   {stats['rmse']:.4f} m", file=sys.stderr)
            else:
                print("    No valid pixels for statistics", file=sys.stderr)
        
        # Generate output path
        if output_path is None:
            base = Path(self.raster2.filename).stem
            output_path = str(Path(self.raster2.filename).parent / f"{base}_diff.tif")
        
        # Write difference raster
        diff_profile = profile2.copy()
        diff_profile.update({
            'dtype': 'float32',
            'nodata': np.nan,
            'count': 1,
        })
        
        if os.path.exists(output_path) and not overwrite:
            raise ValueError(f"Output file exists and overwrite=False: {output_path}")
        
        with rasterio.open(output_path, 'w', **diff_profile) as dst:
            dst.write(diff.astype(np.float32), 1)
            
            # Add metadata tags
            dst.update_tags(
                DIFFERENCE_TYPE='raster2_minus_raster1',
                RASTER1_SOURCE=self.raster1.filename,
                RASTER2_SOURCE=self.raster2.filename,
                VALID_PIXEL_COUNT=str(stats['count_valid']),
                TOTAL_PIXEL_COUNT=str(stats['count_total']),
            )
        
        if verbose:
            print(f"\nDifference raster written to: {output_path}", file=sys.stderr)
        
        # Load as Raster object
        diff_raster = Raster.from_file(output_path, rtype='dod', metadata={})
        
        # Copy CRS info from raster2 (the reference)
        diff_raster.current_compound_crs = getattr(self.raster2, 'current_compound_crs', None)
        diff_raster.current_horizontal_crs = getattr(self.raster2, 'current_horizontal_crs', None)
        diff_raster.current_vertical_crs = getattr(self.raster2, 'current_vertical_crs', None)
        diff_raster.current_geoid_model = getattr(self.raster2, 'current_geoid_model', None)
        diff_raster.epoch = getattr(self.raster2, 'epoch', None)
        
        # Create comprehensive CRS history for the difference raster
        # This tracks dual-parent lineage (derived from BOTH raster1 and raster2)
        try:
            diff_raster.crs_history = CRSHistory(diff_raster)
            
            # Set dual-parent lineage
            diff_raster.crs_history.derived_from = self.raster2.filename  # Primary reference
            
            # Record the differencing operation with full provenance
            diff_raster.crs_history.record_raster_creation_entry(
                creation_parameters={
                    'source': 'RasterPair.compute_difference',
                    'operation': 'elevation_differencing',
                    'formula': 'raster2 - raster1',
                    # Parent raster info
                    'raster1_file': self.raster1.filename,
                    'raster2_file': self.raster2.filename,
                    'raster1_epoch': getattr(self.raster1, 'epoch', None),
                    'raster2_epoch': getattr(self.raster2, 'epoch', None),
                    'raster1_geoid': getattr(self.raster1, 'current_geoid_model', None) or getattr(self.raster1, 'original_geoid_model', None),
                    'raster2_geoid': getattr(self.raster2, 'current_geoid_model', None) or getattr(self.raster2, 'original_geoid_model', None),
                    # Processing info
                    'transformation_applied': transform_first,
                    'transformation_steps': self._transformation_history,
                    'interpolation_method': interpolation_method,
                    'clip_to_overlap': clip_to_overlap,
                    # Statistics
                    'stats': stats,
                },
                description=f"Difference raster (raster2 - raster1). "
                           f"Derived from {Path(self.raster1.filename).name} and {Path(self.raster2.filename).name}.",
                interpolation_method=interpolation_method if transform_first else None,
            )
            
            # Copy interpolation history from transformed raster1 if available
            if transform_first and self._raster1_transformed is not None:
                transformed_history = getattr(self._raster1_transformed, 'crs_history', None)
                if transformed_history is not None:
                    # Copy the interpolation entries from the transformation pipeline
                    for interp_entry in transformed_history.interpolation_history:
                        diff_raster.crs_history.interpolation_history.append(interp_entry)
            
            # Record each transformation step that was applied to raster1
            for step in self._transformation_history:
                step_type = step.get('step', 'unknown')
                diff_raster.crs_history.record_transformation_entry(
                    transformation_type=f"pre_difference_{step_type}",
                    source_crs_proj=None,
                    target_crs_proj=None,
                    method=step_type,
                    interpolation_method=interpolation_method,
                    source_file=step.get('source_file'),
                    target_file=step.get('target_file'),
                    note=f"Applied to raster1 before differencing: {step_type}",
                    **{k: v for k, v in step.items() if k not in ('step', 'source_file', 'target_file', 'source_crs', 'target_crs')}
                )
            
        except Exception as e:
            import sys
            if verbose:
                print(f"[WARNING] Could not create CRSHistory for difference raster: {e}", file=sys.stderr)
        
        # Build metadata summary
        metadata = {
            'raster1_source': self.raster1.filename,
            'raster2_source': self.raster2.filename,
            'raster1_epoch': getattr(self.raster1, 'epoch', None),
            'raster2_epoch': getattr(self.raster2, 'epoch', None),
            'transformation_applied': transform_first,
            'interpolation_method': interpolation_method,
            'transformation_history': self._transformation_history,
        }
        
        return {
            'difference_raster': diff_raster,
            'difference_raster_path': output_path,
            'stats': stats,
            'histogram': histogram,
            'transformation_history': self._transformation_history,
            'metadata': metadata,
        }

    # =========================================================================
    # Provenance and History Methods
    # =========================================================================
    
    def get_transformation_history(self) -> List[Dict[str, Any]]:
        """
        Get the transformation history from the last transform_raster1_to_match_raster2 call.
        
        Returns
        -------
        list of dict
            List of transformation steps with details
        """
        return self._transformation_history.copy()
    
    def get_full_provenance(self, include_crs_history: bool = True) -> Dict[str, Any]:
        """
        Get complete provenance information for the RasterPair operations.
        
        This includes information about both input rasters, any transformations
        applied, interpolation methods used, and the resulting products.
        
        Parameters
        ----------
        include_crs_history : bool
            If True, include full CRSHistory from both rasters
            
        Returns
        -------
        dict
            Complete provenance report
        """
        def get_raster_info(r: Raster, name: str) -> Dict[str, Any]:
            info = {
                'name': name,
                'filename': getattr(r, 'filename', None),
                'epoch': getattr(r, 'epoch', None),
                'is_orthometric': getattr(r, 'is_orthometric', None),
                'geoid_model': (getattr(r, 'current_geoid_model', None) or 
                               getattr(r, 'original_geoid_model', None)),
                'horizontal_crs_epsg': None,
                'vertical_crs_epsg': None,
                'bounds': None,
                'resolution': getattr(r, 'resolution', None),
                'shape': (getattr(r, 'height', None), getattr(r, 'width', None)),
            }
            
            # Get EPSG codes
            horiz_crs = (getattr(r, 'current_horizontal_crs', None) or 
                        getattr(r, 'original_horizontal_crs', None))
            if horiz_crs:
                try:
                    info['horizontal_crs_epsg'] = _ensure_crs_obj(horiz_crs).to_epsg()
                except Exception:
                    pass
            
            vert_crs = (getattr(r, 'current_vertical_crs', None) or 
                       getattr(r, 'original_vertical_crs', None))
            if vert_crs:
                try:
                    info['vertical_crs_epsg'] = _ensure_crs_obj(vert_crs).to_epsg()
                except Exception:
                    pass
            
            # Get bounds
            bounds = getattr(r, 'bounds', None)
            if bounds:
                info['bounds'] = {
                    'left': bounds.left,
                    'bottom': bounds.bottom,
                    'right': bounds.right,
                    'top': bounds.top,
                }
            
            # Include CRSHistory if requested
            if include_crs_history and getattr(r, 'crs_history', None) is not None:
                try:
                    info['crs_history'] = r.crs_history.to_dict()
                    # Also include interpolation summary
                    info['interpolation_summary'] = r.crs_history.get_interpolation_summary()
                except Exception:
                    info['crs_history'] = None
                    info['interpolation_summary'] = None
            
            return info
        
        provenance = {
            'raster1': get_raster_info(self.raster1, 'raster1'),
            'raster2': get_raster_info(self.raster2, 'raster2'),
            'comparison': self.check_all_match(),
            'transformation_history': self._transformation_history,
        }
        
        # Include transformed raster1 if available
        if self._raster1_transformed is not None:
            provenance['raster1_transformed'] = get_raster_info(
                self._raster1_transformed, 'raster1_transformed'
            )
            
            # Add interpolation chain summary
            if getattr(self._raster1_transformed, 'crs_history', None) is not None:
                try:
                    provenance['interpolation_chain'] = self._raster1_transformed.crs_history.get_interpolation_chain()
                except Exception:
                    provenance['interpolation_chain'] = None
        
        return provenance
    
    def export_provenance_report(
        self,
        output_path: str,
        format: str = 'json',
        include_crs_history: bool = True,
    ) -> str:
        """
        Export provenance information to a file.
        
        Parameters
        ----------
        output_path : str
            Output file path
        format : str
            Output format: 'json' or 'yaml'
        include_crs_history : bool
            If True, include full CRSHistory from both rasters
            
        Returns
        -------
        str
            Path to the exported file
        """
        import json
        from datetime import datetime
        
        provenance = self.get_full_provenance(include_crs_history=include_crs_history)
        
        # Add export metadata
        provenance['_export_info'] = {
            'exported_at': datetime.now().isoformat(),
            'format': format,
        }
        
        # Make JSON-serializable (handle non-serializable types)
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
        
        provenance = make_serializable(provenance)
        
        if format.lower() == 'yaml':
            try:
                import yaml
                with open(output_path, 'w') as f:
                    yaml.dump(provenance, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                # Fall back to JSON if yaml not available
                with open(output_path, 'w') as f:
                    json.dump(provenance, f, indent=2, default=str)
        else:
            with open(output_path, 'w') as f:
                json.dump(provenance, f, indent=2, default=str)
        
        return output_path
    
    def print_summary(self) -> None:
        """
        Print a human-readable summary of the RasterPair status.
        """
        comparison = self.check_all_match()
        
        print("\n" + "="*60)
        print("RasterPair Summary")
        print("="*60)
        
        print(f"\nRaster 1: {Path(self.raster1.filename).name}")
        print(f"  Epoch: {getattr(self.raster1, 'epoch', 'N/A')}")
        print(f"  Geoid: {getattr(self.raster1, 'current_geoid_model', None) or getattr(self.raster1, 'original_geoid_model', 'N/A')}")
        print(f"  Orthometric: {getattr(self.raster1, 'is_orthometric', 'N/A')}")
        print(f"  Vertical Units: {getattr(self.raster1, 'current_vertical_units', None) or getattr(self.raster1, 'vertical_units', 'unknown')}")
        
        print(f"\nRaster 2: {Path(self.raster2.filename).name}")
        print(f"  Epoch: {getattr(self.raster2, 'epoch', 'N/A')}")
        print(f"  Geoid: {getattr(self.raster2, 'current_geoid_model', None) or getattr(self.raster2, 'original_geoid_model', 'N/A')}")
        print(f"  Orthometric: {getattr(self.raster2, 'is_orthometric', 'N/A')}")
        print(f"  Vertical Units: {getattr(self.raster2, 'current_vertical_units', None) or getattr(self.raster2, 'vertical_units', 'unknown')}")
        
        print(f"\nComparison:")
        print(f"  Horizontal CRS match: {comparison['horizontal_crs']['match']}")
        print(f"  Vertical CRS match:   {comparison['vertical_crs']['match']}")
        print(f"  Geoid match:          {comparison['geoid']['match']}")
        print(f"  Epoch match:          {comparison['epoch']['match']}")
        print(f"  Units match:          {comparison['vertical_units']['match']}")
        print(f"  Grid match:           {comparison['grid']['match']}")
        
        if comparison['transformations_needed']:
            print(f"\nTransformations needed: {', '.join(comparison['transformations_needed'])}")
        else:
            print(f"\nRasters are fully aligned - ready for differencing!")
        
        if self._transformation_history:
            print(f"\nTransformation steps applied:")
            for i, step in enumerate(self._transformation_history, 1):
                print(f"  {i}. {step.get('step', 'unknown')}")
        
        # Show interpolation history if available
        if self._raster1_transformed is not None:
            crs_hist = getattr(self._raster1_transformed, 'crs_history', None)
            if crs_hist is not None:
                try:
                    interp_summary = crs_hist.get_interpolation_summary()
                    if interp_summary['count'] > 0:
                        print(f"\nInterpolation methods applied:")
                        print(f"  Total operations: {interp_summary['count']}")
                        print(f"  Methods used: {', '.join(interp_summary['methods_used'])}")
                        if interp_summary['any_resampling_skipped']:
                            print(f"  Note: Some resampling was skipped (sub-pixel shifts)")
                        print(f"  Chain: {' → '.join(crs_hist.get_interpolation_chain())}")
                except Exception:
                    pass
        
        print("="*60 + "\n")

    # =========================================================================
    # Legacy Methods (for backward compatibility)
    # =========================================================================
    
    def compare_raster_metadata(self) -> Dict[str, Any]:
        """
        Compare metadata of two rasters and return a structured report.
        
        Legacy method - see check_all_match() for more comprehensive comparison.
        """
        def collect(r: Raster) -> Dict[str, Any]:
            filename = getattr(r, "filename", None)
            file_format = None
            if filename:
                file_format = Path(filename).suffix.lower().lstrip(".")

            bounds: Optional[Tuple[float, float, float, float]] = getattr(r, "bounds", None)
            transform = getattr(r, "transform", None)
            width = getattr(r, "width", None)
            height = getattr(r, "height", None)
            geoid_model = getattr(r, "current_geoid_model", None) or getattr(r, "original_geoid_model", None)
            epoch = getattr(r, "epoch", None)

            res_x = res_y = None
            if transform is not None:
                res_x = transform.a
                res_y = -transform.e

            compound_crs_summary = {
                "epsg": None,
                "name": None,
                "proj4": None,
            }
            crs_obj = getattr(r, "crs", None)
            if crs_obj is not None:
                try:
                    crs = _ensure_crs_obj(crs_obj)
                    compound_crs_summary = {
                        "epsg": crs.to_epsg(),
                        "name": crs.name,
                        "proj4": crs.to_string(),
                    }
                except Exception:
                    compound_crs_summary = {
                        "epsg": None,
                        "name": str(crs_obj),
                        "proj4": None,
                    }

            return {
                "filename": filename,
                "file_format": file_format,
                "bounds": bounds,
                "width": width,
                "height": height,
                "res_x": res_x,
                "res_y": res_y,
                "geoid_model": geoid_model,
                "epoch": epoch,
                "compound_crs": compound_crs_summary,
            }

        m1 = collect(self.raster1)
        m2 = collect(self.raster2)

        report: Dict[str, Any] = {"raster1": m1, "raster2": m2, "fields": {}}
        for key in sorted(m1.keys()):
            v1 = m1[key]
            v2 = m2[key]
            same = v1 == v2
            report["fields"][key] = {"raster1": v1, "raster2": v2, "same": same}
        return report
    
    def transform_compare_to_match_reference(
        self,
        skip_epoch: bool = False,
        skip_horizontal: bool = False,
        skip_vertical: bool = False,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> Raster:
        """
        Transform raster1 (compare) to match raster2 (reference)'s reference frame.
        
        All transformations are composed into a single resampling pass for efficiency
        and to preserve data quality.
        
        Parameters
        ----------
        skip_epoch : bool
            Skip epoch transformation even if needed
        skip_horizontal : bool
            Skip horizontal CRS transformation
        skip_vertical : bool
            Skip vertical datum transformation
        overwrite : bool
            Whether to overwrite intermediate files
        verbose : bool
            Print transformation progress
            
        Returns
        -------
        Raster
            Transformed raster1 matching raster2's reference frame
        """
        import sys
        
        comparison = self.check_all_match()
        
        if verbose:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print("RasterPair: Transform compare to match reference", file=sys.stderr)
            print(f"{'=' * 60}", file=sys.stderr)
            print(f"Transformations needed: {comparison['transformations_needed']}", file=sys.stderr)
        
        self._transformation_history = []
        
        # Get target parameters from reference (raster2)
        target_epoch = getattr(self.raster2, 'epoch', None)
        target_crs = (
            getattr(self.raster2, 'crs', None) or
            getattr(self.raster2, 'current_compound_crs', None)
        )
        target_geoid = (
            getattr(self.raster2, 'current_geoid_model', None) or
            getattr(self.raster2, 'geoid_model', None)
        )
        
        # Determine vertical kinds
        source_is_ortho = getattr(self.raster1, 'is_orthometric', None)
        target_is_ortho = getattr(self.raster2, 'is_orthometric', None)
        
        source_vertical_kind = (
            "orthometric" if source_is_ortho else
            "ellipsoidal" if source_is_ortho is False else
            None
        )
        target_vertical_kind = (
            "orthometric" if target_is_ortho else
            "ellipsoidal" if target_is_ortho is False else
            None
        )
        source_geoid = (
            getattr(self.raster1, 'current_geoid_model', None) or
            getattr(self.raster1, 'geoid_model', None)
        )
        
        # Determine what's actually needed
        needs_epoch = (
            not skip_epoch and 
            'epoch' in comparison['transformations_needed'] and
            target_epoch is not None
        )
        needs_vertical = (
            not skip_vertical and
            'vertical_datum' in comparison['transformations_needed']
        )
        needs_horizontal = (
            not skip_horizontal and
            'horizontal_crs' in comparison['transformations_needed']
        )
        
        if verbose:
            src_epoch = getattr(self.raster1, 'epoch', None)
            if needs_epoch:
                print(f"  Epoch: {src_epoch:.4f} → {target_epoch:.4f}", file=sys.stderr)
            if needs_vertical:
                print(f"  Vertical: {source_vertical_kind} → {target_vertical_kind}", file=sys.stderr)
                print(f"  Geoid: {source_geoid} → {target_geoid}", file=sys.stderr)
            if needs_horizontal:
                print(f"  Horizontal CRS reprojection needed", file=sys.stderr)
        
        # SINGLE warp_raster call with ALL parameters
        if needs_epoch or needs_vertical or needs_horizontal:
            if verbose:
                print(f"\nExecuting combined transformation...", file=sys.stderr)
            
            current = self.raster1.warp_raster(
                # Horizontal parameters
                target_crs=target_crs if needs_horizontal else None,
                # Epoch parameters
                dynamic_target_epoch=target_epoch if needs_epoch else None,
                # Vertical parameters
                source_vertical_kind=source_vertical_kind if needs_vertical else None,
                target_vertical_kind=target_vertical_kind if needs_vertical else None,
                source_geoid_model=source_geoid if needs_vertical else None,
                target_geoid_model=target_geoid if needs_vertical else None,
                # Align to reference grid
                align_to=self.raster2,
                overwrite=overwrite,
            )
            
            self._transformation_history.append({
                'step': 'combined_transform',
                'needs_epoch': needs_epoch,
                'needs_vertical': needs_vertical,
                'needs_horizontal': needs_horizontal,
                'source_epoch': getattr(self.raster1, 'epoch', None),
                'target_epoch': target_epoch,
                'source_vertical_kind': source_vertical_kind,
                'target_vertical_kind': target_vertical_kind,
                'output_file': current.filename,
            })
            
            if verbose:
                print(f"  ✓ Combined transformation complete", file=sys.stderr)
        else:
            # May still need grid alignment
            if 'grid' in comparison['transformations_needed']:
                if verbose:
                    print(f"\nAligning to reference grid...", file=sys.stderr)
                current = self.raster1.warp_raster(
                    align_to=self.raster2,
                    overwrite=overwrite,
                )
            else:
                current = self.raster1
                if verbose:
                    print(f"\nNo transformations needed.", file=sys.stderr)
        
        # Update metadata to match reference
        current.add_metadata(
            compound_CRS=target_crs,
            geoid_model=target_geoid,
            epoch=target_epoch,
        )
        
        # Cache result
        self._raster1_transformed = current
        
        if verbose:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print("Transformation complete", file=sys.stderr)
            print(f"Output: {current.filename}", file=sys.stderr)
            print(f"{'=' * 60}\n", file=sys.stderr)
        
        return current


    def check_all_match(self) -> Dict[str, Any]:
        """
        Check if all CRS/metadata parameters match between raster1 and raster2.
        
        Returns
        -------
        dict
            Dictionary with match status for each parameter
        """
        result = {
            'compound_crs': {'match': False, 'r1': None, 'r2': None},
            'horizontal_crs': {'match': False, 'r1': None, 'r2': None},
            'vertical_crs': {'match': False, 'r1': None, 'r2': None},
            'geoid': {'match': False, 'r1': None, 'r2': None},
            'epoch': {'match': False, 'r1': None, 'r2': None},
            'vertical_units': {'match': False, 'r1': None, 'r2': None},
            'grid': {'match': False, 'r1': None, 'r2': None},
            'transformations_needed': [],
        }
        
        # CRS comparison
        r1_crs = getattr(self.raster1, 'crs', None)
        r2_crs = getattr(self.raster2, 'crs', None)
        result['horizontal_crs']['r1'] = str(r1_crs)[:50] if r1_crs else None
        result['horizontal_crs']['r2'] = str(r2_crs)[:50] if r2_crs else None
        result['horizontal_crs']['match'] = _crs_equivalent(r1_crs, r2_crs)
        if not result['horizontal_crs']['match']:
            result['transformations_needed'].append('horizontal_crs')
        
        # Vertical CRS / kind
        r1_ortho = getattr(self.raster1, 'is_orthometric', None)
        r2_ortho = getattr(self.raster2, 'is_orthometric', None)
        r1_vert = "orthometric" if r1_ortho else "ellipsoidal" if r1_ortho is False else None
        r2_vert = "orthometric" if r2_ortho else "ellipsoidal" if r2_ortho is False else None
        result['vertical_crs']['r1'] = r1_vert
        result['vertical_crs']['r2'] = r2_vert
        result['vertical_crs']['match'] = r1_vert == r2_vert
        
        # Geoid model
        r1_geoid = getattr(self.raster1, 'current_geoid_model', None) or getattr(self.raster1, 'geoid_model', None)
        r2_geoid = getattr(self.raster2, 'current_geoid_model', None) or getattr(self.raster2, 'geoid_model', None)
        result['geoid']['r1'] = r1_geoid
        result['geoid']['r2'] = r2_geoid
        result['geoid']['match'] = _geoid_equivalent(r1_geoid, r2_geoid)
        
        # Check if vertical datum transformation needed
        if not result['vertical_crs']['match'] or not result['geoid']['match']:
            result['transformations_needed'].append('vertical_datum')
        
        # Epoch
        r1_epoch = getattr(self.raster1, 'epoch', None)
        r2_epoch = getattr(self.raster2, 'epoch', None)
        result['epoch']['r1'] = r1_epoch
        result['epoch']['r2'] = r2_epoch
        if r1_epoch is not None and r2_epoch is not None:
            result['epoch']['match'] = abs(r1_epoch - r2_epoch) < 0.001
        else:
            result['epoch']['match'] = r1_epoch is None and r2_epoch is None
        if not result['epoch']['match'] and r1_epoch is not None and r2_epoch is not None:
            result['transformations_needed'].append('epoch')
        
        # Vertical units
        r1_vunit = getattr(self.raster1, 'vertical_unit', None)
        r2_vunit = getattr(self.raster2, 'vertical_unit', None)
        r1_vunit_name = r1_vunit.name if hasattr(r1_vunit, 'name') else str(r1_vunit) if r1_vunit else "unknown"
        r2_vunit_name = r2_vunit.name if hasattr(r2_vunit, 'name') else str(r2_vunit) if r2_vunit else "unknown"
        result['vertical_units']['r1'] = r1_vunit_name
        result['vertical_units']['r2'] = r2_vunit_name
        units_match, _ = _units_equivalent(r1_vunit, r2_vunit)
        result['vertical_units']['match'] = units_match
        if not units_match:
            result['transformations_needed'].append('vertical_units')

        # Grid alignment
        try:
            with rasterio.open(self.raster1.filename) as src1, rasterio.open(self.raster2.filename) as src2:
                grid_match = (
                    src1.transform == src2.transform and
                    src1.width == src2.width and
                    src1.height == src2.height
                )
                result['grid']['r1'] = f"{src1.width}x{src1.height}, res={src1.res}"
                result['grid']['r2'] = f"{src2.width}x{src2.height}, res={src2.res}"
                result['grid']['match'] = grid_match
                if not grid_match:
                    result['transformations_needed'].append('grid')
        except Exception:
            result['grid']['match'] = False
            result['transformations_needed'].append('grid')
        
        return result


    def print_comparison(self) -> None:
        """Print a human-readable comparison of the two rasters."""
        comparison = self.check_all_match()
        
        print("\n" + "=" * 60)
        print("RasterPair Comparison")
        print("=" * 60)
        
        print(f"\nCompare (raster1):   {Path(self.raster1.filename).name}")
        print(f"Reference (raster2): {Path(self.raster2.filename).name}")
        
        print(f"\n{'Parameter':<20} {'Match':<8} {'R1':<20} {'R2':<20}")
        print("-" * 70)
        
        # Horizontal CRS
        try:
            r1_crs = getattr(self.raster1, 'crs', None)
            r2_crs = getattr(self.raster2, 'crs', None)
            r1_epsg = _ensure_crs_obj(r1_crs).to_epsg() if r1_crs else None
            r2_epsg = _ensure_crs_obj(r2_crs).to_epsg() if r2_crs else None
            r1_str = f"EPSG:{r1_epsg}" if r1_epsg else "Custom"
            r2_str = f"EPSG:{r2_epsg}" if r2_epsg else "Custom"
        except Exception:
            r1_str = "Unknown"
            r2_str = "Unknown"
        match_str = "✓" if comparison['horizontal_crs']['match'] else "✗"
        print(f"{'Horizontal CRS':<20} {match_str:<8} {r1_str:<20} {r2_str:<20}")
        
        # Vertical CRS
        match_str = "✓" if comparison['vertical_crs']['match'] else "✗"
        r1_vert = comparison['vertical_crs']['r1'] or "Unknown"
        r2_vert = comparison['vertical_crs']['r2'] or "Unknown"
        print(f"{'Vertical CRS':<20} {match_str:<8} {r1_vert:<20} {r2_vert:<20}")
        
        # Geoid
        match_str = "✓" if comparison['geoid']['match'] else "✗"
        r1_geoid = comparison['geoid']['r1'] or "None"
        r2_geoid = comparison['geoid']['r2'] or "None"
        r1_geoid = r1_geoid[:18] + ".." if len(r1_geoid) > 20 else r1_geoid
        r2_geoid = r2_geoid[:18] + ".." if len(r2_geoid) > 20 else r2_geoid
        print(f"{'Geoid Model':<20} {match_str:<8} {r1_geoid:<20} {r2_geoid:<20}")
        
        # Epoch
        match_str = "✓" if comparison['epoch']['match'] else "✗"
        r1_epoch = f"{comparison['epoch']['r1']:.4f}" if comparison['epoch']['r1'] else "None"
        r2_epoch = f"{comparison['epoch']['r2']:.4f}" if comparison['epoch']['r2'] else "None"
        print(f"{'Epoch':<20} {match_str:<8} {r1_epoch:<20} {r2_epoch:<20}")
        
        # Units
        match_str = "✓" if comparison['vertical_units']['match'] else "✗"
        print(f"{'Vertical Units':<20} {match_str:<8} {comparison['vertical_units']['r1']:<20} {comparison['vertical_units']['r2']:<20}")
        
        # Grid
        match_str = "✓" if comparison['grid']['match'] else "✗"
        r1_grid = comparison['grid']['r1'] or "Unknown"
        r2_grid = comparison['grid']['r2'] or "Unknown"
        r1_grid = r1_grid[:18] + ".." if len(str(r1_grid)) > 20 else str(r1_grid)
        r2_grid = r2_grid[:18] + ".." if len(str(r2_grid)) > 20 else str(r2_grid)
        print(f"{'Grid':<20} {match_str:<8} {r1_grid:<20} {r2_grid:<20}")
        
        print("-" * 70)
        
        if comparison['transformations_needed']:
            print(f"\nTransformations needed: {', '.join(comparison['transformations_needed'])}")
        else:
            print(f"\n✓ Rasters are fully aligned!")
        
        if hasattr(self, '_transformation_history') and self._transformation_history:
            print(f"\nTransformation steps applied:")
            for i, step in enumerate(self._transformation_history, 1):
                print(f"  {i}. {step.get('step', 'unknown')}")
        
        print("=" * 60 + "\n")
    
    def warp_rasters_to_common_crs(
        self,
        target_raster: str = "raster2",
        target_crs: Optional[Any] = None,
    ) -> "RasterPair":
        """
        Warp rasters to a common CRS.
        
        Legacy method - see transform_raster1_to_match_raster2() for the 
        comprehensive transformation pipeline.
        """
        if target_raster not in ("raster1", "raster2", None):
            raise ValueError("target_raster must be 'raster1', 'raster2', or None.")

        if target_raster is None and target_crs is None:
            raise ValueError("When target_raster is None, target_crs must be provided.")

        if target_raster == "raster1":
            target_crs = self.raster1.crs
            warped_r2 = self.raster2.warp_raster(target_crs=target_crs)
            return RasterPair(self.raster1, warped_r2)

        if target_raster == "raster2":
            target_crs = self.raster2.crs
            warped_r1 = self.raster1.warp_raster(target_crs=target_crs)
            return RasterPair(warped_r1, self.raster2)

        # target_raster is None: warp both to explicit target_crs
        warped_r1 = self.raster1.warp_raster(target_crs=target_crs)
        warped_r2 = self.raster2.warp_raster(target_crs=target_crs)
        return RasterPair(warped_r1, warped_r2)
    
    def raster_difference(
        self,
        compare: str = "raster1",
        reference: str = "raster2",
        align: bool = False,
        method: str = "bilinear",
        resolution: Optional[float] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute raster difference (compare - reference).
        
        Legacy method - see compute_difference() for the enhanced version.
        Note: compute_difference() uses (raster2 - raster1) convention.
        """
        if compare not in ("raster1", "raster2") or reference not in ("raster1", "raster2"):
            raise ValueError("compare and reference must be 'raster1' or 'raster2'.")
        if compare == reference:
            raise ValueError("compare and reference must be different rasters.")

        if compare == "raster1":
            cmp_raster = self.raster1
            ref_raster = self.raster2
        else:
            cmp_raster = self.raster2
            ref_raster = self.raster1

        if align:
            pair_aligned = self.warp_rasters_to_common_crs(target_raster=reference)
            if compare == "raster1":
                cmp_raster = pair_aligned.raster1
                ref_raster = pair_aligned.raster2
            else:
                cmp_raster = pair_aligned.raster2
                ref_raster = pair_aligned.raster1

        resampling_enum = getattr(Resampling, method, Resampling.bilinear)

        with rasterio.open(ref_raster.filename) as src_ref:
            ref_data = src_ref.read(1)
            ref_profile = src_ref.profile
            ref_transform = src_ref.transform
            ref_crs = src_ref.crs

        with rasterio.open(cmp_raster.filename) as src_cmp:
            cmp_data = src_cmp.read(1)
            cmp_transform = src_cmp.transform
            cmp_crs = src_cmp.crs

        if (cmp_crs != ref_crs) or (cmp_data.shape != ref_data.shape) or (cmp_transform != ref_transform):
            cmp_reprojected = np.empty_like(ref_data, dtype="float32")
            reproject(
                source=cmp_data,
                destination=cmp_reprojected,
                src_transform=cmp_transform,
                src_crs=cmp_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling_enum,
            )
        else:
            cmp_reprojected = cmp_data.astype("float32", copy=False)

        nodata_ref = ref_profile.get("nodata", None)
        nodata_cmp = getattr(cmp_raster, "nodata", None)
        if nodata_cmp is None:
            with rasterio.open(cmp_raster.filename) as src_cmp2:
                nodata_cmp = src_cmp2.profile.get("nodata", None)

        mask = np.zeros_like(ref_data, dtype=bool)
        if nodata_ref is not None:
            mask |= (ref_data == nodata_ref)
        if nodata_cmp is not None:
            mask |= (cmp_reprojected == nodata_cmp)

        diff = cmp_reprojected.astype("float64", copy=False) - ref_data.astype("float64", copy=False)
        diff = np.where(mask, np.nan, diff)

        if np.all(np.isnan(diff)):
            stats = {"min": None, "max": None, "mean": None, "std": None}
            histogram = {"hist": [], "bin_edges": []}
        else:
            stats = {
                "min": float(np.nanmin(diff)),
                "max": float(np.nanmax(diff)),
                "mean": float(np.nanmean(diff)),
                "std": float(np.nanstd(diff)),
            }
            hist, bin_edges = np.histogram(diff[~np.isnan(diff)], bins=50)
            histogram = {
                "hist": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
            }

        if output_path is None:
            base, ext = os.path.splitext(ref_raster.filename)
            output_path = f"{base}_diff{ext}"

        diff_profile = ref_profile.copy()
        diff_profile.update({
            "dtype": "float32",
            "nodata": np.nan,
        })

        with rasterio.open(output_path, "w", **diff_profile) as dst:
            dst.write(diff.astype("float32"), 1)

        diff_raster = Raster.from_file(output_path, rtype=getattr(ref_raster, "rtype", None), metadata={})

        if getattr(ref_raster, "crs_history", None) is not None:
            try:
                diff_raster.crs_history = ref_raster.crs_history
                diff_raster.crs_history.record_raster_creation_entry(
                    creation_parameters={
                        "source": "RasterPair.raster_difference",
                        "compare": compare,
                        "reference": reference,
                        "method": method,
                        "resolution": resolution,
                    },
                    description="Difference raster created from two DEMs.",
                )
            except Exception:
                pass

        return {
            "difference_raster": diff_raster,
            "difference_raster_path": output_path,
            "stats": stats,
            "histogram": histogram,
        }
    
    def transform_vertical_datum(
        self,
        source_kind: str,
        target_kind: str,
        source_geoid_model: Optional[str] = None,
        target_geoid_model: Optional[str] = None,
        which: str = "both",
        overwrite: bool = False,
    ) -> "RasterPair":
        """
        Apply vertical datum transformation to one or both rasters.
        """
        if which not in ("raster1", "raster2", "both"):
            raise ValueError("which must be 'raster1', 'raster2', or 'both'.")

        r1 = self.raster1
        r2 = self.raster2

        if which in ("raster1", "both"):
            r1 = r1.warp_raster(
                source_vertical_kind=source_kind,
                target_vertical_kind=target_kind,
                source_geoid_model=source_geoid_model,
                target_geoid_model=target_geoid_model,
                overwrite=overwrite,
            )

        if which in ("raster2", "both"):
            r2 = r2.warp_raster(
                source_vertical_kind=source_kind,
                target_vertical_kind=target_kind,
                source_geoid_model=source_geoid_model,
                target_geoid_model=target_geoid_model,
                overwrite=overwrite,
            )

        return RasterPair(r1, r2)

    def dynamic_epoch_transform(
        self,
        target_epoch: float,
        which: str = "both",
    ) -> "RasterPair":
        """
        Apply epoch transformation to one or both rasters.
        """
        if which not in ("raster1", "raster2", "both"):
            raise ValueError("which must be 'raster1', 'raster2', or 'both'.")

        r1 = self.raster1
        r2 = self.raster2

        if which in ("raster1", "both"):
            r1 = r1.warp_raster(dynamic_target_epoch=target_epoch)
        if which in ("raster2", "both"):
            r2 = r2.warp_raster(dynamic_target_epoch=target_epoch)

        return RasterPair(r1, r2)
    
    def align_rasters(self, target: str = "smaller") -> "RasterPair":
        """
        Ensure the two rasters have matching grids and CRS.
        
        Parameters
        ----------
        target : str, {"smaller", "larger", "raster1", "raster2"}
            Which raster to use as the reference grid
        
        Returns
        -------
        RasterPair
            New RasterPair with aligned rasters
        """
        def get_key(r: Raster):
            try:
                with rasterio.open(r.filename) as src:
                    return (src.crs, src.transform, (src.height, src.width))
            except Exception:
                return None
        
        key1 = get_key(self.raster1)
        key2 = get_key(self.raster2)
        
        if key1 == key2 and key1 is not None:
            return RasterPair(self.raster1, self.raster2)
        
        if target == "smaller":
            size1 = self.raster1.width * self.raster1.height if hasattr(self.raster1, 'width') else 0
            size2 = self.raster2.width * self.raster2.height if hasattr(self.raster2, 'width') else 0
            if size1 < size2:
                new_r2 = self.raster2.warp_raster(align_to=self.raster1)
                return RasterPair(self.raster1, new_r2)
            else:
                new_r1 = self.raster1.warp_raster(align_to=self.raster2)
                return RasterPair(new_r1, self.raster2)
        elif target == "larger":
            size1 = self.raster1.width * self.raster1.height if hasattr(self.raster1, 'width') else 0
            size2 = self.raster2.width * self.raster2.height if hasattr(self.raster2, 'width') else 0
            if size1 > size2:
                new_r2 = self.raster2.warp_raster(align_to=self.raster1)
                return RasterPair(self.raster1, new_r2)
            else:
                new_r1 = self.raster1.warp_raster(align_to=self.raster2)
                return RasterPair(new_r1, self.raster2)
        elif target == "raster1":
            new_r2 = self.raster2.warp_raster(align_to=self.raster1)
            return RasterPair(self.raster1, new_r2)
        elif target == "raster2":
            new_r1 = self.raster1.warp_raster(align_to=self.raster2)
            return RasterPair(new_r1, self.raster2)
        else:
            raise ValueError(f"Invalid target: {target}")

    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def generate_derivative(
        self,
        derivative_type: str = "hillshade",
        azimuth: float = 315.0,
        altitude: float = 45.0,
    ) -> Tuple[Raster, Raster]:
        """
        Generate a derivative product (hillshade, slope, aspect, roughness) for both rasters.

        Parameters
        ----------
        derivative_type : str
            Type of derivative to generate: 'hillshade', 'slope', 'aspect', 'roughness', 'dem'
            Default: 'hillshade'
        azimuth : float
            Sun azimuth angle for hillshade (0-360 degrees, clockwise from north)
            Only used when derivative_type='hillshade'. Default: 315
        altitude : float
            Sun altitude angle for hillshade (0-90 degrees above horizon)
            Only used when derivative_type='hillshade'. Default: 45

        Returns
        -------
        tuple[Raster, Raster]
            (raster1_derivative, raster2_derivative)

        Examples
        --------
        >>> pair = RasterPair(raster1=dem1, raster2=dem2)
        >>> hillshade1, hillshade2 = pair.generate_derivative("hillshade")
        >>> slope1, slope2 = pair.generate_derivative("slope")
        """
        from pathlib import Path
        import os

        derivative_type = derivative_type.lower()

        if derivative_type == "dem":
            # Return original DEMs
            return self.raster1, self.raster2

        # Verify input files exist and are accessible
        for i, raster in enumerate([self.raster1, self.raster2], 1):
            if not os.path.exists(raster.filename):
                raise FileNotFoundError(f"Raster {i} file not found: {raster.filename}")

        # Generate derivatives with proper error handling
        if derivative_type == "hillshade":
            deriv1 = self.raster1.hillshade(azimuth=azimuth, altitude=altitude)
            deriv2 = self.raster2.hillshade(azimuth=azimuth, altitude=altitude)
        elif derivative_type == "slope":
            deriv1 = self.raster1.slope()
            deriv2 = self.raster2.slope()
        elif derivative_type == "aspect":
            deriv1 = self.raster1.aspect()
            deriv2 = self.raster2.aspect()
        elif derivative_type == "roughness":
            deriv1 = self.raster1.roughness()
            deriv2 = self.raster2.roughness()
        else:
            raise ValueError(
                f"Unknown derivative_type: {derivative_type}. "
                f"Must be one of: 'dem', 'hillshade', 'slope', 'aspect', 'roughness'"
            )

        return deriv1, deriv2

    def plot_pair(
        self,
        *,
        derivative: str = "dem",
        azimuth: float = 315.0,
        altitude: float = 45.0,
        overlay: Optional[Raster] = None,
        overlay_alpha: float = 0.5,
        vmin=None,
        vmax=None,
        cmap=None,
        title1="Raster 1",
        title2="Raster 2",
        figsize=(12, 5),
        axes=None,
    ):
        """
        Display the two rasters side-by-side with optional derivative visualization.

        Parameters
        ----------
        derivative : str
            Type of visualization: 'dem', 'hillshade', 'slope', 'aspect', 'roughness'
            Default: 'dem' (original elevation data)
        azimuth : float
            Sun azimuth angle for hillshade (0-360 degrees). Default: 315
        altitude : float
            Sun altitude angle for hillshade (0-90 degrees). Default: 45
        overlay : Raster, optional
            An optional raster to overlay on both plots
        overlay_alpha : float
            Transparency for overlay (0=transparent, 1=opaque)
        vmin, vmax : float, optional
            Color scale limits applied to both rasters
        cmap : str, optional
            Colormap name. If None, auto-selects based on derivative type:
            - dem: 'terrain'
            - hillshade: 'gray'
            - slope: 'YlOrRd'
            - aspect: 'hsv'
            - roughness: 'viridis'
        title1, title2 : str
            Titles for the subplots
        figsize : tuple
            Figure size (only used if axes is None)
        axes : tuple of 2 Axes, optional
            Existing axes to plot on. If None, creates new figure.

        Returns
        -------
        fig, (ax1, ax2) : matplotlib figure and axes (fig is None if axes provided)

        Examples
        --------
        >>> pair = RasterPair(raster1=dem1, raster2=dem2)
        >>> # Plot original DEMs
        >>> fig, axes = pair.plot_pair(derivative='dem')
        >>> # Plot hillshades
        >>> fig, axes = pair.plot_pair(derivative='hillshade', azimuth=315, altitude=45)
        >>> # Plot slopes
        >>> fig, axes = pair.plot_pair(derivative='slope')
        """
        import matplotlib.pyplot as plt

        # Auto-select colormap based on derivative type
        if cmap is None:
            cmap_map = {
                'dem': 'terrain',
                'hillshade': 'gray',
                'slope': 'YlOrRd',
                'aspect': 'hsv',
                'roughness': 'viridis',
            }
            cmap = cmap_map.get(derivative.lower(), 'viridis')

        # Generate derivatives if needed
        derivative_lower = derivative.lower()
        if derivative_lower == "dem":
            plot_raster1 = self.raster1
            plot_raster2 = self.raster2
        else:
            plot_raster1, plot_raster2 = self.generate_derivative(
                derivative_type=derivative_lower,
                azimuth=azimuth,
                altitude=altitude,
            )

        fig = None
        if axes is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            ax1, ax2 = axes

        # Read and plot rasters with proper nodata masking
        with rasterio.open(plot_raster1.filename) as src:
            data1 = src.read(1)
            extent1 = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            nodata1 = src.nodata

        with rasterio.open(plot_raster2.filename) as src:
            data2 = src.read(1)
            extent2 = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            nodata2 = src.nodata

        # Mask nodata values explicitly
        data1 = np.ma.masked_invalid(data1)
        if nodata1 is not None:
            data1 = np.ma.masked_where(data1 == nodata1, data1)

        data2 = np.ma.masked_invalid(data2)
        if nodata2 is not None:
            data2 = np.ma.masked_where(data2 == nodata2, data2)

        im1 = ax1.imshow(data1, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent1)
        ax1.set_title(f"{title1} ({derivative})")
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(data2, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent2)
        ax2.set_title(f"{title2} ({derivative})")
        plt.colorbar(im2, ax=ax2)

        if overlay is not None:
            try:
                with rasterio.open(overlay.filename) as src:
                    overlay_arr = src.read(1)
                overlay_arr = np.ma.masked_invalid(overlay_arr)
                for ax in (ax1, ax2):
                    ax.imshow(overlay_arr, alpha=overlay_alpha, cmap='gray')
            except Exception:
                pass

        if fig is not None:
            plt.tight_layout()

        return fig, (ax1, ax2)
    
    def _extent(self, raster: Raster) -> List[float]:
        """Get extent [left, right, bottom, top] from a raster."""
        with rasterio.open(raster.filename) as src:
            return [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    def _sym_range(self, arr: np.ndarray) -> Tuple[float, float]:
        """
        Compute symmetric color bounds around zero for a difference array.
        """
        valid = ~np.isnan(arr)
        if not np.any(valid):
            return 0.0, 0.0
        v = np.abs(arr[valid])
        m = v.max()
        return -m, m

    def difference_da(self, pair: Optional["RasterPair"] = None):
        """
        Get difference as xarray DataArray.

        Parameters
        ----------
        pair : RasterPair, optional
            If provided, uses that pair. Otherwise uses self.

        Returns
        -------
        xarray.DataArray
            Difference raster as DataArray
        """
        import rioxarray as rio

        target_pair = pair if pair is not None else self
        result = target_pair.compute_difference(verbose=False)
        diff_raster = result['difference_raster']

        # Load as xarray DataArray
        diff_da = rio.open_rasterio(diff_raster.filename, masked=True)
        return diff_da

    def plot_difference(
        self,
        *,
        pair: Optional["RasterPair"] = None,
        diff_path: Optional[Union[Path, str]] = None,
        overlay: Optional[Raster] = None,
        mask_overlay: bool = True,
        cmap: str = "RdBu_r",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        center_zero: bool = True,
        overlay_alpha: float = 0.4,
        title: str = "Difference",
        save_path: Optional[Union[Path, str]] = None,
        dpi: int = 300,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Plot (and optionally save) a difference raster with optional hillshade overlay.

        If *overlay* is provided it is drawn in grayscale beneath the diff map
        (alpha-blended). A combined NoData mask is applied when
        *mask_overlay* is True so only pixels valid in **both** rasters are
        shown.

        Parameters
        ----------
        pair : RasterPair, optional
            RasterPair to compute difference from. If None, uses self.
        diff_path : Path or str, optional
            Path to pre-computed difference raster. Provide exactly one of 'pair' or 'diff_path'.
        overlay : Raster, optional
            Raster to display as grayscale overlay (e.g., hillshade)
        mask_overlay : bool, default=True
            If True, mask pixels that are invalid in either diff or overlay
        cmap : str, default='RdBu_r'
            Colormap for difference (diverging recommended)
        vmin, vmax : float, optional
            Color scale limits. If not provided and center_zero=True, auto-computed symmetrically.
        center_zero : bool, default=True
            If True and vmin/vmax not provided, center colormap at zero
        overlay_alpha : float, default=0.4
            Alpha transparency for overlay layer
        title : str, default='Difference'
            Plot title
        save_path : Path or str, optional
            If provided, save figure to this path
        dpi : int, default=300
            DPI for saved figure
        figsize : tuple, default=(10, 6)
            Figure size in inches

        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        import matplotlib.pyplot as plt

        # Default to using self if neither pair nor diff_path provided
        if pair is None and diff_path is None:
            pair = self

        if (pair is None) == (diff_path is None):
            raise ValueError("Provide exactly one of 'pair' or 'diff_path'")

        # Obtain diff DataArray and drop singleton band dimension
        if pair is not None:
            diff_da = self.difference_da(pair)
            diff_da = diff_da.squeeze(dim=[d for d in diff_da.dims if diff_da[d].size == 1], drop=True)
            base = pair.raster1
        else:
            from raster import Raster
            diff_r = Raster(diff_path)
            # raster data has shape (bands, y, x) or (y, x)
            diff_da = diff_r.data.squeeze()
            base = diff_r

        # Convert to numpy and ensure 2D
        diff_arr = np.squeeze(diff_da.values)
        extent = self._extent(base)

        # Determine colour limits
        if center_zero and vmin is None and vmax is None:
            vmin, vmax = self._sym_range(diff_arr)

        # Prepare overlay alignment & masking
        if overlay is not None:
            ov = overlay
            if ov.shape[1:] != base.shape[1:] or ov.crs != base.crs:
                ov = ov.reproject_to(base)
            ov_arr = np.squeeze(ov.data.values)
            if mask_overlay:
                valid = ~np.logical_or(
                    np.isnan(diff_arr), np.isnan(ov_arr)
                )
                diff_arr = np.where(valid, diff_arr, np.nan)
                ov_arr = np.where(valid, ov_arr, np.nan)
        else:
            ov_arr = None

        # --- plotting ---
        fig, ax = plt.subplots(figsize=figsize)
        cmap_obj = plt.get_cmap(cmap)
        cmap_obj.set_bad(color="none")
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(diff_arr, cmap=cmap_obj, norm=norm, extent=extent)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ Elevation (m)")

        if ov_arr is not None:
            shade_cmap = plt.get_cmap("gray")
            shade_cmap.set_bad(color="none")
            ax.imshow(ov_arr, cmap=shade_cmap, alpha=overlay_alpha, extent=extent)

        ax.axis("off")
        ax.set_title(title)
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

        return fig