"""CRS conversion and transformation utilities.

Provides functions for converting between CRS formats (WKT, PROJ, EPSG),
detecting vertical/horizontal components, and building coordinate transformers.
"""
from typing import Any, Dict, Optional, Union

import numpy as _np
from pyproj import CRS as _CRS
from pyproj.enums import WktVersion
from pyproj.transformer import TransformerGroup as _TransformerGroup

Number = Union[int, float]


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


def crs_to_wkt2_2019(crs: Union[str, _CRS, Dict[str, Any]], pretty: bool = True) -> str:
    """
    Normalize any CRS input to WKT2:2019 text.
    """
    crs_obj = _ensure_crs_obj(crs)
    return crs_obj.to_wkt(WktVersion.WKT2_2019, pretty=pretty)


def wrap_coordinate_metadata_wkt(
    crs: Union[str, _CRS, Dict[str, Any]],
    epoch: Number,
) -> str:
    """
    Produce a WKT2:2019 COORDINATEMETADATA wrapper with EPOCH[Ã¢â‚¬Â¦].

    Returns:
        COORDINATEMETADATA[
          <WKT2:2019 CRS...>,
          EPOCH[<decimal>]
        ]
    """
    wkt2 = crs_to_wkt2_2019(crs, pretty=False)
    return f"COORDINATEMETADATA[{wkt2},EPOCH[{float(epoch)}]]"


def crs_to_projjson(crs: Union[str, _CRS, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize any CRS input to PROJJSON (as a Python dict).
    """
    crs_obj = _ensure_crs_obj(crs)
    return crs_obj.to_json_dict()


def make_coordinate_metadata_projjson(
    crs: Union[str, _CRS, Dict[str, Any]],
    epoch: Number,
) -> Dict[str, Any]:
    """
    Produce a PROJJSON CoordinateMetadata object with an epoch.
    """
    return {
        "type": "CoordinateMetadata",
        "epoch": float(epoch),
        "crs": crs_to_projjson(crs),
    }


def is_orthometric(vertical_crs: Optional[_CRS]) -> Optional[bool]:
    """
    Return True if the vertical CRS represents gravity-related (orthometric) height,
    False if clearly not, or None if it cannot be determined.

    Defensive: if vertical_crs is None or invalid, None is returned instead of raising.
    """
    if not vertical_crs:
        return None

    try:
        v = _ensure_crs_obj(vertical_crs)
    except Exception:
        return None

    if v is None:
        return None

    js = v.to_json_dict()
    axes = (js.get("coordinate_system") or {}).get("axis", []) or []
    axname = axes[0].get("name") if axes else None
    if axname and any(
        token in axname for token in ("Gravity", "gravity", "Orthometric", "orthometric")
    ):
        return True

    name = js.get("name") or v.name or ""
    name_lower = name.lower()
    if any(t in name_lower for t in ("orthometric", "geoid", "gravity", "navd88", "egm96", "egm2008")):
        return True
    if "ellipsoidal height" in name_lower:
        return False

    return None

def is_3d_geographic_crs(crs: Union[str, _CRS, Dict[str, Any]]) -> bool:
    """
    Check if a CRS is a 3D geographic CRS (lat, lon, ellipsoidal height).
    
    These CRS (like EPSG:4979) are NOT compound CRS but contain an integrated
    vertical dimension. They cannot be used directly as the vertical component
    of a CompoundCRS per OGC/ISO standards.
    
    Parameters
    ----------
    crs : str, pyproj.CRS, or dict
        The CRS to check
        
    Returns
    -------
    bool
        True if CRS is 3D geographic with ellipsoidal height
        
    Examples
    --------
    >>> is_3d_geographic_crs("EPSG:4979")  # WGS 84 3D
    True
    >>> is_3d_geographic_crs("EPSG:4326")  # WGS 84 2D
    False
    >>> is_3d_geographic_crs("EPSG:32611")  # UTM (projected)
    False
    """
    try:
        crs_obj = _ensure_crs_obj(crs)
    except Exception:
        return False
    
    # Must be geographic (not projected, not compound, not vertical-only)
    if not crs_obj.is_geographic:
        return False
    if crs_obj.is_compound:
        return False
    
    # Check for 3 axes with the third being height
    cs = crs_obj.coordinate_system
    if cs is None or cs.axis_list is None:
        return False
    
    if len(cs.axis_list) != 3:
        return False
    
    # Third axis should be ellipsoidal height (direction "up")
    third_axis = cs.axis_list[2]
    return third_axis.direction.lower() == "up"


def extract_ellipsoidal_height_as_vertical_crs(
    crs_3d: Union[str, _CRS, Dict[str, Any]],
) -> _CRS:
    """
    Extract the ellipsoidal height component from a 3D geographic CRS
    and return it as a standalone 1D Vertical CRS.
    
    This is necessary because OGC/ISO CompoundCRS requires:
        2D horizontal + 1D vertical
    
    A 3D geographic CRS like EPSG:4979 cannot be used directly as the 
    vertical component of a CompoundCRS—we must synthesize a 1D vertical 
    CRS from its datum information.
    
    Parameters
    ----------
    crs_3d : str, pyproj.CRS, or dict
        A 3D geographic CRS (e.g., EPSG:4979)
        
    Returns
    -------
    pyproj.CRS
        A 1D Vertical CRS representing ellipsoidal height
        
    Raises
    ------
    ValueError
        If the input is not a 3D geographic CRS
        
    Examples
    --------
    >>> vert = extract_ellipsoidal_height_as_vertical_crs("EPSG:4979")
    >>> vert.is_vertical
    True
    >>> "ellipsoidal" in vert.name.lower()
    True
    """
    crs_obj = _ensure_crs_obj(crs_3d)
    
    if not is_3d_geographic_crs(crs_obj):
        raise ValueError(
            f"Expected a 3D geographic CRS, got: {crs_obj.type_name}"
        )
    
    # Extract datum info for the vertical CRS name
    datum_name = "Unknown Datum"
    if crs_obj.datum:
        datum_name = crs_obj.datum.name
    elif hasattr(crs_obj, 'datum_ensemble') and crs_obj.datum_ensemble:
        datum_name = crs_obj.datum_ensemble.name
    
    # Extract the vertical axis unit (default to metre)
    unit_name = "metre"
    unit_factor = 1.0
    cs = crs_obj.coordinate_system
    if cs and cs.axis_list and len(cs.axis_list) >= 3:
        third_axis = cs.axis_list[2]
        if hasattr(third_axis, 'unit_name') and third_axis.unit_name:
            unit_name = third_axis.unit_name
        if hasattr(third_axis, 'unit_conversion_factor') and third_axis.unit_conversion_factor:
            unit_factor = third_axis.unit_conversion_factor
    
    # Build a custom 1D Vertical CRS WKT for ellipsoidal height
    # This follows WKT2:2019 structure
    vert_wkt = f'''VERTCRS["{datum_name} Ellipsoidal Height",
    VDATUM["{datum_name}"],
    CS[vertical,1],
        AXIS["ellipsoidal height (h)",up,
            LENGTHUNIT["{unit_name}",{unit_factor}]]]'''
    
    return _CRS.from_wkt(vert_wkt)

def create_compound_crs(
    horizontal_crs: Union[str, _CRS, Dict[str, Any]],
    vertical_crs: Union[str, _CRS, Dict[str, Any]],
) -> _CRS:
    """
    Create a compound CRS from separate horizontal and vertical CRS components.
    
    Parameters
    ----------
    horizontal_crs : str, pyproj.CRS, or dict
        The horizontal (geographic or projected) CRS component
    vertical_crs : str, pyproj.CRS, or dict
        The vertical CRS component
        
    Returns
    -------
    pyproj.CRS
        A compound CRS combining both components
        
    Examples
    --------
    >>> from pyproj import CRS
    >>> horiz = CRS.from_epsg(32610)  # UTM Zone 10N
    >>> vert = CRS.from_epsg(5703)    # NAVD88 height
    >>> compound = create_compound_crs(horiz, vert)
    >>> compound.is_compound
    True
    """
    horiz_obj = _ensure_crs_obj(horizontal_crs)
    vert_obj = _ensure_crs_obj(vertical_crs)
    
    # Create compound CRS using WKT concatenation
    horiz_wkt = horiz_obj.to_wkt()
    vert_wkt = vert_obj.to_wkt()
    
    # Build compound WKT
    compound_wkt = f'COMPOUNDCRS["{horiz_obj.name} + {vert_obj.name}",{horiz_wkt},{vert_wkt}]'
    
    return _CRS.from_wkt(compound_wkt)


def parse_crs_components(crs: Any) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse a CRS into its compound, horizontal, and vertical components.
    
    Parameters
    ----------
    crs : rasterio.crs.CRS, pyproj.CRS, str, or None
        The CRS to parse
        
    Returns
    -------
    tuple[Optional[str], Optional[str], Optional[str]]
        (compound_crs_wkt, horizontal_crs_wkt, vertical_crs_wkt)
        
    Logic
    -----
    1. If CRS is compound:
       - compound_crs = full WKT
       - horizontal_crs = WKT of first sub-CRS (horizontal component)
       - vertical_crs = WKT of second sub-CRS (vertical component)
       
    2. If CRS is vertical only:
       - compound_crs = None
       - horizontal_crs = None
       - vertical_crs = CRS WKT
       
    3. If CRS is horizontal only (geographic or projected):
       - compound_crs = None
       - horizontal_crs = CRS WKT
       - vertical_crs = None
       
    4. If CRS is None:
       - All return None
       
    Examples
    --------
    >>> from pyproj import CRS
    >>> utm_crs = CRS.from_epsg(32610)
    >>> compound, horiz, vert = parse_crs_components(utm_crs)
    >>> print(f"Horizontal only: {horiz is not None and vert is None}")
    True
    """
    if crs is None:
        return None, None, None
    
    try:
        # Convert to pyproj CRS for consistent API
        if not isinstance(crs, _CRS):
            pyproj_crs = _CRS.from_user_input(crs)
        else:
            pyproj_crs = crs
            
    except Exception:
        # If conversion fails, try to get WKT directly from rasterio CRS
        try:
            if hasattr(crs, 'wkt'):
                # Assume it's horizontal-only since we can't parse it
                return None, crs.wkt, None
            else:
                return None, str(crs), None
        except Exception:
            return None, None, None
    
    # Case 1: Compound CRS (has both horizontal and vertical components)
    if pyproj_crs.is_compound:
        compound_wkt = pyproj_crs.to_wkt()
        
        # Extract sub-CRS components
        sub_crs_list = getattr(pyproj_crs, 'sub_crs_list', None) or []
        
        # First component is horizontal, second is vertical
        horizontal_component = sub_crs_list[0] if len(sub_crs_list) >= 1 else None
        vertical_component = sub_crs_list[1] if len(sub_crs_list) >= 2 else None
        
        horizontal_wkt = horizontal_component.to_wkt() if horizontal_component else None
        vertical_wkt = vertical_component.to_wkt() if vertical_component else None
        
        return compound_wkt, horizontal_wkt, vertical_wkt
    
    # Case 2: Vertical CRS only
    if getattr(pyproj_crs, 'is_vertical', False):
        vertical_wkt = pyproj_crs.to_wkt()
        return None, None, vertical_wkt
    
    # Case 3: Horizontal CRS only (geographic or projected)
    # This is the most common case for rasters
    horizontal_wkt = pyproj_crs.to_wkt()
    return None, horizontal_wkt, None


def transformer_with_epoch(
    src_crs: Union[str, _CRS, Dict[str, Any]],
    dst_crs: Union[str, _CRS, Dict[str, Any]],
    src_epoch: Optional[Number] = None,
    dst_epoch: Optional[Number] = None,
):
    """
    Get a pyproj Transformer that is optionally aware of source/target epochs.
    
    Uses Transformer.from_crs() with epoch parameters (pyproj >= 3.4.0).
    Falls back to non-epoch transform if epochs aren't supported.
    """
    from pyproj import Transformer
    
    src = _ensure_crs_obj(src_crs)
    dst = _ensure_crs_obj(dst_crs)
    
    # Try epoch-aware transform first (pyproj >= 3.4.0)
    # Correct parameter names are source_crs_epoch and target_crs_epoch
    try:
        transformer = Transformer.from_crs(
            src,
            dst,
            always_xy=True,
            source_crs_epoch=float(src_epoch) if src_epoch is not None else None,
            target_crs_epoch=float(dst_epoch) if dst_epoch is not None else None,
        )
        return transformer
    except TypeError:
        # Fallback for older pyproj versions without epoch support
        pass
    
    # Fallback: standard transform without epoch awareness
    return Transformer.from_crs(src, dst, always_xy=True)


# ---------------------------------------------------------------------------
# Unit scaling helpers
# ---------------------------------------------------------------------------

_HORIZONTAL_UNIT_FACTORS = {
    "metre": 1.0,
    "meter": 1.0,
    "m": 1.0,
    "kilometre": 1000.0,
    "km": 1000.0,
    "foot": 0.3048,
    "feet": 0.3048,
    "us_survey_foot": 1200.0 / 3937.0,
    "ft": 0.3048,
}

_VERTICAL_UNIT_FACTORS = _HORIZONTAL_UNIT_FACTORS  # same physical units


def _unit_factor_to_meters(unit_name: Optional[str]) -> Optional[float]:
    if not unit_name:
        return None
    key = unit_name.lower()
    return _HORIZONTAL_UNIT_FACTORS.get(key)


def horizontal_unit_scale(src_crs: Any, target_unit: str) -> Optional[float]:
    """
    Return scale factor to go from source horizontal units to target_unit.
    (e.g. metres -> feet => factor ~ 3.28084)

    If units are unknown, returns None.
    """
    crs = _ensure_crs_obj(src_crs)
    src_units = None
    if crs.coordinate_system and crs.coordinate_system.axis_list:
        src_units = crs.coordinate_system.axis_list[0].unit_name

    src_m = _unit_factor_to_meters(src_units)
    tgt_m = _unit_factor_to_meters(target_unit)

    if src_m is None or tgt_m is None:
        return None

    return src_m / tgt_m


def vertical_unit_scale(src_crs: Any, target_unit: str) -> Optional[float]:
    """
    Return scale factor to go from source vertical units to target_unit.
    """
    crs = _ensure_crs_obj(src_crs)
    src_units = None
    if crs.coordinate_system and crs.coordinate_system.axis_list:
        src_units = crs.coordinate_system.axis_list[-1].unit_name

    src_m = _unit_factor_to_meters(src_units)
    tgt_m = _unit_factor_to_meters(target_unit)

    if src_m is None or tgt_m is None:
        return None

    return src_m / tgt_m


# ---------------------------------------------------------------------------
# Vertical datum / dynamic helpers at geometry level
# ---------------------------------------------------------------------------


def apply_vertical_datum_transform(
    z: "_np.ndarray",
    source_vertical_crs: Any,
    target_vertical_crs: Any,
    geoid_model: Optional[str] = None,
) -> "_np.ndarray":
    """
    Vertical datum conversion (orthometric <-> ellipsoidal, geoid A -> geoid B).

    This function assumes that PROJ knows how to transform between the two
    vertical CRSs (e.g., via appropriate geoid grids). It uses a pyproj
    transformer on z-values alone.

    If no valid transformation exists, z is returned unchanged.
    """
    try:
        src = _ensure_crs_obj(source_vertical_crs)
        dst = _ensure_crs_obj(target_vertical_crs)
    except Exception:
        return z

    try:
        tg = _TransformerGroup(src, dst, always_xy=True)
        if not tg.transformers:
            return z
        transformer = tg.transformers[0]

        dummy_x = _np.zeros_like(z, dtype="float64")
        dummy_y = _np.zeros_like(z, dtype="float64")
        _, _, z_out = transformer.transform(dummy_x, dummy_y, z.astype("float64"))
        return _np.asarray(z_out, dtype=z.dtype)
    except Exception:
        return z


def apply_dynamic_transform(
    x: "_np.ndarray",
    y: "_np.ndarray",
    z: Optional["_np.ndarray"],
    src_crs: Any,
    dst_crs: Any,
    src_epoch: Optional[Number],
    dst_epoch: Optional[Number],
):
    """
    Apply a dynamic (epoch-aware) transformation using transformer_with_epoch.

    If the CRS is not actually dynamic, this falls back to a standard transform.
    """
    transformer = transformer_with_epoch(src_crs, dst_crs, src_epoch=src_epoch, dst_epoch=dst_epoch)

    if z is None:
        x_out, y_out = transformer.transform(x, y)
        return x_out, y_out, None
    else:
        x_out, y_out, z_out = transformer.transform(x, y, z)
        return x_out, y_out, z_out