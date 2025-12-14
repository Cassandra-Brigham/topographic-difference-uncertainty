"""Raster loading, metadata extraction, and transformation.

Provides the Raster class for loading GeoTIFF files, extracting CRS and
metadata, and performing coordinate transformations and reprojections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path
import sys

import rasterio

from .geoid_utils import parse_geoid_info
from .crs_history import CRSHistory
from .crs_utils import (
    _ensure_crs_obj,
    apply_dynamic_transform,
    parse_crs_components,
    is_3d_geographic_crs,
    extract_ellipsoidal_height_as_vertical_crs,
)
from .time_utils import _parse_epoch_string_to_decimal
from .unit_utils import (
    UnitInfo,
    UNKNOWN_UNIT,
    METER,
    FOOT,
    US_SURVEY_FOOT,
    get_horizontal_unit,
    get_vertical_unit,
    get_crs_units,
    convert_length,
    convert_to_meters,
    get_conversion_factor,
    lookup_unit,
    parse_unit_string,
    format_value_with_unit,
    describe_unit,
    # Backward-compatible functions
    horizontal_unit_scale,
    vertical_unit_scale,
)


def has_rasterio() -> bool:
    try:
        import rasterio  # noqa: F401
        return True
    except Exception:
        return False


def has_scipy() -> bool:
    try:
        import scipy  # noqa: F401
        return True
    except Exception:
        return False


def parse_time_info(tags: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse time-related information from raster metadata tags.
    """
    from datetime import datetime

    time_info: Dict[str, Any] = {}

    if not tags:
        return time_info

    # Normalize keys to lowercase for case-insensitive matching
    tags_lower = {k.lower(): v for k, v in tags.items()}

    # 1. Check for explicit epoch value (already in decimal years)
    if "epoch" in tags_lower:
        try:
            time_info["epoch"] = float(tags_lower["epoch"])
            time_info["epoch_source"] = "metadata_epoch_field"
        except (ValueError, TypeError):
            pass

    # 2. Check for TIFFTAG_DATETIME (standard GeoTIFF format: "YYYY:MM:DD HH:MM:SS")
    datetime_candidates = [
        "tifftag_datetime",
        "datetime",
        "date_time",
        "acquisition_datetime",
        "acq_datetime",
    ]

    for key in datetime_candidates:
        if key in tags_lower:
            datetime_str = str(tags_lower[key]).strip()
            time_info["datetime_str"] = datetime_str

            # Try GeoTIFF format first
            try:
                dt = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
                if "epoch" not in time_info:
                    from time_utils import _datetime_to_decimal_year

                    time_info["epoch"] = _datetime_to_decimal_year(dt)
                    time_info["epoch_source"] = f"parsed_{key}"
                break
            except ValueError:
                # Try ISO format
                try:
                    dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
                    if dt.tzinfo is not None:
                        dt = dt.replace(tzinfo=None)
                    if "epoch" not in time_info:
                        from time_utils import _datetime_to_decimal_year

                        time_info["epoch"] = _datetime_to_decimal_year(dt)
                        time_info["epoch_source"] = f"parsed_{key}_iso"
                    break
                except (ValueError, AttributeError):
                    pass

    # 3. Check for separate date and time fields
    date_candidates = [
        "acquisition_date",
        "acq_date",
        "date",
        "image_date",
        "sensing_date",
        "date_acquired",
        "scene_date",
        "capture_date",
        "observation_date",
    ]
    time_candidates = [
        "acquisition_time",
        "acq_time",
        "time",
        "image_time",
        "sensing_time",
    ]

    date_str = None
    for key in date_candidates:
        if key in tags_lower:
            date_str = str(tags_lower[key]).strip()
            time_info["date_str"] = date_str
            break

    time_str = None
    for key in time_candidates:
        if key in tags_lower:
            time_str = str(tags_lower[key]).strip()
            time_info["time_str"] = time_str
            break

    # If we have a date string, try to parse it
    if date_str and "epoch" not in time_info:
        try:
            epoch_val = _parse_epoch_string_to_decimal(date_str)
            if isinstance(epoch_val, tuple):
                time_info["epoch"] = epoch_val[0]
                time_info["epoch_range"] = epoch_val
            else:
                time_info["epoch"] = epoch_val
            time_info["epoch_source"] = "parsed_date_field"
        except Exception:
            pass

    # 4. Check for individual year/month/day components
    if "year" in tags_lower:
        time_info["year"] = tags_lower["year"]
    if "month" in tags_lower:
        time_info["month"] = tags_lower["month"]
    if "day" in tags_lower:
        time_info["day"] = tags_lower["day"]

    # Try to construct epoch from year/month/day if we have them
    if "epoch" not in time_info and "year" in time_info:
        try:
            year = int(time_info["year"])
            month = int(time_info.get("month", 1))
            day = int(time_info.get("day", 1))
            dt = datetime(year, month, day)
            from time_utils import _datetime_to_decimal_year

            time_info["epoch"] = _datetime_to_decimal_year(dt)
            time_info["epoch_source"] = "constructed_from_ymd"
        except (ValueError, TypeError):
            pass

    # 5. Check for GPS time
    if "gps_time" in tags_lower or "gpstime" in tags_lower:
        gps_key = "gps_time" if "gps_time" in tags_lower else "gpstime"
        try:
            gps_seconds = float(tags_lower[gps_key])
            if "epoch" not in time_info:
                from time_utils import gps_seconds_to_decimal_year_utc

                time_info["epoch"] = gps_seconds_to_decimal_year_utc(gps_seconds)
                time_info["epoch_source"] = "gps_time"
                time_info["gps_seconds"] = gps_seconds
        except (ValueError, TypeError):
            pass

    return time_info


def get_units_from_rasterio(src) -> tuple[UnitInfo, UnitInfo]:
    """
    Extract units from an open rasterio dataset as UnitInfo objects.
    
    Parameters
    ----------
    src : rasterio.DatasetReader
        Open rasterio dataset
        
    Returns
    -------
    tuple[UnitInfo, UnitInfo]
        (horizontal_unit, vertical_unit)
    """
    horiz_unit = UNKNOWN_UNIT
    vert_unit = UNKNOWN_UNIT
    
    if src.crs is not None:
        try:
            horiz_unit = get_horizontal_unit(src.crs)
            vert_unit = get_vertical_unit(src.crs)
        except Exception:
            pass
    
    return (horiz_unit, vert_unit)


class DependencyMissingError(ImportError):
    def __init__(self, package: str, where: str = ""):
        msg = f"Required dependency '{package}' is not available"
        if where:
            msg += f" (needed by {where})"
        super().__init__(msg)


@dataclass
class Raster:
    """
    Raster with CRS and metadata tracking.
    
    Attributes
    ----------
    filename : str
        Path to the raster file.
    horizontal_unit : UnitInfo
        Full unit metadata for horizontal coordinates.
    vertical_unit : UnitInfo
        Full unit metadata for vertical (Z/elevation) values.
    original_horizontal_units : str
        Display name of original horizontal units (backward compatible).
    original_vertical_units : str
        Display name of original vertical units (backward compatible).
    current_horizontal_units : str
        Display name of current horizontal units (backward compatible).
    current_vertical_units : str
        Display name of current vertical units (backward compatible).
    """

    filename: str
    _data: Optional[Any] = None  # Cached rioxarray DataArray

    def __post_init__(self):
        """Initialize dynamic attributes that aren't dataclass fields."""
        if not hasattr(self, "crs"):
            self.crs = None
        # Use private attributes for CRS components, managed by properties
        if not hasattr(self, "_original_compound_crs"):
            self._original_compound_crs = None
        if not hasattr(self, "_original_horizontal_crs"):
            self._original_horizontal_crs = None
        if not hasattr(self, "_original_vertical_crs"):
            self._original_vertical_crs = None
        if not hasattr(self, "_current_compound_crs"):
            self._current_compound_crs = None
        if not hasattr(self, "_current_horizontal_crs"):
            self._current_horizontal_crs = None
        if not hasattr(self, "_current_vertical_crs"):
            self._current_vertical_crs = None
        if not hasattr(self, "original_proj_string"):
            self.original_proj_string = None
        if not hasattr(self, "original_geoid_model"):
            self.original_geoid_model = None
        if not hasattr(self, "current_proj_string"):
            self.current_proj_string = None
        if not hasattr(self, "current_geoid_model"):
            self.current_geoid_model = None
        if not hasattr(self, "transform"):
            self.transform = None
        if not hasattr(self, "bounds"):
            self.bounds = None
        if not hasattr(self, "width"):
            self.width = None
        if not hasattr(self, "height"):
            self.height = None
        if not hasattr(self, "metadata"):
            self.metadata = {}
        if not hasattr(self, "epoch"):
            self.epoch = None
        if not hasattr(self, "epoch_start"):
            self.epoch_start = None
        if not hasattr(self, "epoch_end"):
            self.epoch_end = None
        if not hasattr(self, 'is_orthometric'):
            self.is_orthometric = None 
        
        # Unit info objects - enhanced
        if not hasattr(self, "original_horizontal_unit"):
            self.original_horizontal_unit: UnitInfo = UNKNOWN_UNIT
        if not hasattr(self, "original_vertical_unit"):
            self.original_vertical_unit: UnitInfo = UNKNOWN_UNIT
        if not hasattr(self, "current_horizontal_unit"):
            self.current_horizontal_unit: UnitInfo = UNKNOWN_UNIT
        if not hasattr(self, "current_vertical_unit"):
            self.current_vertical_unit: UnitInfo = UNKNOWN_UNIT
        
        # Backward compatible string properties
        if not hasattr(self, "original_vertical_units"):
            self.original_vertical_units = None
        if not hasattr(self, "original_horizontal_units"):
            self.original_horizontal_units = None
        if not hasattr(self, "current_vertical_units"):
            self.current_vertical_units = None
        if not hasattr(self, "current_horizontal_units"):
            self.current_horizontal_units = None
    
    # =========================================================================
    # CRS Component Properties - Auto-sync compound/horizontal/vertical
    # =========================================================================
    
    @property
    def original_compound_crs(self) -> Optional[str]:
        """Get the original compound CRS WKT."""
        return self._original_compound_crs
    
    @original_compound_crs.setter
    def original_compound_crs(self, value: Optional[str]):
        """
        Set compound CRS and auto-extract horizontal/vertical components.
        
        When you set a compound CRS, the horizontal and vertical components
        are automatically extracted from it.
        """
        self._original_compound_crs = value
        
        if value is not None:
            # Parse and extract components
            from crs_utils import parse_crs_components
            _, horiz_wkt, vert_wkt = parse_crs_components(value)
            self._original_horizontal_crs = horiz_wkt
            self._original_vertical_crs = vert_wkt
        # If value is None, we don't clear horizontal/vertical automatically
        # since they might have been set independently
    
    @property
    def original_horizontal_crs(self) -> Optional[str]:
        """Get the original horizontal CRS WKT."""
        return self._original_horizontal_crs
    
    @original_horizontal_crs.setter
    def original_horizontal_crs(self, value: Optional[str]):
        """
        Set horizontal CRS and auto-create compound if vertical exists.
        
        When both horizontal and vertical CRS are set, a compound CRS
        is automatically created from them.
        """
        self._original_horizontal_crs = value
        self._update_original_compound_from_components()
    
    @property
    def original_vertical_crs(self) -> Optional[str]:
        """Get the original vertical CRS WKT."""
        return self._original_vertical_crs
    
    @original_vertical_crs.setter
    def original_vertical_crs(self, value: Optional[str]):
        """
        Set vertical CRS and auto-create compound if horizontal exists.
        
        When both horizontal and vertical CRS are set, a compound CRS
        is automatically created from them.
        """
        self._original_vertical_crs = value
        self._update_original_compound_from_components()
    
    def _update_original_compound_from_components(self):
        """
        Update compound CRS from horizontal and vertical components.
        
        This is called internally when horizontal or vertical CRS is set.
        """
        if self._original_horizontal_crs and self._original_vertical_crs:
            try:
                from crs_utils import create_compound_crs
                compound_obj = create_compound_crs(
                    self._original_horizontal_crs,
                    self._original_vertical_crs
                )
                self._original_compound_crs = compound_obj.to_wkt()
            except Exception:
                # If compound creation fails, leave compound as-is
                pass
        elif self._original_horizontal_crs and not self._original_vertical_crs:
            # Only horizontal - compound should be None
            self._original_compound_crs = None
        elif self._original_vertical_crs and not self._original_horizontal_crs:
            # Only vertical - compound should be None
            self._original_compound_crs = None
    
    @property
    def current_compound_crs(self) -> Optional[str]:
        """Get the current compound CRS WKT."""
        return self._current_compound_crs
    
    @current_compound_crs.setter
    def current_compound_crs(self, value: Optional[str]):
        """
        Set compound CRS and auto-extract horizontal/vertical components.
        
        When you set a compound CRS, the horizontal and vertical components
        are automatically extracted from it.
        """
        self._current_compound_crs = value
        
        if value is not None:
            # Parse and extract components
            from crs_utils import parse_crs_components
            _, horiz_wkt, vert_wkt = parse_crs_components(value)
            self._current_horizontal_crs = horiz_wkt
            self._current_vertical_crs = vert_wkt
    
    @property
    def current_horizontal_crs(self) -> Optional[str]:
        """Get the current horizontal CRS WKT."""
        return self._current_horizontal_crs
    
    @current_horizontal_crs.setter
    def current_horizontal_crs(self, value: Optional[str]):
        """
        Set horizontal CRS and auto-create compound if vertical exists.
        
        When both horizontal and vertical CRS are set, a compound CRS
        is automatically created from them.
        """
        self._current_horizontal_crs = value
        self._update_current_compound_from_components()
    
    @property
    def current_vertical_crs(self) -> Optional[str]:
        """Get the current vertical CRS WKT."""
        return self._current_vertical_crs
    
    @current_vertical_crs.setter
    def current_vertical_crs(self, value: Optional[str]):
        """
        Set vertical CRS and auto-create compound if horizontal exists.
        
        When both horizontal and vertical CRS are set, a compound CRS
        is automatically created from them.
        """
        self._current_vertical_crs = value
        self._update_current_compound_from_components()
    
    def _update_current_compound_from_components(self):
        """
        Update compound CRS from horizontal and vertical components.
        
        This is called internally when horizontal or vertical CRS is set.
        """
        if self._current_horizontal_crs and self._current_vertical_crs:
            try:
                from crs_utils import create_compound_crs
                compound_obj = create_compound_crs(
                    self._current_horizontal_crs,
                    self._current_vertical_crs
                )
                self._current_compound_crs = compound_obj.to_wkt()
            except Exception:
                # If compound creation fails, leave compound as-is
                pass
        elif self._current_horizontal_crs and not self._current_vertical_crs:
            # Only horizontal - compound should be None
            self._current_compound_crs = None
        elif self._current_vertical_crs and not self._current_horizontal_crs:
            # Only vertical - compound should be None
            self._current_compound_crs = None
    
    @property
    def original_full_crs(self) -> Optional[str]:
        """Return best available CRS (compound > horiz > vert)."""
        if self.original_compound_crs:
            return self.original_compound_crs
        elif self.original_horizontal_crs:
            return self.original_horizontal_crs
        elif self.original_vertical_crs:
            return self.original_vertical_crs
        return None
    
    @property
    def current_full_crs(self) -> Optional[str]:
        """
        Get the most complete current CRS representation available.
        
        Returns compound CRS if both horizontal and vertical exist,
        otherwise horizontal only, otherwise vertical only.
        
        This is useful when you want a CRS that's always populated
        regardless of whether it's truly compound.
        """
        if self.current_compound_crs:
            return self.current_compound_crs
        elif self.current_horizontal_crs:
            return self.current_horizontal_crs
        elif self.current_vertical_crs:
            return self.current_vertical_crs
        return None

    # =========================================================================
    # Unit convenience properties
    # =========================================================================
    
    @property
    def horizontal_unit(self) -> UnitInfo:
        """Get the current horizontal unit."""
        return self.current_horizontal_unit
    
    @horizontal_unit.setter
    def horizontal_unit(self, value: UnitInfo):
        """Set the current horizontal unit."""
        self.current_horizontal_unit = value
        self.current_horizontal_units = value.display_name
    
    @property
    def vertical_unit(self) -> UnitInfo:
        """Get the current vertical unit."""
        return self.current_vertical_unit
    
    @vertical_unit.setter
    def vertical_unit(self, value: UnitInfo):
        """Set the current vertical unit."""
        self.current_vertical_unit = value
        self.current_vertical_units = value.display_name

    @property
    def data(self):
        """Return the raster data as a rioxarray DataArray (lazy load)."""
        if self._data is None:
            try:
                import rioxarray as rio

                self._data = rio.open_rasterio(self.filename, masked=True)
                if "band" in self._data.dims and self._data.sizes.get("band", 1) == 1:
                    self._data = self._data.squeeze("band", drop=True)
            except ImportError:
                raise DependencyMissingError("rioxarray", "Raster.data property")
        return self._data

    @property
    def shape(self) -> Optional[tuple]:
        """Return the (band, row, column) dimensions of the raster."""
        try:
            return self.data.shape
        except Exception:
            if has_rasterio():
                with rasterio.open(self.filename) as src:
                    return (src.count, src.height, src.width)
            return None

    @property
    def path(self) -> Path:
        """Return Path object for the filename."""
        return Path(self.filename)

    @property
    def resolution(self) -> Optional[float]:
        """
        Return the raster resolution (pixel size) in map units.
        """
        if self.transform is not None:
            return abs(self.transform.a)
        if has_rasterio():
            try:
                with rasterio.open(self.filename) as src:
                    return abs(src.transform.a)
            except Exception:
                pass
        return None

    # =========================================================================
    # Unit conversion methods
    # =========================================================================
    
    def convert_values_to_meters(self, values) -> Any:
        """
        Convert elevation/Z values from the raster's vertical unit to meters.
        
        Parameters
        ----------
        values : array-like
            Elevation values in the raster's vertical unit
            
        Returns
        -------
        array-like
            Values converted to meters
            
        Examples
        --------
        >>> raster = Raster.from_file("dem.tif")
        >>> z_meters = raster.convert_values_to_meters(raster.data.values)
        """
        import numpy as np
        
        if self.current_vertical_unit.name == "unknown":
            import warnings
            warnings.warn(
                "Vertical unit is unknown, assuming meters. "
                "Use raster.vertical_unit = lookup_unit('foot') to set manually."
            )
            return values
        
        return convert_length(np.asarray(values), self.current_vertical_unit, METER)
    
    def convert_values_from_meters(self, values_meters) -> Any:
        """
        Convert elevation values from meters to the raster's vertical unit.
        
        Parameters
        ----------
        values_meters : array-like
            Values in meters
            
        Returns
        -------
        array-like
            Values in the raster's vertical unit
        """
        import numpy as np
        
        if self.current_vertical_unit.name == "unknown":
            import warnings
            warnings.warn("Vertical unit is unknown, assuming meters.")
            return values_meters
        
        return convert_length(np.asarray(values_meters), METER, self.current_vertical_unit)
    
    def get_value_conversion_factor(self, target_unit: str = "meter") -> float:
        """
        Get the factor to multiply elevation values by to convert to target unit.
        
        Parameters
        ----------
        target_unit : str
            Target unit (default: "meter")
            
        Returns
        -------
        float
            Conversion factor
            
        Examples
        --------
        >>> raster.get_value_conversion_factor("meter")
        0.3048  # if vertical unit is feet
        """
        target = lookup_unit(target_unit)
        if target is None:
            raise ValueError(f"Unknown unit: {target_unit}")
        
        if self.current_vertical_unit.name == "unknown":
            return 1.0
        
        return get_conversion_factor(self.current_vertical_unit, target)
    
    def get_resolution_in_meters(self) -> Optional[float]:
        """
        Get the raster resolution converted to meters.
        
        Returns
        -------
        float or None
            Resolution in meters, or None if resolution or unit unknown
        """
        res = self.resolution
        if res is None:
            return None
        
        if self.current_horizontal_unit.name == "unknown":
            return res  # Assume meters if unknown
        
        if self.current_horizontal_unit.category != "linear":
            # Geographic CRS (degrees) - can't directly convert
            return None
        
        return res * self.current_horizontal_unit.to_base_factor
    
    def are_units_metric(self) -> tuple[bool, bool]:
        """
        Check if horizontal and vertical units are metric (meter-based).
        
        Returns
        -------
        tuple[bool, bool]
            (horizontal_is_metric, vertical_is_metric)
        """
        metric_units = {"meter", "kilometer", "centimeter", "millimeter"}
        h_metric = self.current_horizontal_unit.name in metric_units
        v_metric = self.current_vertical_unit.name in metric_units
        return h_metric, v_metric
    
    def set_units(
        self,
        horizontal_unit: Optional[str] = None,
        vertical_unit: Optional[str] = None,
    ) -> None:
        """
        Manually set the horizontal and/or vertical units.
        
        Use this when the units aren't correctly detected from the file metadata.
        
        Parameters
        ----------
        horizontal_unit : str, optional
            Horizontal unit name (e.g., "meter", "us_survey_foot")
        vertical_unit : str, optional
            Vertical unit name
            
        Examples
        --------
        >>> raster.set_units(vertical_unit="us_survey_foot")
        >>> raster.set_units(horizontal_unit="foot", vertical_unit="foot")
        """
        if horizontal_unit is not None:
            unit = lookup_unit(horizontal_unit)
            if unit is None:
                raise ValueError(f"Unknown horizontal unit: {horizontal_unit}")
            self.current_horizontal_unit = unit
            self.current_horizontal_units = unit.display_name
        
        if vertical_unit is not None:
            unit = lookup_unit(vertical_unit)
            if unit is None:
                raise ValueError(f"Unknown vertical unit: {vertical_unit}")
            self.current_vertical_unit = unit
            self.current_vertical_units = unit.display_name
    
    def convert_vertical_units(
        self,
        target_units: str,
        output_path: Optional[str] = None,
        overwrite: bool = True,
    ) -> "Raster":
        """
        Convert elevation values to a different vertical unit.
        
        Creates a new raster file with the elevation values scaled to the target unit.
        This is useful when comparing DEMs that have different vertical units.
        
        Parameters
        ----------
        target_units : str
            Target vertical unit (e.g., "meter", "foot", "us_survey_foot")
        output_path : str, optional
            Output file path. If None, generates a path based on input filename.
        overwrite : bool, default True
            Whether to overwrite existing output file.
            
        Returns
        -------
        Raster
            New Raster object with converted elevation values.
            
        Raises
        ------
        ValueError
            If the current vertical unit is unknown and cannot be converted.
            
        Notes
        -----
        This method physically changes the elevation values in the raster data.
        It does NOT change the CRS (use warp_raster for CRS transformations).
        
        Examples
        --------
        >>> # Convert from US survey feet to meters
        >>> raster_ft = Raster.from_file("dem_feet.tif")
        >>> raster_ft.set_units(vertical_unit="us_survey_foot")
        >>> raster_m = raster_ft.convert_vertical_units("meter")
        >>> print(raster_m.vertical_unit)  # "metre (m)"
        """
        import os
        import numpy as np
        import rasterio
        from pathlib import Path

        # Get target unit info
        target_unit = lookup_unit(target_units)
        if target_unit is None:
            raise ValueError(f"Unknown target unit: {target_units}")
        
        # Check if source unit is known
        if self.current_vertical_unit.name == "unknown":
            raise ValueError(
                "Cannot convert units: current vertical unit is unknown. "
                "Use set_units() to specify the current unit first."
            )
        
        # Get conversion factor
        source_unit = self.current_vertical_unit
        if source_unit.name == target_unit.name:
            # Units are the same, just return a copy
            import warnings
            warnings.warn(f"Source and target units are both {source_unit.name}, no conversion needed")
            return self
        
        factor = get_conversion_factor(source_unit, target_unit)
        
        # Generate output path if not specified
        if output_path is None:
            stem = Path(self.filename).stem
            suffix = Path(self.filename).suffix
            parent = Path(self.filename).parent
            output_path = str(parent / f"{stem}_units_{target_unit.name}{suffix}")
        
        # Check if output exists
        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(f"Output file exists and overwrite=False: {output_path}")
        
        # Read source data, apply conversion, write to new file
        with rasterio.open(self.filename) as src:
            data = src.read(1).astype('float32')
            profile = src.profile.copy()
            nodata = src.nodata
            
            # Create mask for valid data
            valid_mask = ~np.isnan(data) & ~np.isinf(data)
            if nodata is not None:
                valid_mask &= (data != nodata)
            
            # Apply conversion factor to valid pixels only
            converted = data.copy()
            converted[valid_mask] = data[valid_mask] * factor
            
            # Update profile
            profile.update(dtype='float32')
            
            # Write converted raster
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(converted, 1)
        
        # Load the new raster
        result = Raster.from_file(output_path, rtype=self.rtype, metadata=self.metadata)
        
        # Copy metadata from source
        result.epoch = self.epoch
        result.is_orthometric = self.is_orthometric
        result.original_horizontal_crs = self.original_horizontal_crs
        result.current_horizontal_crs = self.current_horizontal_crs
        result.original_vertical_crs = self.original_vertical_crs
        result.current_vertical_crs = self.current_vertical_crs
        result.original_compound_crs = self.original_compound_crs
        result.current_compound_crs = self.current_compound_crs
        result.original_geoid_model = self.original_geoid_model
        result.current_geoid_model = self.current_geoid_model
        result.crs_history = self.crs_history
        
        # Keep original unit info for reference
        result.original_horizontal_unit = self.original_horizontal_unit
        result.original_vertical_unit = self.original_vertical_unit
        result.original_horizontal_units = self.original_horizontal_units
        result.original_vertical_units = self.original_vertical_units
        
        # Update current unit to the target
        result.current_horizontal_unit = self.current_horizontal_unit
        result.current_vertical_unit = target_unit
        result.current_horizontal_units = self.current_horizontal_units
        result.current_vertical_units = target_unit.display_name
        
        # Record in CRS history if available
        if result.crs_history is not None:
            try:
                result.crs_history.add_entry({
                    'operation': 'unit_conversion',
                    'source_unit': source_unit.name,
                    'target_unit': target_unit.name,
                    'conversion_factor': factor,
                    'source_file': self.filename,
                    'target_file': output_path,
                })
            except Exception:
                pass  # Graceful degradation
        
        return result
    
    def print_unit_info(self) -> None:
        """Print detailed information about the raster's units."""
        print("==================")
        print("==== Unit Info ====")
        print("==================")
        print(f"\nOriginal Horizontal: {describe_unit(self.original_horizontal_unit)}")
        print(f"Original Vertical:   {describe_unit(self.original_vertical_unit)}")
        print(f"\nCurrent Horizontal:  {describe_unit(self.current_horizontal_unit)}")
        print(f"Current Vertical:    {describe_unit(self.current_vertical_unit)}")
        
        h_metric, v_metric = self.are_units_metric()
        print(f"\nHorizontal is metric: {h_metric}")
        print(f"Vertical is metric: {v_metric}")
        
        if self.current_vertical_unit.name != "meter":
            factor = self.get_value_conversion_factor("meter")
            print(f"\nTo convert values to meters, multiply by: {factor:.10f}")
        
        res_m = self.get_resolution_in_meters()
        if res_m is not None:
            print(f"Resolution in meters: {res_m:.6f}")

    @classmethod
    def from_file(
        cls,
        filename: str,
        rtype: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Raster":
        """
        Load raster from file with full metadata extraction.
        """
        if not has_rasterio():
            raise DependencyMissingError("rasterio", "Raster.from_file")

        with rasterio.open(filename) as src:
            profile = src.profile
            bounds = src.bounds
            crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height
            horiz_unit, vert_unit = get_units_from_rasterio(src)
            try:
                raster_tags = src.tags()
            except Exception:
                raster_tags = {}

        # Create instance
        obj = cls(filename=filename)

        # =====================================================================
        # Parse CRS into compound/horizontal/vertical components
        # =====================================================================
        from crs_utils import parse_crs_components
        
        compound_wkt, horizontal_wkt, vertical_wkt = parse_crs_components(crs)
        
        # Store the rasterio CRS object and parsed WKT strings
        obj.crs = crs
        obj.original_compound_crs = compound_wkt
        obj.original_horizontal_crs = horizontal_wkt
        obj.original_vertical_crs = vertical_wkt
        
        # Get PROJ string representation
        if crs:
            try:
                from pyproj import CRS as _CRS
                pyproj_crs = _CRS.from_user_input(crs)
                obj.original_proj_string = pyproj_crs.to_proj4()
            except Exception:
                obj.original_proj_string = str(crs)
        else:
            obj.original_proj_string = None

        # Set current to match original
        obj.current_compound_crs = obj.original_compound_crs
        obj.current_horizontal_crs = obj.original_horizontal_crs
        obj.current_vertical_crs = obj.original_vertical_crs
        obj.current_proj_string = obj.original_proj_string
        # =====================================================================
        
        # Determine if heights are orthometric or ellipsoidal
        if obj.original_vertical_crs is not None:
            from crs_utils import is_orthometric
            try:
                obj.is_orthometric = is_orthometric(obj.original_vertical_crs)
            except Exception:
                obj.is_orthometric = False
        else:
            obj.is_orthometric = False

        obj.bounds = bounds
        obj.transform = transform
        obj.width = width
        obj.height = height
        
        # =====================================================================
        # Store unit information - enhanced with UnitInfo objects
        # =====================================================================
        obj.original_horizontal_unit = horiz_unit
        obj.original_vertical_unit = vert_unit
        obj.current_horizontal_unit = horiz_unit
        obj.current_vertical_unit = vert_unit
        
        # Backward compatible string properties
        obj.original_horizontal_units = horiz_unit.display_name
        obj.original_vertical_units = vert_unit.display_name
        obj.current_horizontal_units = horiz_unit.display_name
        obj.current_vertical_units = vert_unit.display_name

        # Merge tags + user metadata for geoid/time parsing
        meta: Dict[str, Any] = dict(raster_tags)
        if metadata:
            meta.update(metadata)

        geoid_info = parse_geoid_info(meta)
        obj.original_geoid_model = geoid_info.get("geoid_model")
        obj.current_geoid_model = obj.original_geoid_model

        # Parse time information from raster tags only (tags are more standardized)
        time_info = parse_time_info(raster_tags)
        if time_info:
            obj.time_info = time_info
            if "epoch" in time_info:
                obj.epoch = time_info["epoch"]

        # Store additional metadata
        obj.metadata = meta
        if rtype is not None:
            obj.metadata["raster_type"] = rtype
        obj.rtype = rtype

        # Epoch in metadata overrides parsed values
        if metadata and "epoch" in metadata:
            try:
                obj.epoch = float(metadata["epoch"])
                if hasattr(obj, "time_info"):
                    obj.time_info["epoch"] = obj.epoch
                    obj.time_info["epoch_source"] = "user_metadata_override"
            except Exception:
                if not hasattr(obj, "epoch"):
                    obj.epoch = None

        # Initialize CRS history
        try:
            if getattr(obj, "crs_history", None) is None:
                obj.crs_history = CRSHistory(obj)
                obj.crs_history.record_raster_creation_entry(
                    creation_parameters={
                        "source": "Raster.from_file",
                        "raster_type": rtype,
                        "metadata": metadata,
                    },
                    description="Loaded raster from existing file.",
                )
        except Exception:
            obj.crs_history = None

        return obj

    def add_metadata(
        self,
        compound_CRS: Optional[Any] = None,
        horizontal_CRS: Optional[Any] = None,
        vertical_CRS: Optional[Any] = None,
        geoid_model: Optional[str] = None,
        epoch: Optional[Any] = None,
    ) -> None:
        """
        Add or update CRS, geoid, and temporal metadata.

        You can pass:
          - compound_CRS: a full compound CRS, or a purely horizontal or purely
            vertical CRS. The function will inspect it and decide whether it is
            horizontal, vertical, or truly compound.
          - horizontal_CRS: horizontal CRS only
          - vertical_CRS: vertical CRS only

        Any combination is allowed:
          - compound only
          - horizontal only
          - vertical only
          - compound + horizontal (horizontal overrides compound's horizontal)
          - compound + vertical (vertical overrides compound's vertical)
          - horizontal + vertical (compound is built from them)
          - compound + horizontal + vertical (horizontal/vertical override)

        If only horizontal and/or vertical are given, a compound CRS is
        generated from them when possible.

        `epoch` can be:
          - float/int: decimal year
          - datetime/date: a single epoch date
          - string: single date ("2006-04-06", "04/06/2006", etc.)
          - string range: "04/06/2006 - 05/01/2006"
          - (start, end): 2-element iterable of dates/datetimes/decimal years

        Parameters
        ----------
        compound_CRS : Any, optional
            Compound CRS or any CRS input
        horizontal_CRS : Any, optional
            Horizontal CRS component
        vertical_CRS : Any, optional
            Vertical CRS component
        geoid_model : str, optional
            Geoid model name or file path
        epoch : Any, optional
            Epoch as decimal year, datetime, date, string, or range
        """
        import datetime
        from pyproj import CRS as _CRS
        from pyproj.crs import CompoundCRS
        from crs_utils import is_orthometric
        from geoid_utils import select_geoid_grid
        from time_utils import _datetime_to_decimal_year

        # -------------------------------
        # Helper: safely coerce to CRS
        # -------------------------------
        def _crs_or_none(value: Any) -> Optional[_CRS]:
            if value is None:
                return None
            try:
                return _ensure_crs_obj(value)
            except Exception:
                return None

        # -------------------------------
        # 1. Start from existing state
        # -------------------------------
        existing_horiz = _crs_or_none(
            getattr(self, "current_horizontal_crs", None)
            or getattr(self, "original_horizontal_crs", None)
        )
        existing_vert = _crs_or_none(
            getattr(self, "current_vertical_crs", None)
            or getattr(self, "original_vertical_crs", None)
        )
        existing_comp = _crs_or_none(
            getattr(self, "current_compound_crs", None)
            or getattr(self, "original_compound_crs", None)
        )

        new_horiz = existing_horiz
        new_vert = existing_vert
        new_comp = existing_comp

        crs_changed = False
        geoid_changed = False
        epoch_changed = False
        # -------------------------------
        # 2. Interpret compound_CRS, if any
        # -------------------------------
        if compound_CRS is not None:
            comp = _ensure_crs_obj(compound_CRS)

            if comp.is_compound:
                # Try to split into horizontal + vertical
                sub = getattr(comp, "sub_crs_list", None) or []
                horiz_candidate = sub[0] if len(sub) >= 1 else None
                vert_candidate = sub[1] if len(sub) >= 2 else None

                if horiz_candidate is not None:
                    new_horiz = horiz_candidate
                if vert_candidate is not None:
                    new_vert = vert_candidate

                new_comp = comp
                crs_changed = True

            elif is_3d_geographic_crs(comp):
                # 3D geographic CRS (e.g., EPSG:4979) - use directly as full CRS
                # and derive a synthetic 1D vertical CRS for the vertical_crs attribute
                new_comp = comp
                new_vert = extract_ellipsoidal_height_as_vertical_crs(comp)
                # Horizontal stays as-is (user may have set it separately)
                crs_changed = True

            else:
                # Non-compound: decide whether horizontal or vertical
                if getattr(comp, "is_vertical", False):
                    new_vert = comp
                else:
                    # Geographic or projected -> horizontal
                    new_horiz = comp
                # Compound will be rebuilt later from horiz/vert
                new_comp = None
                crs_changed = True


        # -------------------------------
        # 3. Explicit horizontal / vertical overrides
        # -------------------------------
        if horizontal_CRS is not None:
            new_horiz = _ensure_crs_obj(horizontal_CRS)
            new_comp = None
            crs_changed = True

        if vertical_CRS is not None:
            vert_candidate = _ensure_crs_obj(vertical_CRS)
            
            # Check if user passed a 3D geographic CRS (e.g., EPSG:4979) as vertical
            if is_3d_geographic_crs(vert_candidate):
                # Use the 3D CRS as the full CRS, derive synthetic 1D vertical
                new_comp = vert_candidate
                new_vert = extract_ellipsoidal_height_as_vertical_crs(vert_candidate)
            else:
                new_vert = vert_candidate
                new_comp = None
            crs_changed = True

        # -------------------------------
        # 4. Rebuild compound if needed
        # -------------------------------
        if new_comp is None:
            if new_horiz is not None and new_vert is not None:
                # Check if new_vert is actually a 3D geographic CRS
                # (shouldn't happen after section 3, but defensive check)
                if is_3d_geographic_crs(new_vert):
                    # Use the 3D CRS directly as the full CRS
                    new_comp = new_vert
                    new_vert = extract_ellipsoidal_height_as_vertical_crs(new_vert)
                else:
                    # Normal case: build true compound from 2D + 1D
                    comp_name = f"{new_horiz.name} + {new_vert.name}"
                    new_comp = CompoundCRS(name=comp_name, components=[new_horiz, new_vert])
            elif new_horiz is not None:
                # Horizontal only
                new_comp = new_horiz
            elif new_vert is not None:
                # Vertical only - check if it's actually a 3D CRS
                if is_3d_geographic_crs(new_vert):
                    new_comp = new_vert
                    new_vert = extract_ellipsoidal_height_as_vertical_crs(new_vert)
                else:
                    new_comp = new_vert
            else:
                new_comp = existing_comp  # nothing better to do

        # -------------------------------
        # 5. Write back CRS using auto-sync properties
        # -------------------------------
        if crs_changed and new_comp is not None:
            # Use auto-sync properties for consistency
            self.current_compound_crs = new_comp.to_wkt()
            
        if crs_changed and new_horiz is not None:
            self.current_horizontal_crs = new_horiz.to_wkt()
            # Update horizontal unit from new CRS
            self.current_horizontal_unit = get_horizontal_unit(new_horiz)
            self.current_horizontal_units = self.current_horizontal_unit.display_name
            
        if crs_changed and new_vert is not None:
            self.current_vertical_crs = new_vert.to_wkt()
            # Update vertical unit from new CRS, but preserve if already explicitly set
            new_vert_unit = get_vertical_unit(new_vert)
            # Only update if new CRS has known units, OR if current unit is unknown
            # This preserves explicitly-set units when transforming to CRS without unit info
            if new_vert_unit.name != "unknown" or self.current_vertical_unit.name == "unknown":
                self.current_vertical_unit = new_vert_unit
                self.current_vertical_units = self.current_vertical_unit.display_name

        # Update legacy crs attribute for backward compatibility
        if crs_changed and new_comp is not None:
            self.crs = new_comp

        # Orthometric flag can update when vertical changes
        if crs_changed and new_vert is not None:
            self.is_orthometric = is_orthometric(new_vert.to_wkt())

        # -------------------------------
        # 6. Geoid model
        # -------------------------------
        if geoid_model is not None:
            gm_str = str(geoid_model)

            # Case A: looks like a file path or filename (e.g., "us_noaa_geoid03_conus.tif")
            #         -> just store the basename, do NOT call select_geoid_grid again.
            # Case B: looks like an alias (e.g., "GEOID03", "GEOID18")
            #         -> resolve to a grid path via select_geoid_grid.
            if gm_str.lower().endswith(".tif") or "/" in gm_str or "\\" in gm_str:
                self.current_geoid_model = Path(gm_str).name
            else:
                selected_geoid, _ = select_geoid_grid(gm_str, verbose=False)
                self.current_geoid_model = Path(selected_geoid).name

            geoid_changed = True

        # -------------------------------
        # 7. Epoch handling
        # -------------------------------
        if epoch is not None:

            def _epoch_to_decimal_year(value: Any) -> float:
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, datetime.datetime):
                    return _datetime_to_decimal_year(value)
                if isinstance(value, datetime.date):
                    dt = datetime.datetime.combine(value, datetime.time())
                    return _datetime_to_decimal_year(dt)
                if isinstance(value, str):
                    parsed = _parse_epoch_string_to_decimal(value)
                    if isinstance(parsed, tuple):
                        # If string itself is a range, we take the mid-point.
                        return 0.5 * (parsed[0] + parsed[1])
                    return float(parsed)
                raise TypeError(
                    "epoch must be a float/int (decimal year), a datetime/date, "
                    "a string (date or 'start - end'), or a 2-element range of those."
                )

            # String range: "start - end"
            if isinstance(epoch, str):
                parsed = _parse_epoch_string_to_decimal(epoch)
                if isinstance(parsed, tuple):
                    start_dec, end_dec = parsed
                    self.epoch_start = start_dec
                    self.epoch_end = end_dec
                    self.epoch = 0.5 * (start_dec + end_dec)
                else:
                    epoch_dec = float(parsed)
                    self.epoch = epoch_dec
                    self.epoch_start = epoch_dec
                    self.epoch_end = epoch_dec
            # Tuple/list range: (start, end)
            elif isinstance(epoch, (list, tuple)) and len(epoch) == 2:
                start_dec = _epoch_to_decimal_year(epoch[0])
                end_dec = _epoch_to_decimal_year(epoch[1])
                self.epoch_start = min(start_dec, end_dec)
                self.epoch_end = max(start_dec, end_dec)
                self.epoch = 0.5 * (self.epoch_start + self.epoch_end)
            else:
                # Single value (numeric, date, datetime, etc.)
                epoch_dec = _epoch_to_decimal_year(epoch)
                self.epoch = epoch_dec
                self.epoch_start = epoch_dec
                self.epoch_end = epoch_dec

            epoch_changed = True

        # -------------------------------
        # 8. Record a single CRSHistory entry summarizing all changes
        # -------------------------------
        try:
            from crs_history import CRSHistory as _CRSHistory  # type: ignore
        except Exception:
            _CRSHistory = None  # type: ignore

        # Initialize history if missing and possible
        if getattr(self, "crs_history", None) is None and _CRSHistory is not None:
            try:
                self.crs_history = _CRSHistory(self)
            except Exception:
                self.crs_history = None

        if getattr(self, "crs_history", None) is not None:
            # Only pass updated pieces; CRSHistory keeps its own current state.
            try:
                self.crs_history.add_manual_change_entry(
                    new_compound_crs_proj=new_comp if crs_changed else None,
                    new_horizontal_crs_proj=new_horiz if crs_changed else None,
                    new_vertical_crs_proj=new_vert if crs_changed else None,
                    geoid_model=self.current_geoid_model if geoid_changed else None,
                    epoch=self.epoch if epoch_changed else None,
                    horizontal_units=getattr(self, "current_horizontal_units", None),
                    vertical_units=getattr(self, "current_vertical_units", None),
                    note="Raster.add_metadata manual update.",
                )
            except Exception:
                pass

    def plot(self, *, ax=None, cmap="viridis", vmin=None, vmax=None, title=None, **imshow_kw):
        """
        Display the raster as an image using Matplotlib.
        """
        try:
            arr = self.data.squeeze().values
        except Exception:
            if not has_rasterio():
                raise RuntimeError("rasterio or rioxarray required for plotting")
            with rasterio.open(self.filename) as src:
                arr = src.read(1)

        import matplotlib.pyplot as plt
        import numpy as np

        if ax is None:
            _, ax = plt.subplots()

        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)

        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kw)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title)
        return ax

    # =========================================================================
    # Derived raster products
    # =========================================================================

    def _process_dem(
        self,
        dem_path: Path,
        output_filename: str,
        processing_mode: str,
        options: Optional[Any] = None,
    ) -> "Raster":
        """
        Internal method to process DEM using GDAL DEMProcessing.

        Parameters
        ----------
        dem_path : Path
            Path to the input DEM.
        output_filename : str
            Name for the output file.
        processing_mode : str
            GDAL processing mode: 'hillshade', 'slope', 'aspect', 'roughness', etc.
        options : gdal.DEMProcessingOptions, optional
            GDAL options for the processing.

        Returns
        -------
        Raster
            New Raster object for the derived product.
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise DependencyMissingError("gdal", "Raster DEM processing methods")

        import os

        # Build output path in same directory as input
        output_path = os.path.join(os.path.dirname(self.filename), output_filename)

        # Open source to check nodata value
        src_ds = gdal.Open(str(dem_path), gdal.GA_ReadOnly)
        if src_ds is None:
            raise ValueError(f"Could not open DEM file: {dem_path}")

        src_band = src_ds.GetRasterBand(1)
        src_nodata = src_band.GetNoDataValue()
        src_ds = None  # Close dataset

        # Prepare processing options with proper nodata handling
        # Use -9999 as default nodata if source doesn't have one
        output_nodata = src_nodata if src_nodata is not None else -9999.0

        # Build options: merge user options with essential settings
        if options is not None:
            # Try to extract keyword arguments from existing options
            try:
                import inspect
                # Get the options as keyword arguments if possible
                opts_dict = {}
                for key in dir(options):
                    if not key.startswith('_'):
                        val = getattr(options, key, None)
                        if val is not None and not callable(val):
                            opts_dict[key] = val

                # Add our required options
                opts_dict['computeEdges'] = True
                opts_dict['format'] = 'GTiff'

                # Create new options with merged settings
                options = gdal.DEMProcessingOptions(**opts_dict)
            except (TypeError, AttributeError, ImportError):
                # If merging fails, create fresh options with just computeEdges
                options = gdal.DEMProcessingOptions(computeEdges=True, format='GTiff')
        else:
            # Create default options
            options = gdal.DEMProcessingOptions(computeEdges=True, format='GTiff')

        # Run GDAL DEM processing
        result_ds = gdal.DEMProcessing(output_path, str(dem_path), processing_mode, options=options)

        # Explicitly set nodata value on output band
        if result_ds is not None:
            result_band = result_ds.GetRasterBand(1)
            result_band.SetNoDataValue(output_nodata)
            result_band.FlushCache()
            result_ds.FlushCache()

        result_ds = None  # Close dataset

        # Load as new Raster
        result = Raster.from_file(
            output_path,
            rtype=f"{processing_mode}",
            metadata={"source_dem": str(dem_path), "processing": processing_mode},
        )

        return result

    def hillshade(
        self,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        output_path: Optional[str] = None,
    ) -> "Raster":
        """
        Compute a hillshade raster from this DEM.

        Parameters
        ----------
        azimuth : float, default 315
            Sun azimuth angle in degrees (0-360, clockwise from north).
        altitude : float, default 45
            Sun altitude angle in degrees above the horizon (0-90).
        output_path : str, optional
            Output file path. If None, generates based on input filename.

        Returns
        -------
        Raster
            Hillshade raster.

        Examples
        --------
        >>> dem = Raster.from_file("elevation.tif")
        >>> hillshade = dem.hillshade(azimuth=315, altitude=45)
        >>> hillshade.plot(cmap="gray")
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise DependencyMissingError("gdal", "Raster.hillshade")

        if output_path is None:
            output_path = f"{Path(self.filename).stem}_hillshade.tif"

        opts = gdal.DEMProcessingOptions(azimuth=azimuth, altitude=altitude, computeEdges=True)
        return self._process_dem(Path(self.filename), output_path, "hillshade", options=opts)

    def slope(self, output_path: Optional[str] = None) -> "Raster":
        """
        Derive a slope raster from this DEM.

        Slope is computed as the maximum rate of change in elevation
        between each cell and its neighbors, expressed in degrees.

        Parameters
        ----------
        output_path : str, optional
            Output file path. If None, generates based on input filename.

        Returns
        -------
        Raster
            Slope raster (values in degrees).

        Examples
        --------
        >>> dem = Raster.from_file("elevation.tif")
        >>> slope = dem.slope()
        >>> slope.plot(cmap="YlOrRd")
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise DependencyMissingError("gdal", "Raster.slope")

        if output_path is None:
            output_path = f"{Path(self.filename).stem}_slope.tif"

        opts = gdal.DEMProcessingOptions(computeEdges=True)
        return self._process_dem(Path(self.filename), output_path, "slope", options=opts)

    def aspect(self, output_path: Optional[str] = None) -> "Raster":
        """
        Derive an aspect raster from this DEM.

        Aspect is the compass direction that the slope faces, expressed
        in degrees (0-360, clockwise from north). Flat areas are assigned 0.

        Parameters
        ----------
        output_path : str, optional
            Output file path. If None, generates based on input filename.

        Returns
        -------
        Raster
            Aspect raster (values in degrees, 0-360).

        Notes
        -----
        In very old GDAL versions (<3.4), the zeroForFlat option may not
        be available, in which case a warning is emitted.

        Examples
        --------
        >>> dem = Raster.from_file("elevation.tif")
        >>> aspect = dem.aspect()
        >>> aspect.plot(cmap="hsv")
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise DependencyMissingError("gdal", "Raster.aspect")

        if output_path is None:
            output_path = f"{Path(self.filename).stem}_aspect.tif"

        # Try to use zeroForFlat option (GDAL >= 3.4)
        try:
            opts = gdal.DEMProcessingOptions(zeroForFlat=True, computeEdges=True)
        except TypeError:
            import warnings
            warnings.warn(
                "GDAL < 3.4 lacks zeroForFlat option - flat areas will have default aspect value"
            )
            opts = gdal.DEMProcessingOptions(computeEdges=True)

        return self._process_dem(Path(self.filename), output_path, "aspect", options=opts)

    def roughness(self, output_path: Optional[str] = None) -> "Raster":
        """
        Compute terrain roughness from this DEM.

        Roughness is the largest inter-cell difference of a central pixel
        and its surrounding cells, providing a measure of terrain variability.

        Parameters
        ----------
        output_path : str, optional
            Output file path. If None, generates based on input filename.

        Returns
        -------
        Raster
            Roughness raster.

        Examples
        --------
        >>> dem = Raster.from_file("elevation.tif")
        >>> roughness = dem.roughness()
        >>> roughness.plot(cmap="terrain")
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise DependencyMissingError("gdal", "Raster.roughness")

        if output_path is None:
            output_path = f"{Path(self.filename).stem}_roughness.tif"

        opts = gdal.DEMProcessingOptions(computeEdges=True)
        return self._process_dem(Path(self.filename), output_path, "roughness", options=opts)

    def warp_raster(
        self,
        target_crs: Optional[Any] = None,
        dynamic_target_epoch: Optional[float] = None,
        source_vertical_kind: Optional[str] = None,
        target_vertical_kind: Optional[str] = None,
        source_geoid_model: Optional[str] = None,
        target_geoid_model: Optional[str] = None,
        resolution: Optional[float] = None,
        interpolation_method: str = "bilinear",
        align_to: Optional["Raster"] = None,
        output_path: Optional[str] = None,
        overwrite: bool = False,
    ) -> "Raster":
        """
        Unified warp method that handles horizontal, vertical, and epoch transforms.
        
        When multiple transformation types are requested, they are composed into
        a single operation for efficiency. This avoids multiple resampling passes
        which degrade data quality.
        
        Parameters
        ----------
        target_crs : Any, optional
            Target horizontal CRS (EPSG code, WKT, or pyproj CRS object)
        dynamic_target_epoch : float, optional
            Target epoch for dynamic coordinate transformation (decimal year)
        source_vertical_kind : str, optional
            Source vertical datum kind: 'ellipsoidal' or 'orthometric'
        target_vertical_kind : str, optional
            Target vertical datum kind: 'ellipsoidal' or 'orthometric'
        source_geoid_model : str, optional
            Source geoid model name/path
        target_geoid_model : str, optional
            Target geoid model name/path
        resolution : float, optional
            Output resolution in target CRS units
        interpolation_method : str
            Interpolation method: 'nearest', 'bilinear', 'cubic'
        align_to : Raster, optional
            Align output to match this raster's grid exactly
        output_path : str, optional
            Output file path (auto-generated if not provided)
        overwrite : bool
            Overwrite existing output file
            
        Returns
        -------
        Raster
            Warped raster with updated metadata
            
        Notes
        -----
        For rasters, vertical datum transformation is applied to the Z-values
        (elevation data) rather than coordinates. Epoch transformation affects
        the horizontal coordinates.
        
        The transformation order is:
        1. Horizontal CRS reprojection + epoch transformation (grid resampling)
        2. Vertical datum transformation (value adjustment)
        """
        import os
        import numpy as np
        from pyproj import CRS as _CRS
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        from rasterio.transform import xy as transform_xy
        
        # Determine what transformations are needed
        needs_epoch = dynamic_target_epoch is not None
        needs_vertical = (
            source_vertical_kind is not None or 
            target_vertical_kind is not None or
            (source_geoid_model is not None and target_geoid_model is not None and 
            source_geoid_model != target_geoid_model)
        )
        needs_horizontal = target_crs is not None
        needs_align = align_to is not None
        
        # If nothing to do, return self
        if not any([needs_epoch, needs_vertical, needs_horizontal, needs_align]):
            return self
        
        # Get source CRS and epoch
        src_crs = _ensure_crs_obj(self.crs) if self.crs else None
        if src_crs is None and (needs_horizontal or needs_epoch):
            raise ValueError("Raster CRS is required for horizontal/epoch transformation")
        
        src_epoch = getattr(self, 'epoch', None)
        if needs_epoch and src_epoch is None:
            raise ValueError("Raster epoch is required for dynamic epoch transformation")
        
        dst_epoch = float(dynamic_target_epoch) if dynamic_target_epoch else src_epoch
        
        # Determine target CRS
        if target_crs is not None:
            dst_crs = _ensure_crs_obj(target_crs)
        elif align_to is not None and align_to.crs is not None:
            dst_crs = _ensure_crs_obj(align_to.crs)
        else:
            dst_crs = src_crs
        
        # Determine vertical parameters
        src_vert_kind = source_vertical_kind
        if src_vert_kind is None:
            is_ortho = getattr(self, 'is_orthometric', None)
            if is_ortho is True:
                src_vert_kind = "orthometric"
            elif is_ortho is False:
                src_vert_kind = "ellipsoidal"
        
        dst_vert_kind = target_vertical_kind or src_vert_kind
        src_geoid = source_geoid_model or getattr(self, 'current_geoid_model', None) or getattr(self, 'geoid_model', None)
        dst_geoid = target_geoid_model or src_geoid
        
        # Build output path
        if output_path is None:
            base, ext = os.path.splitext(self.filename)
            parts = []
            if needs_epoch:
                parts.append(f"epoch{dst_epoch:.2f}".replace(".", "p"))
            if needs_vertical and dst_vert_kind:
                parts.append(dst_vert_kind[:4])
            if needs_horizontal:
                parts.append("reproj")
            if needs_align:
                parts.append("aligned")
            tag = "_".join(parts) if parts else "warped"
            output_path = f"{base}_{tag}{ext}"
        
        if os.path.exists(output_path) and not overwrite:
            raise ValueError(f"Output file exists and overwrite=False: {output_path}")
        
        # Select resampling method
        resampling_map = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic,
            'cubic_spline': Resampling.cubic_spline,
            'lanczos': Resampling.lanczos,
        }
        resampling = resampling_map.get(interpolation_method.lower(), Resampling.bilinear)
        
        with rasterio.open(self.filename) as src:
            src_data = src.read(1).astype('float64')
            src_profile = src.profile.copy()
            src_transform = src.transform
            src_bounds = src.bounds
            src_width = src.width
            src_height = src.height
            src_nodata = src_profile.get('nodata', src.nodata)
            
            # Mask nodata
            if src_nodata is not None:
                nodata_mask = (src_data == src_nodata) | np.isnan(src_data)
                src_data[nodata_mask] = np.nan
        
        # Determine output grid
        if align_to is not None:
            # Match the target raster's grid exactly
            with rasterio.open(align_to.filename) as ref:
                dst_transform = ref.transform
                dst_width = ref.width
                dst_height = ref.height
                dst_crs = _ensure_crs_obj(ref.crs) if ref.crs else dst_crs
        elif needs_horizontal or needs_epoch:
            # Calculate appropriate output grid
            if resolution is not None:
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs, dst_crs, src_width, src_height, *src_bounds,
                    resolution=resolution,
                )
            else:
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs, dst_crs, src_width, src_height, *src_bounds,
                )
        else:
            # Keep same grid
            dst_transform = src_transform
            dst_width = src_width
            dst_height = src_height
        
        dst_width = int(dst_width)
        dst_height = int(dst_height)
        
        # Perform coordinate transformation (horizontal + epoch)
        if needs_epoch or (needs_horizontal and not src_crs.equals(dst_crs)):
            # Create output coordinate arrays
            rows_dst, cols_dst = np.indices((dst_height, dst_width))
            xs_dst, ys_dst = transform_xy(dst_transform, rows_dst, cols_dst, offset='center')
            xs_dst = np.array(xs_dst, dtype='float64')
            ys_dst = np.array(ys_dst, dtype='float64')
            
            # Transform destination coords back to source coords
            if needs_epoch:
                # Use dynamic transform with epoch
                z0 = np.zeros_like(xs_dst, dtype='float64')
                x_src_flat, y_src_flat, _ = apply_dynamic_transform(
                    x=xs_dst.ravel(),
                    y=ys_dst.ravel(),
                    z=z0.ravel(),
                    src_crs=dst_crs,
                    dst_crs=src_crs,
                    src_epoch=dst_epoch,
                    dst_epoch=src_epoch,
                )
                x_src = x_src_flat.reshape(xs_dst.shape)
                y_src = y_src_flat.reshape(ys_dst.shape)
            else:
                # Simple CRS transform (no epoch)
                from pyproj import Transformer
                transformer = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
                x_src, y_src = transformer.transform(xs_dst, ys_dst)
            
            # Convert to pixel coordinates in source
            inv_src_transform = ~src_transform
            cols_src, rows_src = inv_src_transform * (x_src, y_src)
            
            # Interpolate
            dst_data = np.full((dst_height, dst_width), np.nan, dtype='float64')
            
            if interpolation_method.lower() == 'nearest':
                rows_nn = np.rint(rows_src).astype(int)
                cols_nn = np.rint(cols_src).astype(int)
                valid = (
                    (rows_nn >= 0) & (rows_nn < src_height) &
                    (cols_nn >= 0) & (cols_nn < src_width)
                )
                dst_data[valid] = src_data[rows_nn[valid], cols_nn[valid]]
            else:
                # Bilinear interpolation
                rows0 = np.floor(rows_src).astype(int)
                cols0 = np.floor(cols_src).astype(int)
                rows1 = rows0 + 1
                cols1 = cols0 + 1
                
                valid = (
                    (rows0 >= 0) & (rows1 < src_height) &
                    (cols0 >= 0) & (cols1 < src_width)
                )

                if np.any(valid):
                    # Flatten everything for easier indexing
                    valid_flat = valid.ravel()
                    rows0_flat = rows0.ravel()
                    cols0_flat = cols0.ravel()
                    rows1_flat = rows1.ravel()
                    cols1_flat = cols1.ravel()
                    rows_src_flat = rows_src.ravel()
                    cols_src_flat = cols_src.ravel()

                    # Get the valid pixel indices for interpolation
                    r0 = rows0_flat[valid_flat]
                    c0 = cols0_flat[valid_flat]
                    r1 = rows1_flat[valid_flat]
                    c1 = cols1_flat[valid_flat]
                    dr = rows_src_flat[valid_flat] - r0
                    dc = cols_src_flat[valid_flat] - c0

                    # Get corner values
                    v00 = src_data[r0, c0]
                    v10 = src_data[r1, c0]
                    v01 = src_data[r0, c1]
                    v11 = src_data[r1, c1]

                    # Bilinear weights
                    w00 = (1 - dr) * (1 - dc)
                    w10 = dr * (1 - dc)
                    w01 = (1 - dr) * dc
                    w11 = dr * dc

                    # NaN-aware bilinear interpolation
                    # Create validity masks for each corner
                    valid00 = ~np.isnan(v00)
                    valid10 = ~np.isnan(v10)
                    valid01 = ~np.isnan(v01)
                    valid11 = ~np.isnan(v11)

                    # Zero out weights for invalid corners
                    w00_adj = np.where(valid00, w00, 0.0)
                    w10_adj = np.where(valid10, w10, 0.0)
                    w01_adj = np.where(valid01, w01, 0.0)
                    w11_adj = np.where(valid11, w11, 0.0)

                    # Sum of valid weights
                    weight_sum = w00_adj + w10_adj + w01_adj + w11_adj

                    # Replace NaN values with 0 for computation
                    v00_safe = np.where(valid00, v00, 0.0)
                    v10_safe = np.where(valid10, v10, 0.0)
                    v01_safe = np.where(valid01, v01, 0.0)
                    v11_safe = np.where(valid11, v11, 0.0)

                    # Compute weighted sum
                    weighted_sum = v00_safe * w00_adj + v10_safe * w10_adj + v01_safe * w01_adj + v11_safe * w11_adj

                    # Normalize by weight sum (avoid division by zero)
                    # If no valid corners, result is NaN
                    interp = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)

                    # Assign to output (using flat indexing)
                    dst_data_flat = dst_data.ravel()
                    dst_data_flat[valid_flat] = interp
                    dst_data = dst_data_flat.reshape(dst_height, dst_width)
        else:
            # No coordinate transform needed, just copy or use rasterio reproject
            if needs_align and align_to is not None:
                # Check if grids are already identical
                transform_match = src_transform == dst_transform
                width_match = src_width == dst_width
                height_match = src_height == dst_height

                # Compare CRS more robustly
                try:
                    crs_match = src_crs.equals(dst_crs)
                except:
                    crs_match = (src_crs == dst_crs)

                # If transform and dimensions match, grids are functionally identical
                # even if CRS metadata differs (e.g., WKT variations of same EPSG code)
                grids_functionally_identical = transform_match and width_match and height_match

                if grids_functionally_identical:
                    # Grids are already aligned - just copy the data
                    dst_data = src_data.copy()
                else:
                    # Need to resample to different grid
                    dst_data = np.full((dst_height, dst_width), np.nan, dtype='float64')
                    reproject(
                        source=src_data,
                        destination=dst_data,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=resampling,
                        src_nodata=np.nan,
                        dst_nodata=np.nan,
                    )
            else:
                dst_data = src_data.copy()

        # Apply vertical datum transformation to Z values
        if needs_vertical and src_vert_kind and dst_vert_kind and src_vert_kind != dst_vert_kind:
            # Get geoid grid
            from geoid_utils import select_geoid_grid
            
            if src_vert_kind == "ellipsoidal" and dst_vert_kind == "orthometric":
                # Need to subtract geoid height: h_ortho = h_ellip - N
                geoid_name = dst_geoid or src_geoid
                if geoid_name:
                    dst_data = self._apply_geoid_to_raster(
                        dst_data, dst_transform, dst_crs, geoid_name, 
                        direction="subtract"
                    )
            elif src_vert_kind == "orthometric" and dst_vert_kind == "ellipsoidal":
                # Need to add geoid height: h_ellip = h_ortho + N
                geoid_name = src_geoid or dst_geoid
                if geoid_name:
                    dst_data = self._apply_geoid_to_raster(
                        dst_data, dst_transform, dst_crs, geoid_name,
                        direction="add"
                    )
        elif needs_vertical and src_geoid and dst_geoid and src_geoid != dst_geoid:
            # Geoid-to-geoid transform (both orthometric but different geoids)
            # h_ortho_dst = h_ellip - N_dst = (h_ortho_src + N_src) - N_dst
            dst_data = self._apply_geoid_to_raster(
                dst_data, dst_transform, dst_crs, src_geoid, direction="add"
            )
            dst_data = self._apply_geoid_to_raster(
                dst_data, dst_transform, dst_crs, dst_geoid, direction="subtract"
            )
        
        # Handle nodata - always use a canonical nodata value
        # Use source nodata if available, otherwise default to NaN for float data
        nan_mask = np.isnan(dst_data)
        if src_nodata is not None and not np.isnan(src_nodata):
            # Use the source's numeric nodata value
            if np.any(nan_mask):
                dst_data[nan_mask] = src_nodata
            out_nodata = src_nodata
        else:
            # Use NaN as canonical nodata for float data
            out_nodata = np.nan

        # Write output
        dst_data_float32 = dst_data.astype('float32')

        # Prepare output CRS
        if hasattr(dst_crs, 'to_wkt'):
            output_crs_wkt = dst_crs.to_wkt()
        else:
            output_crs_wkt = str(dst_crs)

        dst_profile = src_profile.copy()
        dst_profile.update({
            'crs': output_crs_wkt,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'dtype': 'float32',
            'nodata': out_nodata,
        })

        with rasterio.open(output_path, 'w', **dst_profile) as dst:
            dst.write(dst_data_float32, 1)

        # Load output raster and update metadata
        out_raster = Raster.from_file(
            output_path,
            rtype=getattr(self, 'rtype', None),
            metadata={},
        )

        out_raster.add_metadata(
            compound_CRS=dst_crs,
            geoid_model=dst_geoid,
            epoch=dst_epoch,
        )

        # Update vertical kind tracking
        if dst_vert_kind:
            out_raster.is_orthometric = (dst_vert_kind.lower() == "orthometric")

        # Preserve vertical unit from source raster (AFTER add_metadata to avoid being overwritten)
        if hasattr(self, 'current_vertical_unit') and self.current_vertical_unit.name != "unknown":
            out_raster.current_vertical_unit = self.current_vertical_unit
            out_raster.current_vertical_units = self.current_vertical_units
            out_raster.original_vertical_unit = self.original_vertical_unit
            out_raster.original_vertical_units = self.original_vertical_units
        
        # Record in CRS history
        if getattr(self, 'crs_history', None) is not None:
            try:
                transform_desc = []
                if needs_epoch:
                    transform_desc.append(f"epoch {src_epoch:.4f} → {dst_epoch:.4f}")
                if needs_horizontal:
                    transform_desc.append("horizontal CRS")
                if needs_vertical:
                    transform_desc.append(f"vertical {src_vert_kind} → {dst_vert_kind}")
                
                self.crs_history.record_transformation_entry(
                    transformation_type=f"Combined warp ({', '.join(transform_desc)})",
                    source_crs_proj=src_crs,
                    target_crs_proj=dst_crs,
                    method="Combined raster warp",
                    src_epoch=src_epoch,
                    dst_epoch=dst_epoch,
                    geoid_model=dst_geoid,
                    source_file=str(self.filename),
                    target_file=str(output_path),
                )
            except Exception:
                pass
        
        return out_raster


    def _apply_geoid_to_raster(
        self,
        data: np.ndarray,
        transform: Any,
        crs: Any,
        geoid_name: str,
        direction: str = "subtract",
    ) -> np.ndarray:
        """
        Apply geoid correction to raster elevation values.

        Parameters
        ----------
        data : np.ndarray
            Elevation data array
        transform : Affine
            Raster transform
        crs : CRS
            Raster CRS
        geoid_name : str
            Geoid model name or path
        direction : str
            'add' to convert ortho→ellip, 'subtract' for ellip→ortho

        Returns
        -------
        np.ndarray
            Corrected elevation data
        """
        import numpy as np
        from rasterio.transform import xy as transform_xy
        from pyproj import Transformer
        from pyproj import CRS as _CRS

        # Resolve geoid grid
        from geoid_utils import select_geoid_grid
        try:
            geoid_grid, _ = select_geoid_grid(geoid_name, verbose=False)
        except Exception:
            geoid_grid = geoid_name
        
        height, width = data.shape
        
        # Get coordinates of all pixels
        rows, cols = np.indices((height, width))
        xs, ys = transform_xy(transform, rows, cols, offset='center')
        xs = np.array(xs, dtype='float64')
        ys = np.array(ys, dtype='float64')
        
        # Transform to geographic if needed (geoid grids expect lat/lon)
        crs_obj = _ensure_crs_obj(crs)
        if not crs_obj.is_geographic:
            # Get geographic CRS
            if crs_obj.geodetic_crs:
                geog_crs = crs_obj.geodetic_crs
            else:
                geog_crs = _CRS.from_epsg(4326)

            transformer = Transformer.from_crs(crs_obj, geog_crs, always_xy=True)
            lons, lats = transformer.transform(xs, ys)
        else:
            lons, lats = xs, ys
        
        # Sample geoid at these locations
        try:
            # Use PROJ to get geoid heights
            from pyproj import Transformer
            
            # Create a transformer that applies the geoid
            # We transform from geographic 3D with h=0 to get geoid undulation
            geog_3d = _CRS.from_epsg(4979)  # WGS84 3D
            
            # Build a pipeline to get geoid heights
            # The vgridshift gives us the geoid undulation
            import subprocess
            
            # Simpler approach: use PROJ cs2cs or a vertical shift grid directly
            # For now, use a simplified approximation or external library
            
            # Try using pyproj's transformation with vertical CRS
            # This is a placeholder - actual implementation depends on available tools
            
            # Fallback: try to load geoid grid directly with rasterio
            try:
                import rasterio
                from scipy.interpolate import RegularGridInterpolator
                
                # Try common paths for PROJ data
                import os
                proj_data = os.environ.get('PROJ_DATA', '/usr/share/proj')
                geoid_path = None
                
                for search_path in [proj_data, '/usr/share/proj', '/usr/local/share/proj']:
                    candidate = os.path.join(search_path, geoid_grid)
                    if os.path.exists(candidate):
                        geoid_path = candidate
                        break
                
                if geoid_path is None:
                    # Try as absolute path
                    if os.path.exists(geoid_grid):
                        geoid_path = geoid_grid
                
                if geoid_path is not None:
                    with rasterio.open(geoid_path) as geoid_src:
                        geoid_data = geoid_src.read(1)
                        geoid_transform = geoid_src.transform
                        geoid_bounds = geoid_src.bounds

                        # Create interpolator
                        geoid_height, geoid_width = geoid_data.shape
                        geoid_xs = np.linspace(geoid_bounds.left, geoid_bounds.right, geoid_width)
                        geoid_ys = np.linspace(geoid_bounds.top, geoid_bounds.bottom, geoid_height)

                        # Convert longitudes to 0-360 if geoid grid uses that convention
                        lons_adjusted = lons.copy()
                        if geoid_bounds.left > 180:  # Geoid uses 0-360 convention
                            lons_adjusted = np.where(lons_adjusted < 0, lons_adjusted + 360, lons_adjusted)

                        interp = RegularGridInterpolator(
                            (geoid_ys, geoid_xs),
                            geoid_data,
                            method='linear',
                            bounds_error=False,
                            fill_value=0.0,
                        )

                        # Sample geoid heights
                        points = np.column_stack([lats.ravel(), lons_adjusted.ravel()])
                        N = interp(points).reshape(height, width)

                        # Apply correction
                        result = data.copy()
                        if direction == "add":
                            result = result + N
                        else:  # subtract
                            result = result - N

                        return result
            except Exception as e:
                import sys
                import traceback
                print(f"[ERROR] Could not apply geoid correction: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return data

        except Exception as e:
            import sys
            import traceback
            print(f"[ERROR] Geoid correction failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return data

        return data

    def transform_vertical_datum(
        self,
        source_kind: str,
        target_kind: str,
        source_geoid_model: Optional[str] = None,
        target_geoid_model: Optional[str] = None,
        output_path: Optional[str] = None,
        overwrite: bool = True,
    ) -> "Raster":
        """
        Perform a vertical datum transformation on this DEM using PROJ geoid grids.
        """
        if not has_rasterio():
            raise RuntimeError(
                "rasterio is required for Raster.transform_vertical_datum"
            )

        import os
        import numpy as np
        import rasterio
        from pyproj import CRS as _CRS
        from pyproj import Transformer
        from rasterio.transform import xy as transform_xy
        from geoid_utils import select_geoid_grid

        source_kind = source_kind.lower()
        target_kind = target_kind.lower()

        if source_kind not in ("orthometric", "ellipsoidal"):
            raise ValueError("source_kind must be 'orthometric' or 'ellipsoidal'.")
        if target_kind not in ("orthometric", "ellipsoidal"):
            raise ValueError("target_kind must be 'orthometric' or 'ellipsoidal'.")

        if output_path is None:
            base, ext = os.path.splitext(self.filename)
            tag = f"{source_kind}_to_{target_kind}"
            if (
                source_geoid_model
                and target_geoid_model
                and source_geoid_model != target_geoid_model
            ):
                tag += f"_{source_geoid_model}_to_{target_geoid_model}"
            output_path = f"{base}_{tag}{ext}"
        if os.path.exists(output_path) and not overwrite:
            raise ValueError(
                f"Output file already exists and overwrite=False: {output_path}"
            )

        if self.crs is None:
            raise ValueError(
                "Raster.crs is None; vertical datum transformation requires a known horizontal CRS."
            )
        base_crs = _CRS.from_user_input(self.crs)
        base_proj4 = base_crs.to_proj4()

        def _proj_with_geoid(grid_path: str) -> str:
            return f"{base_proj4} +geoidgrids={grid_path}"

        source_grid_path = None
        target_grid_path = None
        source_geoid_label = None
        target_geoid_label = None

        if source_kind == "orthometric":
            if not source_geoid_model:
                raise ValueError(
                    "source_geoid_model must be provided when source_kind='orthometric'."
                )
            source_grid_path, source_geoid_label = select_geoid_grid(
                source_geoid_model
            )
            source_grid_path = str(source_grid_path)

        if target_kind == "orthometric":
            if not target_geoid_model:
                raise ValueError(
                    "target_geoid_model must be provided when target_kind='orthometric'."
                )
            target_grid_path, target_geoid_label = select_geoid_grid(
                target_geoid_model
            )
            target_grid_path = str(target_grid_path)

        with rasterio.open(self.filename) as src:
            data = src.read(1).astype("float64")
            profile = src.profile.copy()
            transform = src.transform
            rows, cols = np.indices(data.shape)
            xs, ys = transform_xy(transform, rows, cols, offset="center")
            xs = np.array(xs, dtype="float64")
            ys = np.array(ys, dtype="float64")

            def _apply_vertical_transform_once(
                in_srs: str, out_srs: str, z_array: np.ndarray
            ) -> np.ndarray:
                transformer = Transformer.from_crs(
                    _CRS.from_user_input(in_srs),
                    _CRS.from_user_input(out_srs),
                    always_xy=True,
                )
                x_flat = xs.ravel()
                y_flat = ys.ravel()
                z_flat = z_array.ravel()
                x_out, y_out, z_out = transformer.transform(x_flat, y_flat, z_flat)
                return np.asarray(z_out, dtype="float64").reshape(z_array.shape)

            if (
                source_kind == "orthometric"
                and target_kind == "ellipsoidal"
                and (not target_geoid_model or target_geoid_model == source_geoid_model)
            ):
                in_srs = _proj_with_geoid(source_grid_path)
                out_srs = base_proj4
                z_new = _apply_vertical_transform_once(in_srs, out_srs, data)

            elif (
                source_kind == "ellipsoidal"
                and target_kind == "orthometric"
                and (not source_geoid_model or source_geoid_model == target_geoid_model)
            ):
                in_srs = base_proj4
                out_srs = _proj_with_geoid(target_grid_path)
                z_new = _apply_vertical_transform_once(in_srs, out_srs, data)

            elif (
                source_kind == "orthometric"
                and target_kind == "orthometric"
                and source_geoid_model
                and target_geoid_model
                and source_geoid_model != target_geoid_model
            ):
                in_srs1 = _proj_with_geoid(source_grid_path)
                out_srs1 = base_proj4
                z_ellip = _apply_vertical_transform_once(in_srs1, out_srs1, data)
                in_srs2 = base_proj4
                out_srs2 = _proj_with_geoid(target_grid_path)
                z_new = _apply_vertical_transform_once(in_srs2, out_srs2, z_ellip)

            else:
                raise ValueError(
                    "Unsupported combination of source_kind, target_kind, "
                    "source_geoid_model, and target_geoid_model."
                )

            profile.update(
                {
                    "dtype": "float32",
                }
            )
            z_new = z_new.astype("float32")

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(z_new, 1)

        out_raster = Raster.from_file(
            output_path,
            rtype=getattr(self, "rtype", None),
            metadata={},
        )

        if target_kind == "orthometric":
            final_geoid = target_geoid_model or source_geoid_model
        else:
            final_geoid = None

        out_raster.add_metadata(
            compound_CRS=self.crs,
            geoid_model=final_geoid,
            epoch=getattr(self, "epoch", None),
        )

        if getattr(self, "crs_history", None) is not None:
            try:
                out_raster.crs_history = self.crs_history
                out_raster.crs_history.record_transformation_entry(
                    transformation_type="Vertical datum transformation",
                    source_crs_proj=self.crs,
                    target_crs_proj=self.crs,
                    method="pyproj vertical transform with geoidgrids",
                    source_kind=source_kind,
                    target_kind=target_kind,
                    source_geoid_model=source_geoid_model,
                    target_geoid_model=target_geoid_model,
                    note="Raster vertical datum transformation via geoid grids.",
                )
                out_raster.crs_history.record_raster_creation_entry(
                    creation_parameters={
                        "source": "Raster.transform_vertical_datum",
                        "source_kind": source_kind,
                        "target_kind": target_kind,
                        "source_geoid_model": source_geoid_model,
                        "target_geoid_model": target_geoid_model,
                    },
                    description="DEM created by vertical datum transformation.",
                )
            except Exception:
                pass

        return out_raster

    def dynamic_epoch_transform(
        self,
        target_epoch: float,
        target_crs_proj: Optional[Any] = None,
        output_path: Optional[str] = None,
        resolution: Optional[float] = None,
        interpolation_method: str = "bilinear",
        overwrite: bool = True,
    ) -> "Raster":
        """
        Epoch-aware transformation for rasters, creating a new raster whose
        coordinates have been dynamically transformed between epochs.
        """
        if not has_rasterio():
            raise RuntimeError("rasterio is required for Raster.dynamic_epoch_transform")

        import os
        import numpy as np
        import rasterio
        from pyproj import CRS as _CRS
        from rasterio.transform import xy as transform_xy
        from rasterio.warp import calculate_default_transform

        if self.crs is None:
            raise ValueError(
                "Raster.crs is None; dynamic epoch transformation requires a known CRS."
            )

        src_crs = _CRS.from_user_input(self.crs)
        dst_crs = (
            _ensure_crs_obj(target_crs_proj) if target_crs_proj is not None else src_crs
        )

        src_epoch = getattr(self, "epoch", None)
        if src_epoch is None:
            raise ValueError(
                "Raster.epoch is not set; dynamic epoch transformation requires a known source epoch."
            )
        src_epoch = float(src_epoch)
        dst_epoch = float(target_epoch)

        if output_path is None:
            base, ext = os.path.splitext(self.filename)
            tag = f"epoch{dst_epoch:.3f}".replace(".", "p")
            output_path = f"{base}_{tag}{ext}"

        if os.path.exists(output_path) and not overwrite:
            raise ValueError(
                f"Output file already exists and overwrite=False: {output_path}"
            )

        with rasterio.open(self.filename) as src:
            src_data = src.read(1).astype("float32")
            src_profile = src.profile.copy()
            src_transform = src.transform
            src_bounds = src.bounds
            src_width = src.width
            src_height = src.height
            src_nodata = src_profile.get("nodata", src.nodata)

        data = src_data.astype("float64")
        if src_nodata is not None:
            nodata_mask = data == src_nodata
            if np.any(nodata_mask):
                data = data.copy()
                data[nodata_mask] = np.nan

        if target_crs_proj is None:
            dst_transform = src_transform
            dst_width = src_width
            dst_height = src_height
        else:
            if resolution is not None:
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs,
                    dst_crs,
                    src_width,
                    src_height,
                    *src_bounds,
                    resolution=resolution,
                )
            else:
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs,
                    dst_crs,
                    src_width,
                    src_height,
                    *src_bounds,
                )

        dst_width = int(dst_width)
        dst_height = int(dst_height)

        rows_dst, cols_dst = np.indices((dst_height, dst_width))
        xs_dst, ys_dst = transform_xy(
            dst_transform, rows_dst, cols_dst, offset="center"
        )
        xs_dst = np.array(xs_dst, dtype="float64")
        ys_dst = np.array(ys_dst, dtype="float64")

        z0 = np.zeros_like(xs_dst, dtype="float64")

        x_src_flat, y_src_flat, _z_src_flat = apply_dynamic_transform(
            x=xs_dst.ravel(),
            y=ys_dst.ravel(),
            z=z0.ravel(),
            src_crs=dst_crs,
            dst_crs=src_crs,
            src_epoch=dst_epoch,
            dst_epoch=src_epoch,
        )

        x_src = x_src_flat.reshape(xs_dst.shape)
        y_src = y_src_flat.reshape(ys_dst.shape)

        inv_src_transform = ~src_transform
        cols_src, rows_src = inv_src_transform * (x_src, y_src)

        dst_data = np.full((dst_height, dst_width), np.nan, dtype="float64")

        if interpolation_method.lower() == "nearest":
            rows_nn = np.rint(rows_src).astype(int)
            cols_nn = np.rint(cols_src).astype(int)

            valid = (
                (rows_nn >= 0)
                & (rows_nn < src_height)
                & (cols_nn >= 0)
                & (cols_nn < src_width)
            )

            dst_data[valid] = data[rows_nn[valid], cols_nn[valid]]

        else:
            rows0 = np.floor(rows_src).astype(int)
            cols0 = np.floor(cols_src).astype(int)
            rows1 = rows0 + 1
            cols1 = cols0 + 1

            valid = (
                (rows0 >= 0)
                & (rows1 < src_height)
                & (cols0 >= 0)
                & (cols1 < src_width)
            )

            if np.any(valid):
                r0 = rows0[valid]
                c0 = cols0[valid]
                r1 = rows1[valid]
                c1 = cols1[valid]

                dr = rows_src[valid] - r0
                dc = cols_src[valid] - c0

                v00 = data[r0, c0]
                v10 = data[r1, c0]
                v01 = data[r0, c1]
                v11 = data[r1, c1]

                w00 = (1.0 - dr) * (1.0 - dc)
                w10 = dr * (1.0 - dc)
                w01 = (1.0 - dr) * dc
                w11 = dr * dc

                interp = (
                    v00 * w00
                    + v10 * w10
                    + v01 * w01
                    + v11 * w11
                )

                dst_data[valid] = interp

        if src_nodata is None:
            out_nodata = None
        else:
            out_nodata = src_nodata
            nan_mask = np.isnan(dst_data)
            if np.any(nan_mask):
                dst_data = dst_data.copy()
                dst_data[nan_mask] = out_nodata

        dst_profile = src_profile.copy()
        dst_profile.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "dtype": "float32",
                "nodata": out_nodata,
            }
        )

        with rasterio.open(output_path, "w", **dst_profile) as dst:
            dst.write(dst_data.astype("float32"), 1)

        out_raster = Raster.from_file(
            output_path,
            rtype=getattr(self, "rtype", None),
            metadata={},
        )

        out_raster.add_metadata(
            compound_CRS=dst_crs,
            geoid_model=getattr(self, "current_geoid_model", None),
            epoch=dst_epoch,
        )

        if getattr(self, "crs_history", None) is not None:
            try:
                out_raster.crs_history = self.crs_history
                out_raster.crs_history.record_transformation_entry(
                    transformation_type="Dynamic epoch transformation",
                    source_crs_proj=src_crs,
                    target_crs_proj=dst_crs,
                    method="epoch-aware grid warp via apply_dynamic_transform",
                    src_epoch=src_epoch,
                    dst_epoch=dst_epoch,
                    note="Raster coordinates moved using time-dependent CRS transformation.",
                )
                out_raster.crs_history.record_raster_creation_entry(
                    creation_parameters={
                        "source": "Raster.dynamic_epoch_transform",
                        "target_epoch": dst_epoch,
                        "target_crs_proj": str(target_crs_proj)
                        if target_crs_proj is not None
                        else None,
                        "resolution": resolution,
                        "interpolation_method": interpolation_method,
                    },
                    description="Raster created by dynamic epoch transformation.",
                )
            except Exception:
                pass

        return out_raster