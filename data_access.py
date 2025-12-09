"""Data access utilities for OpenTopography and point cloud APIs.

This module provides classes for querying and downloading data from
OpenTopography's catalog API, including:
- Interactive AOI definition via maps
- Manual AOI definition via bounds or files
- Catalog querying
- DEM and point cloud download

Integration with domain classes:
- Raster: CRS-aware raster with metadata tracking
- PointCloud: CRS-aware point cloud with metadata tracking
- RasterPair: Pair of rasters for differencing
- PointCloudPair: Pair of point clouds for comparison
"""

from __future__ import annotations

import os
import re
import json
import uuid
import tempfile
import warnings
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import rasterio
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import shape, box, mapping, Polygon
from shapely.ops import unary_union
from pyproj import Transformer, CRS
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest
from ipyleaflet import Map, GeomanDrawControl, GeoJSON, LegendControl, basemaps, ScaleControl
from scipy.interpolate import griddata, Rbf
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional boto3 for S3 access
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False
    boto3 = None


# Optional PDAL
try:
    import pdal
    _PDAL_AVAILABLE = True
except ImportError:
    _PDAL_AVAILABLE = False

# GDAL config
gdal.UseExceptions()


# =============================================================================
# Domain Class Imports (optional - graceful degradation if not available)
# =============================================================================
try:
    from raster import Raster
    from pointcloud import PointCloud
    from rasterpair import RasterPair
    from pointcloudpair import PointCloudPair
    from crs_history import CRSHistory
    from unit_utils import (
        UnitInfo,
        UNKNOWN_UNIT,
        METER,
        FOOT,
        US_SURVEY_FOOT,
        lookup_unit,
        parse_unit_string,
        parse_catalog_vertical_units,
        get_horizontal_unit,
        get_vertical_unit,
        convert_length,
        convert_to_meters,
        get_conversion_factor,
        format_value_with_unit,
        describe_unit,
    )
    _DOMAIN_CLASSES_AVAILABLE = True
    _UNIT_UTILS_AVAILABLE = True
except ImportError:
    try:
        from raster import Raster
        from pointcloud import PointCloud
        from rasterpair import RasterPair
        from pointcloudpair import PointCloudPair
        from crs_history import CRSHistory
        from unit_utils import (
            UnitInfo,
            UNKNOWN_UNIT,
            METER,
            FOOT,
            US_SURVEY_FOOT,
            lookup_unit,
            parse_unit_string,
            parse_catalog_vertical_units,
            get_horizontal_unit,
            get_vertical_unit,
            convert_length,
            convert_to_meters,
            get_conversion_factor,
            format_value_with_unit,
            describe_unit,
        )
        _DOMAIN_CLASSES_AVAILABLE = True
        _UNIT_UTILS_AVAILABLE = True
    except ImportError:
        _DOMAIN_CLASSES_AVAILABLE = False
        _UNIT_UTILS_AVAILABLE = False
        Raster = None
        PointCloud = None
        RasterPair = None
        PointCloudPair = None
        CRSHistory = None
        # Fallback definitions
        UnitInfo = None
        UNKNOWN_UNIT = None
        METER = None
        FOOT = None
        US_SURVEY_FOOT = None


# =============================================================================
# Helper Functions
# =============================================================================

def _date_to_decimal_year(d: Union[date, datetime, None]) -> Optional[float]:
    """
    Convert a date or datetime to a decimal year.
    
    Parameters
    ----------
    d : date, datetime, or None
        The date to convert
        
    Returns
    -------
    float or None
        Decimal year (e.g., 2018.205) or None if input is None
        
    Examples
    --------
    >>> _date_to_decimal_year(date(2018, 3, 15))
    2018.2027...
    >>> _date_to_decimal_year(datetime(2022, 6, 15, 12, 0, 0))
    2022.4534...
    """
    if d is None:
        return None
    
    if isinstance(d, datetime):
        year = d.year
        # Day of year (1-366)
        doy = d.timetuple().tm_yday
        # Add fractional day from time
        fractional_day = (d.hour + d.minute / 60.0 + d.second / 3600.0) / 24.0
        # Days in year (account for leap years)
        days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        return year + (doy - 1 + fractional_day) / days_in_year
    elif isinstance(d, date):
        year = d.year
        doy = d.timetuple().tm_yday
        days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        return year + (doy - 1) / days_in_year
    else:
        return None


def _parse_vertical_crs_string(vertical_str: Optional[str]) -> Dict[str, Any]:
    """
    Parse the 'Vertical Coordinates' string from OpenTopography catalog.
    
    The catalog returns strings like:
    - "NAVD88 (Geoid 12B)"
    - "NAVD88 height - Geoid12B (metre)"
    - "NAVD88 height - Geoid12B (ftUS)"
    - "NAVD88 (GEOID18)"
    - "EGM96"
    - "Ellipsoid"
    - "Ellipsoidal"
    
    Parameters
    ----------
    vertical_str : str or None
        The vertical coordinates string from catalog
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'is_orthometric': bool
        - 'vertical_datum': str or None (e.g., "NAVD88", "EGM96")
        - 'geoid_model': str or None (e.g., "geoid12b", "geoid18")
        - 'units': str or None (e.g., "meter", "foot", "us_survey_foot")
        - 'unit_info': UnitInfo or None (enhanced unit metadata)
    """
    # Handle None, NaN, empty strings
    if vertical_str is None:
        return {'is_orthometric': None, 'vertical_datum': None, 'geoid_model': None, 
                'units': None, 'unit_info': UNKNOWN_UNIT if _UNIT_UTILS_AVAILABLE else None}
    
    # Check for pandas NaN
    try:
        if pd.isna(vertical_str):
            return {'is_orthometric': None, 'vertical_datum': None, 'geoid_model': None, 
                    'units': None, 'unit_info': UNKNOWN_UNIT if _UNIT_UTILS_AVAILABLE else None}
    except (TypeError, ValueError):
        pass
    
    # Handle empty or whitespace-only strings
    vertical_str = str(vertical_str).strip()
    if not vertical_str or vertical_str.lower() == 'nan':
        return {'is_orthometric': None, 'vertical_datum': None, 'geoid_model': None, 
                'units': None, 'unit_info': UNKNOWN_UNIT if _UNIT_UTILS_AVAILABLE else None}
    
    v = vertical_str.lower()
    
    # Check for ellipsoidal heights
    # NOTE: Ellipsoidal heights can be in ANY unit (feet, meters, etc.)
    # The datum is independent of the unit system. We should NOT assume meters.
    if 'ellipsoid' in v:
        # First, try to extract unit from the string (e.g., "Ellipsoid (ftUS)")
        extracted_unit = None
        extracted_unit_info = UNKNOWN_UNIT if _UNIT_UTILS_AVAILABLE else None
        
        # Check for unit indicators in the string
        if _UNIT_UTILS_AVAILABLE:
            # Use the catalog parser which handles various unit formats
            parsed_unit = parse_catalog_vertical_units(vertical_str)
            if parsed_unit.name != "unknown":
                extracted_unit = parsed_unit.name
                extracted_unit_info = parsed_unit
        
        # Fallback: check for common unit patterns manually
        if extracted_unit is None:
            unit_patterns = [
                (r'\(ftus\)', 'us_survey_foot'),
                (r'\(us[\s_]*survey[\s_]*f[eo]*t\)', 'us_survey_foot'),
                (r'\(foot\)', 'foot'),
                (r'\(feet\)', 'foot'),
                (r'\(ft\)', 'foot'),
                (r'\(metre\)', 'meter'),
                (r'\(meter\)', 'meter'),
                (r'\(m\)', 'meter'),
            ]
            for pattern, unit_name in unit_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    extracted_unit = unit_name
                    if _UNIT_UTILS_AVAILABLE:
                        extracted_unit_info = lookup_unit(unit_name)
                    break
        
        # Return with whatever unit we found (or UNKNOWN if none detected)
        # Do NOT default to meters - that's a dangerous assumption
        return {
            'is_orthometric': False, 
            'vertical_datum': 'ellipsoidal', 
            'geoid_model': None, 
            'units': extracted_unit,  # None if not detected
            'unit_info': extracted_unit_info  # UNKNOWN_UNIT if not detected
        }
    
    # Parse orthometric heights with geoid model
    result = {'is_orthometric': True, 'vertical_datum': None, 'geoid_model': None, 
              'units': None, 'unit_info': UNKNOWN_UNIT if _UNIT_UTILS_AVAILABLE else None}
    
    # Use unit_utils for enhanced unit parsing if available
    if _UNIT_UTILS_AVAILABLE:
        unit_info = parse_catalog_vertical_units(vertical_str)
        result['unit_info'] = unit_info
        if unit_info.name != "unknown":
            result['units'] = unit_info.name
    else:
        # Fallback to manual parsing
        # Extract units from the string
        # Common patterns: "(metre)", "(meter)", "(m)", "(ftUS)", "(foot)", "(feet)", "(ft)"
        unit_patterns = [
            (r'\(ftus\)', 'us_survey_foot'),
            (r'\(us\s*survey\s*foot\)', 'us_survey_foot'),
            (r'\(us\s*survey\s*feet\)', 'us_survey_foot'),
            (r'\(foot\)', 'foot'),
            (r'\(feet\)', 'foot'),
            (r'\(ft\)', 'foot'),
            (r'\(metre\)', 'meter'),
            (r'\(meter\)', 'meter'),
            (r'\(meters\)', 'meter'),
            (r'\(metres\)', 'meter'),
            (r'\(m\)', 'meter'),
        ]
        
        for pattern, unit_name in unit_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                result['units'] = unit_name
                break
    
    # Extract vertical datum
    if 'navd88' in v or 'navd 88' in v:
        result['vertical_datum'] = 'NAVD88'
    elif 'ngvd29' in v or 'ngvd 29' in v:
        result['vertical_datum'] = 'NGVD29'
    elif 'egm96' in v:
        result['vertical_datum'] = 'EGM96'
        result['geoid_model'] = 'egm96'
    elif 'egm2008' in v or 'egm08' in v:
        result['vertical_datum'] = 'EGM2008'
        result['geoid_model'] = 'egm2008'
    
    # Look for geoid model ANYWHERE in the string (not just in parentheses)
    # Common patterns: "Geoid12B", "Geoid 12B", "GEOID18", "geoid09", etc.
    
    # First, try to find geoid patterns in the full string
    # Match "geoid" followed by version number (with optional space/dash)
    geoid_patterns = [
        r'geoid\s*[-_]?\s*18',      # geoid18, geoid 18, geoid-18
        r'geoid\s*[-_]?\s*12\s*b',  # geoid12b, geoid 12b, geoid12B
        r'geoid\s*[-_]?\s*12\s*a',  # geoid12a
        r'geoid\s*[-_]?\s*12(?![ab])',  # geoid12 (not followed by a or b)
        r'geoid\s*[-_]?\s*09',      # geoid09
        r'geoid\s*[-_]?\s*03',      # geoid03
        r'geoid\s*[-_]?\s*99',      # geoid99
    ]
    
    geoid_mapping = {
        r'geoid\s*[-_]?\s*18': 'geoid18',
        r'geoid\s*[-_]?\s*12\s*b': 'geoid12b',
        r'geoid\s*[-_]?\s*12\s*a': 'geoid12a',
        r'geoid\s*[-_]?\s*12(?![ab])': 'geoid12b',  # Default 12 to 12B
        r'geoid\s*[-_]?\s*09': 'geoid09',
        r'geoid\s*[-_]?\s*03': 'geoid03',
        r'geoid\s*[-_]?\s*99': 'geoid99',
    }
    
    for pattern, geoid_name in geoid_mapping.items():
        if re.search(pattern, v, re.IGNORECASE):
            result['geoid_model'] = geoid_name
            break
    
    # If no geoid found yet, check for parenthetical content that's NOT a unit
    if result['geoid_model'] is None:
        paren_match = re.search(r'\(([^)]+)\)', vertical_str)
        if paren_match:
            paren_content = paren_match.group(1).strip().lower()
            # Skip if it's just a unit (meter, metre, foot, feet, etc.)
            unit_patterns = ['meter', 'metre', 'foot', 'feet', 'ft', 'm']
            is_unit = any(paren_content == u or paren_content.startswith(u + ' ') for u in unit_patterns)
            
            if not is_unit and 'geoid' in paren_content:
                # Extract geoid name from parentheses
                for pattern, geoid_name in geoid_mapping.items():
                    if re.search(pattern, paren_content, re.IGNORECASE):
                        result['geoid_model'] = geoid_name
                        break
    
    return result


def _normalize_unit_name(units: str) -> str:
    """
    Normalize a unit name string to a canonical form.
    
    Uses unit_utils.lookup_unit() if available for robust parsing.
    
    Parameters
    ----------
    units : str
        Unit name in any common form
        
    Returns
    -------
    str
        Normalized unit name ('meter', 'foot', or 'us_survey_foot')
        
    Raises
    ------
    ValueError
        If unit cannot be recognized
    """
    if _UNIT_UTILS_AVAILABLE:
        unit_info = lookup_unit(units)
        if unit_info is not None:
            return unit_info.name
    
    # Fallback to manual normalization
    valid_units = ['meter', 'foot', 'us_survey_foot']
    units_lower = units.lower().replace(' ', '_').replace('-', '_')
    
    # Normalize unit names
    if units_lower in ['m', 'meters', 'metre', 'metres']:
        return 'meter'
    elif units_lower in ['ft', 'feet', 'foot']:
        return 'foot'
    elif units_lower in ['ftus', 'us_ft', 'usft', 'us_feet', 'us_survey_feet', 
                         'us_survey_foot', 'survey_foot', 'survey_feet']:
        return 'us_survey_foot'
    elif units_lower in valid_units:
        return units_lower
    
    raise ValueError(f"Unrecognized unit: '{units}'")


def _get_unit_info(unit_name: str) -> Optional[Any]:
    """
    Get UnitInfo object for a unit name.
    
    Parameters
    ----------
    unit_name : str
        Unit name (e.g., 'meter', 'foot', 'us_survey_foot')
        
    Returns
    -------
    UnitInfo or None
        UnitInfo object if unit_utils is available, None otherwise
    """
    if not _UNIT_UTILS_AVAILABLE:
        return None
    
    return lookup_unit(unit_name)
class DataAccess:
    """Wrapper for interactive or programmatic AOI definition."""

    # --------------------- static helpers ------------------------------
    @staticmethod
    def _coords_to_wkt(coords):
        """Convert a nested coordinate list into a WKT-compatible string.

        Parameters
        ----------
        coords : iterable
            A sequence of linear rings (each itself a sequence of ``(x, y)`` pairs).

        Returns
        -------
        str
            A comma-separated representation of coordinates suitable for
            inclusion in OpenTopography WKT polygon strings.
        """
        return ", ".join(
            ", ".join(f"{x}, {y}" for x, y in ring) for ring in coords
        )

    @classmethod
    def geojson_to_OTwkt(cls, gj):
        """Convert a GeoJSON polygon into an OpenTopography WKT string.

        Parameters
        ----------
        gj : dict
            A GeoJSON dictionary representing a polygon geometry.

        Returns
        -------
        str
            A comma-separated WKT representation of the polygon's coordinates.

        Raises
        ------
        ValueError
            If the provided GeoJSON does not represent a polygon.
        """
        if gj["type"] != "Polygon":
            raise ValueError("Input must be a Polygon GeoJSON")
        return cls._coords_to_wkt(gj["coordinates"])

    # --------------------- AOI via map draw ----------------------------
    def init_ot_catalog_map(
        self,
        center=(39.8283, -98.5795),
        zoom=3,
        layers=(
            ("3DEP", "https://raw.githubusercontent.com/OpenTopography/Data_Catalog_Spatial_Boundaries/main/usgs_3dep_boundaries.geojson", "#228B22"),
            ("NOAA", "https://raw.githubusercontent.com/OpenTopography/Data_Catalog_Spatial_Boundaries/main/noaa_coastal_lidar_boundaries.geojson", "#0000CD"),
            ("OpenTopography", "https://raw.githubusercontent.com/OpenTopography/Data_Catalog_Spatial_Boundaries/main/OT_PC_boundaries.geojson", "#fca45d"),
        ),
    ):
        """Return ipyleaflet Map with draw control and catalog layers."""
        self.bounds: dict[str, Any] = {}
        self.polygon: dict[str, Any] = {}

        def _on_draw(control, action, geo_json):
            feats = geo_json if isinstance(geo_json, list) else [geo_json]
            shapes = [shape(f["geometry"]) for f in feats]
            wkt_list = [self.geojson_to_OTwkt(f["geometry"]) for f in feats]
            merged = unary_union(shapes)
            minx, miny, maxx, maxy = merged.bounds
            self.bounds.update(dict(south=miny, west=minx, north=maxy, east=maxx, polygon_wkt=wkt_list))
            self.polygon.update(dict(merged_polygon=merged, all_polys=shapes))
            print("AOI bounds:", self.bounds)

        m = Map(center=center, zoom=zoom, basemap=basemaps.Esri.WorldTopoMap)
        dc = GeomanDrawControl(rectangle={"pathOptions": {"color": "#fca45d", "fillColor": "#fca45d"}},
                               polygon={"pathOptions": {"color": "#6be5c3", "fillColor": "#6be5c3"}})
        dc.polyline = dc.circlemarker = {}
        dc.on_draw(_on_draw)
        m.add_control(dc)

        for name, url, color in layers:
            gj = GeoJSON(data=requests.get(url).json(), name=name, style={"color": color})
            m.add_layer(gj)
        m.add_control(LegendControl({n: c for n, _, c in layers}, name="Legend"))
        m.add_control(ScaleControl(position="bottomleft", metric=True, imperial=False))
        return m

    # --------------------- AOI via manual bounds -----------------------
    def define_bounds_manual(self, south, north, west, east):
        """Define the area of interest (AOI) from numeric latitude/longitude bounds.

        Parameters
        ----------
        south, north, west, east : float
            Geographic coordinates delimiting the rectangular AOI in degrees.

        Returns
        -------
        dict
            A dictionary containing south, north, west, east and polygon WKT entries for the AOI.
        """
        poly = box(west, south, east, north)
        self.bounds = dict(south=south, north=north, west=west, east=east,
                           polygon_wkt=[self.geojson_to_OTwkt(mapping(poly))])
        self.polygon = dict(merged_polygon=poly, all_polys=[poly])
        return self.bounds

    # --------------------- AOI via uploaded vector ---------------------
    def define_bounds_from_file(self, vector_path: str, target_crs="EPSG:4326"):
        """Define the AOI using a vector file such as a shapefile or GeoJSON.

        The geometries in the provided file are merged into a single polygon
        in the target CRS.  The AOI bounds and WKT strings are stored on
        this ``DataAccess`` instance for later use in API queries.

        Parameters
        ----------
        vector_path : str
            Path to a vector file readable by GeoPandas.
        target_crs : str, default ``'EPSG:4326'``
            Desired output CRS for the AOI.

        Returns
        -------
        dict
            A dictionary of bounds (south, west, north, east) and polygon WKT strings.
        """
        gdf = gpd.read_file(vector_path)
        if gdf.empty:
            raise ValueError("No geometries in file")
        if gdf.crs is None:
            raise ValueError("Input CRS undefined")
        if gdf.crs.to_string() != target_crs:
            gdf = gdf.to_crs(target_crs)
        merged = unary_union(gdf.geometry)
        minx, miny, maxx, maxy = merged.bounds
        wkts = [self.geojson_to_OTwkt(mapping(geom)) for geom in gdf.geometry]
        self.bounds = dict(south=miny, west=minx, north=maxy, east=maxx, polygon_wkt=wkts)
        self.polygon = dict(merged_polygon=merged, all_polys=list(gdf.geometry))
        return self.bounds

# ----------------------------------------------------------------------
# OpenTopographyQuery
# ----------------------------------------------------------------------
class OpenTopographyQuery:
    def __init__(self, data_access: DataAccess):
        """Create a new query object tied to a particular AOI.

        Parameters
        ----------
        data_access : DataAccess
            An instance of :class:`DataAccess` holding the area of interest
            bounds and polygons.  This object is used to supply spatial
            parameters when querying the OpenTopography catalog.
        """
        self.da = data_access
        self.catalog_df: pd.DataFrame | None = None

    @staticmethod
    def _clean(name: str) -> str:
        """Sanitize dataset names by replacing non-word characters with underscores."""
        return re.sub(r"[^\w]+", "_", name)

    def query_catalog(
        self,
        product_format="PointCloud",
        include_federated=True,
        detail=False,
        save_as="results.json",
        url="https://portal.opentopography.org/API/otCatalog",
    ) -> pd.DataFrame:
        """Query the OpenTopography catalog for datasets within the current AOI.

        Parameters
        ----------
        product_format : str, default ``'PointCloud'``
            Desired data product format (e.g. ``'PointCloud'``).
        include_federated : bool, default ``True``
            Whether to include federated datasets in the search.
        detail : bool, default ``False``
            Request detailed metadata in the API response.
        save_as : str or None, default ``'results.json'``
            Optional filename to save the raw JSON response.  If ``None``,
            the response is not saved.
        url : str, default OpenTopography catalog API endpoint
            The API base URL to query.

        Returns
        -------
        pandas.DataFrame
            A dataframe listing datasets found within the AOI, sorted by
            start date.  The dataframe is also stored on this object as
            ``catalog_df`` for reuse.

        Raises
        ------
        ValueError
            If the area of interest has not been defined on the associated
            :class:`DataAccess` object.
        """
        # ... (no change to the request logic) ...
        if not getattr(self.da, "bounds", None):
            raise ValueError("AOI not defined")
        params = dict(
            productFormat=product_format,
            detail=str(detail).lower(),
            outputFormat="json",
            include_federated=str(include_federated).lower(),
        )
        if self.da.bounds.get("polygon_wkt"):
            params["polygon"] = self.da.bounds["polygon_wkt"]
        else:
            params.update(dict(minx=self.da.bounds["west"], miny=self.da.bounds["south"],
                        maxx=self.da.bounds["east"], maxy=self.da.bounds["north"]))
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        if save_as:
            Path(save_as).write_bytes(r.content)
            
        data = r.json()
        rows = []
        for ds in data["Datasets"]:
            meta = ds["Dataset"]
            
            start_date_str = None
            end_date_str = None
            coverage = meta.get("temporalCoverage")

            if isinstance(coverage, str):
                # Handle the string format, e.g., "2018-01-13 / 2018-06-11"
                if "/" in coverage:
                    parts = coverage.split("/")
                    start_date_str = parts[0].strip()
                    end_date_str = parts[1].strip()
                else:
                    # Handle a single date string, e.g., "2002-09-18"
                    start_date_str = coverage.strip()
                    end_date_str = coverage.strip()
            
            elif isinstance(coverage, dict):
                # Handle the dictionary format for robustness
                start_date_str = coverage.get("startDate")
                end_date_str = coverage.get("endDate")

            rows.append(
                {
                    "Name":           meta["name"],
                    "Short Name":     meta.get("alternateName"),  # S3 bucket folder name
                    "ID type":        meta["identifier"]["propertyID"],
                    "Data Source":    "usgs" if "USGS" in meta["identifier"]["propertyID"] or "usgs" in meta["identifier"]["propertyID"] else
                                    "noaa" if "NOAA" in meta["identifier"]["propertyID"] or "noaa" in meta["identifier"]["propertyID"] else "ot",
                    "Property ID":    meta["identifier"]["value"],
                    "Horizontal EPSG":
                        next((p["value"] for p in meta["spatialCoverage"]["additionalProperty"]
                            if p["name"] == "EPSG (Horizontal)"), None),
                    "Vertical Coordinates":
                        next((p["value"] for p in meta["spatialCoverage"]["additionalProperty"]
                            if p["name"] == "Vertical Coordinates"), None),
                    "Clean Name":     self._clean(meta["name"]),
                    "StartDate":      pd.to_datetime(start_date_str).date() if start_date_str else None,
                    "EndDate":        pd.to_datetime(end_date_str).date() if end_date_str else None,
                }
            )
        
        _catalog_df =  pd.DataFrame(rows)
        catalog_df = _catalog_df.sort_values(by="StartDate").reset_index(drop=True)
        self.catalog_df = catalog_df
        
        return self.catalog_df
    
    
        

    # shorthand to select compare / reference rows by DataFrame index
    def pick(self, idx_compare: int, idx_reference: int):
        """Select the compare and reference datasets by row index.

        After querying the catalog, datasets are presented in a dataframe.
        This method picks a pair of rows corresponding to the compare and
        reference datasets.  It stores various fields (names, CRS
        definitions, etc.) as attributes on the query object for later
        processing and prints the mid-point epoch for each dataset if
        available.

        Parameters
        ----------
        idx_compare : int
            Row index of the dataset to treat as the ``compare`` survey.
        idx_reference : int
            Row index of the dataset to treat as the ``reference`` survey.

        Returns
        -------
        tuple of pandas.Series
            The selected compare and reference rows from ``catalog_df``.
        """
        df = self.catalog_df
        self.compare = df.iloc[idx_compare]
        self.reference = df.iloc[idx_reference]
        self.compare_name = self.catalog_df["Name"].iloc[idx_compare]
        self.compare_short_name = self.catalog_df["Short Name"].iloc[idx_compare]  # For S3 bucket path
        self.compare_data_source = self.catalog_df["Data Source"].iloc[idx_compare]
        self.compare_property_id = self.catalog_df["Property ID"].iloc[idx_compare]
        self.compare_horizontal_crs = self.catalog_df["Horizontal EPSG"].iloc[idx_compare]
        self.compare_vertical_crs = self.catalog_df["Vertical Coordinates"].iloc[idx_compare]
        self.compare_clean_name = self.catalog_df["Clean Name"].iloc[idx_compare]
        self.reference_name = self.catalog_df["Name"].iloc[idx_reference]
        self.reference_short_name = self.catalog_df["Short Name"].iloc[idx_reference]  # For S3 bucket path
        self.reference_data_source = self.catalog_df["Data Source"].iloc[idx_reference]
        self.reference_property_id = self.catalog_df["Property ID"].iloc[idx_reference]
        self.reference_horizontal_crs = self.catalog_df["Horizontal EPSG"].iloc[idx_reference]
        self.reference_vertical_crs = self.catalog_df["Vertical Coordinates"].iloc[idx_reference]
        self.reference_clean_name = self.catalog_df["Clean Name"].iloc[idx_reference]
        
        compare_start = df["StartDate"].iloc[idx_compare]
        compare_end = df["EndDate"].iloc[idx_compare]
        if compare_start and compare_end:
            self.compare_epoch = compare_start + (compare_end - compare_start) / 2
            # Also store as decimal year for compatibility with Raster/PointCloud
            self.compare_epoch_decimal = _date_to_decimal_year(self.compare_epoch)
        else:
            self.compare_epoch = None
            self.compare_epoch_decimal = None

        ref_start = df["StartDate"].iloc[idx_reference]
        ref_end = df["EndDate"].iloc[idx_reference]
        if ref_start and ref_end:
            self.reference_epoch = ref_start + (ref_end - ref_start) / 2
            self.reference_epoch_decimal = _date_to_decimal_year(self.reference_epoch)
        else:
            self.reference_epoch = None
            self.reference_epoch_decimal = None
        
        # Parse vertical CRS info for easier access
        self.compare_vertical_info = _parse_vertical_crs_string(self.compare_vertical_crs)
        self.reference_vertical_info = _parse_vertical_crs_string(self.reference_vertical_crs)
        
        # Log vertical CRS info for debugging
        logger.debug(f"Compare vertical CRS string: '{self.compare_vertical_crs}'")
        logger.debug(f"Compare vertical info parsed: {self.compare_vertical_info}")
        logger.debug(f"Reference vertical CRS string: '{self.reference_vertical_crs}'")
        logger.debug(f"Reference vertical info parsed: {self.reference_vertical_info}")
        
        # Print warnings about vertical CRS
        if self.compare["Vertical Coordinates"] != self.reference["Vertical Coordinates"]:
            print("⚠️  Vertical CRSs differ between datasets")
        
        # Print geoid info for user
        compare_geoid = self.compare_vertical_info.get('geoid_model')
        reference_geoid = self.reference_vertical_info.get('geoid_model')
        if compare_geoid:
            print(f"🔹 Compare Geoid: {compare_geoid}")
        else:
            print(f"⚠️  Compare vertical CRS: '{self.compare_vertical_crs}' - geoid not detected, use set_compare_geoid()")
        if reference_geoid:
            print(f"🔹 Reference Geoid: {reference_geoid}")
        else:
            print(f"⚠️  Reference vertical CRS: '{self.reference_vertical_crs}' - geoid not detected, use set_reference_geoid()")
        
        # Print units info for user (use UnitInfo if available for better display)
        compare_unit_info = self.compare_vertical_info.get('unit_info')
        reference_unit_info = self.reference_vertical_info.get('unit_info')
        compare_units = self.compare_vertical_info.get('units')
        reference_units = self.reference_vertical_info.get('units')
        
        if compare_unit_info is not None and hasattr(compare_unit_info, 'display_name') and compare_unit_info.name != 'unknown':
            print(f"🔹 Compare Units: {compare_unit_info.display_name}")
        elif compare_units:
            print(f"🔹 Compare Units: {compare_units}")
        else:
            print(f"⚠️  Compare units NOT DETECTED from catalog string: '{self.compare_vertical_crs}'")
            print(f"    → Use set_compare_units('us_survey_foot') or set_compare_units('meter') to specify")
        
        if reference_unit_info is not None and hasattr(reference_unit_info, 'display_name') and reference_unit_info.name != 'unknown':
            print(f"🔹 Reference Units: {reference_unit_info.display_name}")
        elif reference_units:
            print(f"🔹 Reference Units: {reference_units}")
        else:
            print(f"⚠️  Reference units NOT DETECTED from catalog string: '{self.reference_vertical_crs}'")
            print(f"    → Use set_reference_units('us_survey_foot') or set_reference_units('meter') to specify")
        
        # Warn about unit mismatch or unknown units
        if compare_units and reference_units and compare_units != reference_units:
            print(f"⚠️  UNIT MISMATCH: Compare is {compare_units}, Reference is {reference_units} - conversion will be needed!")
        elif (compare_units is None or reference_units is None) and (compare_units != reference_units):
            print(f"⚠️  UNIT CHECK REQUIRED: One or both units unknown - verify data values match expected ranges")
        
        print(f"🔹 Compare Epoch: {self.compare_epoch} ({self.compare_epoch_decimal:.4f})" if self.compare_epoch_decimal else f"🔹 Compare Epoch: {self.compare_epoch}")
        print(f"🔹 Reference Epoch: {self.reference_epoch} ({self.reference_epoch_decimal:.4f})" if self.reference_epoch_decimal else f"🔹 Reference Epoch: {self.reference_epoch}")
        
        return self.compare, self.reference
    
    def get_metadata_dict(self, dataset: str = "compare") -> Dict[str, Any]:
        """
        Get metadata for a dataset as a dictionary suitable for Raster/PointCloud.
        
        Parameters
        ----------
        dataset : str
            Either "compare" or "reference"
            
        Returns
        -------
        dict
            Metadata dictionary with keys:
            - epoch: float (decimal year)
            - geoid_model: str or None
            - is_orthometric: bool or None
            - horizontal_crs: str (EPSG code)
            - vertical_datum: str or None
            - vertical_units: str or None
            - vertical_unit_info: UnitInfo or None (if unit_utils available)
        """
        if dataset == "compare":
            return {
                'epoch': self.compare_epoch_decimal,
                'epoch_date': self.compare_epoch,
                'geoid_model': self.compare_vertical_info.get('geoid_model'),
                'is_orthometric': self.compare_vertical_info.get('is_orthometric'),
                'horizontal_crs': self.compare_horizontal_crs,
                'vertical_datum': self.compare_vertical_info.get('vertical_datum'),
                'vertical_units': self.compare_vertical_info.get('units'),
                'vertical_unit_info': self.compare_vertical_info.get('unit_info'),
                'vertical_crs_string': self.compare_vertical_crs,
                'name': self.compare_name,
                'short_name': self.compare_short_name,
                'data_source': self.compare_data_source,
                'property_id': self.compare_property_id,
            }
        elif dataset == "reference":
            return {
                'epoch': self.reference_epoch_decimal,
                'epoch_date': self.reference_epoch,
                'geoid_model': self.reference_vertical_info.get('geoid_model'),
                'is_orthometric': self.reference_vertical_info.get('is_orthometric'),
                'horizontal_crs': self.reference_horizontal_crs,
                'vertical_datum': self.reference_vertical_info.get('vertical_datum'),
                'vertical_units': self.reference_vertical_info.get('units'),
                'vertical_unit_info': self.reference_vertical_info.get('unit_info'),
                'vertical_crs_string': self.reference_vertical_crs,
                'name': self.reference_name,
                'short_name': self.reference_short_name,
                'data_source': self.reference_data_source,
                'property_id': self.reference_property_id,
            }
        else:
            raise ValueError(f"dataset must be 'compare' or 'reference', got '{dataset}'")

# ------------------------------------------------------------------------------------------
# Download data and make DEMs
# ------------------------------------------------------------------------------------------
class GetDEMs:
    """Generate DEMs from local LAZ/point files or AWS EPT sources."""

    def __init__(self, data_access, ot_query):
        self.da = data_access
        self.ot = ot_query
                          
    # -----------------Raster gap‑fill utility--------------------------
    @staticmethod
    def fill_no_data(input_file, output_file, *, method="idw", nodata=-9999, max_dist=100, smooth_iter=0):
        """
        Fill missing values in a raster by interpolation and write to a new file.

        This utility reads the first band of ``input_file``, locates pixels equal
        to the supplied ``nodata`` value, and replaces those gaps using either
        GDAL's built-in inverse distance weighted (IDW) fill routine or a
        SciPy-based interpolator.  For SciPy methods, valid pixel coordinates
        are used to estimate missing values with nearest-neighbour, linear,
        cubic, or thin‑plate spline interpolation.  The resulting filled array
        is written to ``output_file`` with the same geotransform, projection
        and ``nodata`` metadata as the source raster.

        Parameters
        ----------
        input_file : str
            Path to the input raster file with gaps.
        output_file : str
            Destination filename for the filled raster.
        method : {"idw", "nearest", "linear", "cubic", "spline"}, optional
            Algorithm used to fill the gaps.  ``idw`` invokes GDAL's
            ``FillNodata`` function; other options use SciPy's
            :func:`scipy.interpolate.griddata` or radial basis function (Rbf)
            interpolation.  Default is ``"idw"``.
        nodata : float or int, default -9999
            Pixel value representing missing data in the input raster.
        max_dist : int, default 100
            Maximum search distance in pixels for the IDW fill method.
        smooth_iter : int, default 0
            Number of smoothing iterations applied by GDAL's ``FillNodata``.

        Returns
        -------
        None
            The function writes the filled raster to ``output_file`` and has
            no return value.
        """
        ds = gdal.Open(input_file, gdal.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        mask = arr == nodata

        def _interpolate(other):
            valid = np.where(~mask)
            nod = np.where(mask)
            coords = np.column_stack(valid)
            vals = arr[valid]
            if other == "nearest":
                return griddata(coords, vals, np.column_stack(nod), method="nearest")
            if other == "linear":
                return griddata(coords, vals, np.column_stack(nod), method="linear")
            if other == "cubic":
                return griddata(coords, vals, np.column_stack(nod), method="cubic")
            if other == "spline":
                rbf = Rbf(coords[:, 0], coords[:, 1], vals, function="thin_plate")
                return rbf(nod[0], nod[1])
            raise ValueError("Unknown method")

        if method == "idw":
            mem = gdal.GetDriverByName("MEM").CreateCopy("", ds, 0)
            gdal.FillNodata(mem.GetRasterBand(1), None, max_dist, smooth_iter)
            filled = mem.GetRasterBand(1).ReadAsArray()
        else:
            filled = arr.copy()
            filled_vals = _interpolate(method)
            filled[np.where(mask)] = filled_vals

        drv = gdal.GetDriverByName("GTiff")
        out = drv.Create(output_file, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
        out.SetGeoTransform(ds.GetGeoTransform())
        out.SetProjection(ds.GetProjection())
        out.GetRasterBand(1).WriteArray(filled)
        out.GetRasterBand(1).SetNoDataValue(nodata)
        out.FlushCache()

    @staticmethod
    def cleanup_raster_nodata(raster_path, nodata=-9999):
        """
        Clean up a raster to ensure NaN values are replaced with nodata.
        
        PDAL's writers.gdal can output NaN values for cells without data even
        when nodata is set. This function reads the raster, replaces any NaN
        or infinite values with the specified nodata value, and ensures the
        nodata metadata is properly set.
        
        Parameters
        ----------
        raster_path : str
            Path to the raster file to clean up.
        nodata : float or int, default -9999
            Value to use for nodata pixels.
            
        Returns
        -------
        bool
            True if the raster was modified, False otherwise.
        """
        ds = gdal.Open(raster_path, gdal.GA_Update)
        if ds is None:
            logger.warning(f"Could not open {raster_path} for nodata cleanup")
            return False
        
        modified = False
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        
        # Check for NaN or infinite values
        nan_mask = np.isnan(arr) | np.isinf(arr)
        nan_count = np.sum(nan_mask)
        
        if nan_count > 0:
            logger.info(f"Replacing {nan_count} NaN/Inf values with nodata={nodata} in {Path(raster_path).name}")
            arr[nan_mask] = nodata
            band.WriteArray(arr)
            modified = True
        
        # Ensure nodata value is set in metadata
        current_nodata = band.GetNoDataValue()
        if current_nodata != nodata:
            logger.info(f"Setting nodata value to {nodata} (was {current_nodata})")
            band.SetNoDataValue(nodata)
            modified = True
        
        ds.FlushCache()
        ds = None  # Close the dataset
        
        return modified

    # =========================================================================
    # Point Cloud Download Methods (EPT/S3-based)
    # =========================================================================
    
    def download_file(self, url: str, output_path: Path, retry_count: int = 3) -> Dict[str, Any]:
        """
        Download a file with retry logic and metadata collection.
        
        Parameters
        ----------
        url : str
            URL to download from
        output_path : Path
            Local path to save the file
        retry_count : int
            Number of retry attempts
            
        Returns
        -------
        dict
            Metadata about the download including success status
        """
        output_path = Path(output_path)
        file_metadata = {
            'filename': output_path.name,
            'path': str(output_path),
            'url': url,
            'download_timestamp': None,
            'size_bytes': 0,
            'success': False
        }
        
        if output_path.exists():
            logger.info(f"File already exists: {output_path.name}")
            file_metadata['size_bytes'] = output_path.stat().st_size
            file_metadata['success'] = True
            return file_metadata
        
        for attempt in range(retry_count):
            try:
                logger.info(f"Downloading: {url}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_metadata['download_timestamp'] = datetime.now().isoformat()
                file_metadata['size_bytes'] = output_path.stat().st_size
                file_metadata['success'] = True
                
                logger.info(f"Successfully downloaded: {output_path.name}")
                return file_metadata
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
        
        logger.error(f"Failed to download after {retry_count} attempts: {url}")
        return file_metadata

    def get_ept_hierarchy_nodes(self, base_url: str, bounds: List[float], 
                                 aoi_geom: Polygon, max_nodes: int = 100) -> List[str]:
        """
        Get EPT hierarchy nodes that intersect with AOI.
        
        Parameters
        ----------
        base_url : str
            Base URL for the EPT dataset
        bounds : list
            Dataset bounds [minx, miny, minz, maxx, maxy, maxz]
        aoi_geom : Polygon
            AOI geometry in the same CRS as the EPT data
        max_nodes : int
            Maximum number of nodes to return
            
        Returns
        -------
        list
            List of node keys (e.g., "0-0-0-0", "1-0-0-0")
        """
        intersecting_nodes = []
        
        # Start with root node
        queue = ['0-0-0-0']
        visited = set()
        
        while queue and len(intersecting_nodes) < max_nodes:
            node_key = queue.pop(0)
            
            if node_key in visited:
                continue
            visited.add(node_key)
            
            # Parse node key (D-X-Y-Z format)
            parts = node_key.split('-')
            if len(parts) != 4:
                continue
                
            depth, x, y, z = map(int, parts)
            
            # Calculate node bounds
            scale = 2 ** depth
            node_minx = bounds[0] + (bounds[3] - bounds[0]) * x / scale
            node_maxx = bounds[0] + (bounds[3] - bounds[0]) * (x + 1) / scale
            node_miny = bounds[1] + (bounds[4] - bounds[1]) * y / scale
            node_maxy = bounds[1] + (bounds[4] - bounds[1]) * (y + 1) / scale
            
            node_box = box(node_minx, node_miny, node_maxx, node_maxy)
            
            # Check intersection with AOI
            if not node_box.intersects(aoi_geom):
                continue
            
            # Try to get hierarchy file for this node
            hierarchy_url = f"{base_url}/ept-hierarchy/{node_key}.json"
            
            try:
                response = requests.get(hierarchy_url, timeout=5)
                if response.status_code == 200:
                    hierarchy_data = response.json()
                    
                    # Add nodes with points
                    for key, value in hierarchy_data.items():
                        if value > 0:  # Node has points
                            intersecting_nodes.append(key)
                            
                            # Add child nodes to queue for deeper exploration
                            if depth < 10:  # Limit depth
                                key_parts = key.split('-')
                                if len(key_parts) == 4:
                                    child_depth = int(key_parts[0])
                                    if child_depth == depth + 1:
                                        queue.append(key)
                                        
            except Exception as e:
                logger.debug(f"Could not fetch hierarchy for {node_key}: {e}")
        
        return intersecting_nodes
    
    def download_usgs_pointcloud(
        self,
        dataset_id: str,
        output_dir: str,
        max_workers: int = 5,
        max_tiles: int = 50,
    ) -> List[str]:
        """
        Download USGS 3DEP point cloud tiles that intersect the AOI.
        
        Parameters
        ----------
        dataset_id : str
            USGS dataset identifier (property_id from catalog)
        output_dir : str
            Directory to save downloaded tiles
        max_workers : int
            Number of parallel download threads
        max_tiles : int
            Maximum number of tiles to download
            
        Returns
        -------
        list
            List of paths to downloaded LAZ files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get AOI geometry in appropriate CRS
        aoi_geom = self.da.polygon.get("merged_polygon")
        if aoi_geom is None:
            raise ValueError("AOI polygon not defined")
        
        # USGS EPT base URL
        base_url = f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{dataset_id}"
        
        downloaded_files = []
        
        try:
            # Get EPT metadata
            response = requests.get(f"{base_url}/ept.json", timeout=10)
            response.raise_for_status()
            ept_meta = response.json()
            bounds = ept_meta.get('bounds', [])
            
            if len(bounds) < 6:
                logger.error(f"Invalid EPT bounds for {dataset_id}")
                return []
            
            # Transform AOI to EPT CRS if needed
            ept_srs = ept_meta.get('srs', {}).get('horizontal', 'EPSG:4326')
            try:
                transformer = Transformer.from_crs("EPSG:4326", ept_srs, always_xy=True)
                from shapely.ops import transform as shapely_transform
                aoi_transformed = shapely_transform(transformer.transform, aoi_geom)
            except Exception:
                aoi_transformed = aoi_geom
            
            # Get intersecting nodes
            nodes = self.get_ept_hierarchy_nodes(base_url, bounds, aoi_transformed, max_nodes=max_tiles)
            logger.info(f"Found {len(nodes)} EPT nodes intersecting AOI")
            
            if not nodes:
                logger.warning(f"No intersecting nodes found for {dataset_id}")
                return []
            
            # Download LAZ files in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for node in nodes[:max_tiles]:
                    laz_url = f"{base_url}/ept-data/{node}.laz"
                    output_path = output_dir / f"{node}.laz"
                    futures.append(executor.submit(self.download_file, laz_url, output_path))
                
                for future in as_completed(futures):
                    file_metadata = future.result()
                    if file_metadata['success']:
                        downloaded_files.append(file_metadata['path'])
            
            logger.info(f"Downloaded {len(downloaded_files)} tiles for {dataset_id}")
            
        except Exception as e:
            logger.error(f"Error downloading USGS dataset {dataset_id}: {e}")
        
        return downloaded_files
    
    def download_noaa_pointcloud(
        self,
        dataset_id: str,
        output_dir: str,
        max_workers: int = 5,
        max_tiles: int = 50,
    ) -> List[str]:
        """
        Download NOAA Coastal point cloud tiles that intersect the AOI.
        
        Parameters
        ----------
        dataset_id : str
            NOAA dataset identifier (property_id from catalog)
        output_dir : str
            Directory to save downloaded tiles
        max_workers : int
            Number of parallel download threads
        max_tiles : int
            Maximum number of tiles to download
            
        Returns
        -------
        list
            List of paths to downloaded LAZ files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        aoi_geom = self.da.polygon.get("merged_polygon")
        if aoi_geom is None:
            raise ValueError("AOI polygon not defined")
        
        downloaded_files = []
        
        # Get EPT URL from STAC
        stac_url = f"https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/entwine/stac/DigitalCoast_mission_{dataset_id}.json"
        
        try:
            response = requests.get(stac_url, timeout=10)
            if response.status_code != 200:
                logger.error(f"STAC not found for NOAA dataset {dataset_id}")
                return []
            
            stac_data = response.json()
            
            if 'assets' not in stac_data or 'ept' not in stac_data['assets']:
                logger.error(f"No EPT asset in STAC for {dataset_id}")
                return []
            
            ept_url = stac_data['assets']['ept']['href']
            base_url = ept_url.replace('/ept.json', '')
            
            # Get EPT metadata
            response = requests.get(ept_url, timeout=10)
            response.raise_for_status()
            ept_meta = response.json()
            bounds = ept_meta.get('bounds', [])
            
            if len(bounds) < 6:
                logger.error(f"Invalid EPT bounds for {dataset_id}")
                return []
            
            # Transform AOI to EPT CRS if needed
            ept_srs = ept_meta.get('srs', {}).get('horizontal', 'EPSG:4326')
            try:
                transformer = Transformer.from_crs("EPSG:4326", ept_srs, always_xy=True)
                from shapely.ops import transform as shapely_transform
                aoi_transformed = shapely_transform(transformer.transform, aoi_geom)
            except Exception:
                aoi_transformed = aoi_geom
            
            # Get intersecting nodes
            nodes = self.get_ept_hierarchy_nodes(base_url, bounds, aoi_transformed, max_nodes=max_tiles)
            logger.info(f"Found {len(nodes)} EPT nodes intersecting AOI")
            
            if not nodes:
                logger.warning(f"No intersecting nodes found for {dataset_id}")
                return []
            
            # Download LAZ files in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for node in nodes[:max_tiles]:
                    laz_url = f"{base_url}/ept-data/{node}.laz"
                    output_path = output_dir / f"{node}.laz"
                    futures.append(executor.submit(self.download_file, laz_url, output_path))
                
                for future in as_completed(futures):
                    file_metadata = future.result()
                    if file_metadata['success']:
                        downloaded_files.append(file_metadata['path'])
            
            logger.info(f"Downloaded {len(downloaded_files)} tiles for {dataset_id}")
            
        except Exception as e:
            logger.error(f"Error downloading NOAA dataset {dataset_id}: {e}")
        
        return downloaded_files
    
    def download_ot_pointcloud(
        self,
        short_name: str,
        output_dir: str,
        max_workers: int = 5,
        dataset_epsg: Optional[int] = None,
    ) -> List[str]:
        """
        Download OpenTopography point cloud tiles via S3.
        
        This method downloads tiles from OpenTopography's S3 bucket by:
        1. Finding the tile index shapefile
        2. Identifying tiles that intersect the AOI
        3. Downloading those tiles
        
        Parameters
        ----------
        short_name : str
            Dataset short name (alternateName from catalog)
        output_dir : str
            Directory to save downloaded tiles
        max_workers : int
            Number of parallel download threads
        dataset_epsg : int, optional
            EPSG code for the dataset CRS. Used for spatial filtering when
            no tile index is available.
            
        Returns
        -------
        list
            List of paths to downloaded LAZ files
        """
        if not _BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for downloading OpenTopography datasets. "
                "Install with: pip install boto3"
            )
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        aoi_geom = self.da.polygon.get("merged_polygon")
        if aoi_geom is None:
            raise ValueError("AOI polygon not defined")
        
        downloaded_files = []
        
        # Setup S3 client for OpenTopography
        s3_config = Config(
            signature_version=UNSIGNED,
            retries={'max_attempts': 3, 'mode': 'standard'},
            s3={'addressing_style': 'path'}
        )
        s3_client = boto3.client(
            's3', 
            endpoint_url='https://opentopography.s3.sdsc.edu', 
            config=s3_config
        )
        
        try:
            # FIRST: Do a targeted search for tile index files at the root level
            tile_index_candidates = []
            subdirs = []
            
            # List just the first level (no recursive deep dive yet)
            root_response = s3_client.list_objects_v2(
                Bucket='pc-bulk',
                Prefix=f"{short_name}/",
                Delimiter='/',
                MaxKeys=1000
            )
            
            # Check direct files at root level for tile index
            for obj in root_response.get('Contents', []):
                key = obj['Key']
                filename = Path(key).name.lower()
                
                if filename.endswith('.zip'):
                    if any(pattern in filename for pattern in [
                        'tileindex', 'tile_index', 'tiles', 'index',
                        'footprint', 'boundary', 'extent', 'grid'
                    ]):
                        tile_index_candidates.append(('zip', key))
                        logger.info(f"Found tile index candidate: {key}")
                elif filename.endswith('.shp'):
                    if any(pattern in filename for pattern in [
                        'tileindex', 'tile_index', 'tiles', 'index',
                        'footprint', 'boundary', 'extent', 'grid'
                    ]):
                        tile_index_candidates.append(('shp', key))
                        logger.info(f"Found tile index candidate: {key}")
            
            # Get subdirectories
            subdirs = [p['Prefix'] for p in root_response.get('CommonPrefixes', [])]
            if subdirs:
                logger.info(f"Found subdirectories: {subdirs}")
            
            # Log root level files
            root_files = [Path(obj['Key']).name for obj in root_response.get('Contents', [])]
            if root_files:
                logger.info(f"Root level files: {root_files}")
            
            # Try each tile index candidate
            tile_index_found = False
            for idx_type, tile_index_key in tile_index_candidates:
                if tile_index_found:
                    break
                    
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        
                        if idx_type == 'zip':
                            zip_path = temp_path / 'tileindex.zip'
                            logger.info(f"Downloading tile index zip: {tile_index_key}")
                            s3_client.download_file('pc-bulk', tile_index_key, str(zip_path))
                            
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(temp_path)
                            
                            shp_files = list(temp_path.rglob('*.shp'))
                        else:
                            logger.info(f"Downloading tile index shapefile: {tile_index_key}")
                            base_key = tile_index_key[:-4]
                            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                                try:
                                    s3_client.download_file('pc-bulk', base_key + ext,
                                                           str(temp_path / (Path(base_key).name + ext)))
                                except Exception:
                                    pass
                            shp_files = list(temp_path.glob('*.shp'))
                        
                        if shp_files:
                            logger.info(f"Reading shapefile: {shp_files[0].name}")
                            tiles_gdf = gpd.read_file(shp_files[0])
                            logger.info(f"Tile index columns: {list(tiles_gdf.columns)}")
                            tiles_gdf = tiles_gdf.to_crs('EPSG:4326')
                            
                            intersecting_tiles = tiles_gdf[tiles_gdf.intersects(aoi_geom)]
                            logger.info(f"Found {len(intersecting_tiles)} tiles intersecting with AOI "
                                       f"(out of {len(tiles_gdf)} total)")
                            
                            if len(intersecting_tiles) > 0:
                                tile_index_found = True
                                
                                def download_s3_file(s3_key, local_path):
                                    try:
                                        if local_path.exists():
                                            return str(local_path)
                                        logger.info(f"Downloading: {s3_key}")
                                        s3_client.download_file('pc-bulk', s3_key, str(local_path))
                                        return str(local_path)
                                    except Exception as e:
                                        logger.error(f"Failed to download {s3_key}: {e}")
                                        return None
                                
                                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                    futures = []
                                    
                                    for idx, tile in intersecting_tiles.iterrows():
                                        # Initialize variables for this tile
                                        url_val = None
                                        filename = None
                                        s3_key = None
                                        
                                        # Check URL column first - parse full S3 path
                                        if 'URL' in tile.index and pd.notna(tile['URL']):
                                            url_val = str(tile['URL']).strip()
                                            # Parse the URL to extract S3 key
                                            # URL format: https://opentopography.s3.sdsc.edu/pc-bulk/dataset/subdir/file.las
                                            if 'pc-bulk/' in url_val:
                                                s3_key = url_val.split('pc-bulk/')[-1]
                                                filename = Path(s3_key).name
                                                logger.info(f"Extracted S3 key from URL: {s3_key}")
                                        elif 'url' in tile.index and pd.notna(tile['url']):
                                            url_val = str(tile['url']).strip()
                                            if 'pc-bulk/' in url_val:
                                                s3_key = url_val.split('pc-bulk/')[-1]
                                                filename = Path(s3_key).name
                                        
                                        # If no URL or no s3_key from URL, try filename columns
                                        if not filename:
                                            for field in ['Filename', 'filename', 'file', 'name', 'file_name',
                                                         'tilename', 'File', 'Name']:
                                                if field in tile.index:
                                                    val = tile[field]
                                                    if pd.notna(val) and str(val).strip():
                                                        filename = Path(str(val)).name
                                                        break
                                        
                                        if not filename or not filename.lower().endswith(('.laz', '.las')):
                                            continue
                                        
                                        # If we don't have s3_key from URL, search for it
                                        if not s3_key:
                                            s3_key = f"{short_name}/{filename}"
                                            
                                            # Check if file exists at root level
                                            try:
                                                s3_client.head_object(Bucket='pc-bulk', Key=s3_key)
                                            except:
                                                # Search in subdirectories
                                                found_in_subdir = False
                                                for subdir in subdirs:
                                                    subdir_key = f"{subdir}{filename}"
                                                    try:
                                                        s3_client.head_object(Bucket='pc-bulk', Key=subdir_key)
                                                        s3_key = subdir_key
                                                        found_in_subdir = True
                                                        logger.debug(f"Found {filename} in {subdir}")
                                                        break
                                                    except:
                                                        continue
                                                
                                                if not found_in_subdir:
                                                    logger.warning(f"Could not find {filename} in any subdirectory")
                                                    continue
                                        
                                        output_path = output_dir / filename
                                        futures.append(executor.submit(download_s3_file, s3_key, output_path))
                                    
                                    for future in as_completed(futures):
                                        result = future.result()
                                        if result:
                                            downloaded_files.append(result)
                            else:
                                logger.warning(f"No tiles intersect AOI in {Path(tile_index_key).name}")
                        else:
                            logger.warning(f"No shapefile found in {tile_index_key}")
                except Exception as e:
                    logger.warning(f"Failed to process tile index {tile_index_key}: {e}")
                    continue
            
            # Fallback if no tile index worked
            if not tile_index_found:
                if not tile_index_candidates:
                    logger.warning(f"No tile index files found for {short_name}")
                
                # List LAZ files for fallback filtering
                if dataset_epsg:
                    logger.info("Falling back to filename-based tile filtering...")
                    logger.info("Listing all LAZ files in bucket...")
                    
                    laz_files_in_bucket = []
                    paginator = s3_client.get_paginator('list_objects_v2')
                    page_iterator = paginator.paginate(
                        Bucket='pc-bulk',
                        Prefix=f"{short_name}/",
                        PaginationConfig={'MaxItems': 50000}
                    )
                    
                    for page in page_iterator:
                        for obj in page.get('Contents', []):
                            key = obj['Key']
                            if key.lower().endswith(('.laz', '.las')):
                                laz_files_in_bucket.append(key)
                    
                    logger.info(f"Found {len(laz_files_in_bucket)} LAZ files in bucket")
                    
                    if laz_files_in_bucket:
                        try:
                            transformer = Transformer.from_crs(
                                "EPSG:4326", f"EPSG:{dataset_epsg}", 
                                always_xy=True
                            )
                            aoi_bounds_4326 = aoi_geom.bounds
                            aoi_min_transformed = transformer.transform(aoi_bounds_4326[0], aoi_bounds_4326[1])
                            aoi_max_transformed = transformer.transform(aoi_bounds_4326[2], aoi_bounds_4326[3])
                            
                            aoi_minx = min(aoi_min_transformed[0], aoi_max_transformed[0])
                            aoi_miny = min(aoi_min_transformed[1], aoi_max_transformed[1])
                            aoi_maxx = max(aoi_min_transformed[0], aoi_max_transformed[0])
                            aoi_maxy = max(aoi_min_transformed[1], aoi_max_transformed[1])
                            
                            logger.info(f"AOI in EPSG:{dataset_epsg}: "
                                       f"({aoi_minx:.0f}, {aoi_miny:.0f}) - ({aoi_maxx:.0f}, {aoi_maxy:.0f})")
                            
                            coord_pattern = re.compile(r'(\d{5,7})_(\d{6,8})\.la[sz]$', re.IGNORECASE)
                            
                            matching_files = []
                            tile_size = 1000
                            
                            for key in laz_files_in_bucket:
                                filename = Path(key).name
                                match = coord_pattern.search(filename)
                                if match:
                                    tile_easting = float(match.group(1))
                                    tile_northing = float(match.group(2))
                                    
                                    tile_minx = tile_easting
                                    tile_miny = tile_northing
                                    tile_maxx = tile_easting + tile_size
                                    tile_maxy = tile_northing + tile_size
                                    
                                    if (tile_maxx >= aoi_minx and tile_minx <= aoi_maxx and
                                        tile_maxy >= aoi_miny and tile_miny <= aoi_maxy):
                                        matching_files.append(key)
                            
                            if matching_files:
                                logger.info(f"Found {len(matching_files)} tiles intersecting AOI")
                                for key in matching_files:
                                    filename = Path(key).name
                                    output_path = output_dir / filename
                                    try:
                                        if not output_path.exists():
                                            logger.info(f"Downloading: {key}")
                                            s3_client.download_file('pc-bulk', key, str(output_path))
                                        downloaded_files.append(str(output_path))
                                    except Exception as e:
                                        logger.error(f"Failed to download {key}: {e}")
                            else:
                                logger.warning(f"No tiles found intersecting AOI")
                                
                        except Exception as e:
                            logger.error(f"Failed to filter tiles by coordinates: {e}")
                else:
                    logger.warning("No dataset_epsg provided, cannot spatially filter tiles")
                            
        except Exception as e:
            logger.error(f"Error downloading OpenTopography dataset {short_name}: {e}")
        
        logger.info(f"Downloaded {len(downloaded_files)} files for {short_name}")
        return downloaded_files

                    
    # --------------------Internal PDAL helpers-------------------------
    @staticmethod
    def _writer_gdal(filename, *, grid_method="idw", res=1.0, driver="GTiff"):
        """
        Construct a PDAL GDAL writer stage for gridding point cloud data.

        This helper builds a dictionary describing a ``writers.gdal`` stage
        compatible with PDAL pipelines.  It specifies output file name,
        gridding method (e.g. inverse distance weighted or maximum), pixel
        resolution and radius, nodata value, compression, and tiling options.

        Parameters
        ----------
        filename : str
            The destination filename for the output raster (including
            extension).
        grid_method : str, default 'idw'
            The gridding algorithm used to convert points to raster.  Typical
            values include ``'idw'`` for inverse distance weighting and
            ``'max'`` for maximum value gridding.
        res : float, default 1.0
            Output pixel size in map units (meters).
        driver : str, default 'GTiff'
            GDAL driver used to write the raster.  ``'GTiff'`` yields a
            GeoTIFF.

        Returns
        -------
        dict
            A dictionary suitable for inclusion in a PDAL pipeline,
            representing a GDAL writer stage.
        """
        return {
            "type": "writers.gdal",
            "filename": filename,
            "gdaldriver": driver,
            "nodata": -9999,
            "output_type": grid_method,
            "resolution": float(res),
            "radius": 2 * float(res),
            "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES",
        }

    @staticmethod
    def _writer_las(name, ext):
        """
        Construct a PDAL LAS/LAZ writer stage for saving point clouds.

        Parameters
        ----------
        name : str
            Base filename (without extension) for the output point cloud.
        ext : {'las', 'laz'}
            Desired output format.  ``'las'`` produces an uncompressed LAS
            file while ``'laz'`` triggers LASzip compression.

        Returns
        -------
        dict
            A dictionary describing a ``writers.las`` stage for PDAL.

        Raises
        ------
        ValueError
            If ``ext`` is not one of ``'las'`` or ``'laz'``.
        """
        if ext not in {"las", "laz"}:
            raise ValueError("pc_outType must be 'las' or 'laz'")
        w = {"type": "writers.las", "filename": f"{name}.{ext}"}
        if ext == "laz":
            w["compression"] = "laszip"
        return w

    # ----------------------- Local file pipelines ---------------------
    @staticmethod
    def build_pdal_pipeline_from_file(filename, extent, filterNoise=False, reclassify=False, savePointCloud=True, outCRS='EPSG:3857', 
                            pc_outName='filter_test', pc_outType='laz'):
        """
        Construct a PDAL pipeline to read, optionally filter/reclassify, and
        reproject a local point cloud.

        This method builds a list of PDAL stages beginning with reading
        LAS/LAZ data from disk and cropping it to the supplied ``extent``.
        Optional stages remove noise classes 7 and 18, apply SMRF ground
        classification, reclassify ground points, and reproject the point
        cloud into a target CRS.  Finally, the pipeline may include a
        ``writers.las`` stage to save the filtered/reprojected cloud.

        Parameters
        ----------
        filename : str
            Path to a LAS or LAZ file to be processed.
        extent : shapely.geometry.Polygon
            Polygon in the same CRS as the input point cloud used to crop
            the data.
        filterNoise : bool, default False
            Remove noise points classified as 7 (low noise) or 18 (high
            noise).
        reclassify : bool, default False
            Apply SMRF ground classification and restrict the cloud to
            ground returns.
        savePointCloud : bool, default True
            When true, append a ``writers.las`` stage to write the processed
            point cloud.
        outCRS : str, default 'EPSG:3857'
            Desired output coordinate reference system.
        pc_outName : str, default 'filter_test'
            Base filename used when saving the filtered point cloud.
        pc_outType : {'las', 'laz'}, default 'laz'
            File format for the output point cloud.

        Returns
        -------
        list of dict
            A PDAL pipeline specification representing the requested
            operations.
        """
        # Initialize the pipeline with reading and cropping stages
        pointcloud_pipeline = [
            {
                "type": "readers.las",
                "filename": filename
            },
            {
                "type": "filters.crop",
                "polygon": extent.wkt
            }
        ]
        
        # Optionally add a noise filter stage
        if filterNoise:
            pointcloud_pipeline.append({
                "type": "filters.range",
                "limits": "Classification![7:7], Classification![18:18]"
            })
        
        # Optionally add reclassification stages
        if reclassify:
            pointcloud_pipeline += [
                {"type": "filters.assign", "value": "Classification = 0"},
                {"type": "filters.smrf"},
                {"type": "filters.range", "limits": "Classification[2:2]"}
            ]
        
        # Add reprojection stage
        pointcloud_pipeline.append({
            "type": "filters.reprojection",
            "out_srs": outCRS,
        })
        
        # Optionally add a save point cloud stage
        if savePointCloud:
            if pc_outType not in ['las', 'laz']:
                raise Exception("pc_outType must be 'las' or 'laz'.")
            
            writer_stage = {
                "type": "writers.las",
                "filename": f"{pc_outName}.{pc_outType}"
            }
            if pc_outType == 'laz':
                writer_stage["compression"] = "laszip"
            
            pointcloud_pipeline.append(writer_stage)
            
        return pointcloud_pipeline
    
    def make_DEM_pipeline_from_file(self, filename, extent, dem_resolution,
                        filterNoise=True, reclassify=False, savePointCloud=False, outCRS='EPSG:3857',
                        pc_outName='filter_test', pc_outType='laz', demType='dtm', gridMethod='idw', 
                        dem_outName='dem_test', dem_outExt='tif', driver="GTiff"):
        """
        Build a PDAL pipeline to convert a local point cloud into a DEM.

        This method wraps :meth:`build_pdal_pipeline_from_file` and then
        appends stages to generate a Digital Terrain Model (DTM) or
        Digital Surface Model (DSM) from the cropped/filtered point cloud.
        Ground points are optionally selected when ``demType`` is ``'dtm'``.

        Parameters
        ----------
        filename : str
            Path to the input LAS/LAZ file.
        extent : shapely.geometry.Polygon
            Clipping polygon in the input CRS.
        dem_resolution : float
            Pixel size in map units for the output raster.
        filterNoise : bool, default True
            Remove noise classes (7 and 18) prior to gridding.
        reclassify : bool, default False
            Reclassify points using SMRF and retain only ground returns.
        savePointCloud : bool, default False
            Write the intermediate point cloud to disk.
        outCRS : str, default 'EPSG:3857'
            Output CRS for the DEM.
        pc_outName : str, default 'filter_test'
            Base filename for the intermediate point cloud.
        pc_outType : {'las', 'laz'}, default 'laz'
            Format for the intermediate point cloud.
        demType : {'dtm', 'dsm'}, default 'dtm'
            Type of DEM to produce.  ``'dtm'`` filters ground points,
            whereas ``'dsm'`` uses all returns.
        gridMethod : str, default 'idw'
            Gridding algorithm (e.g. ``'idw'`` or ``'max'``) used by
            ``writers.gdal``.
        dem_outName : str, default 'dem_test'
            Base name for the output DEM.
        dem_outExt : str, default 'tif'
            File extension for the output raster.
        driver : str, default 'GTiff'
            GDAL driver used to create the DEM.

        Returns
        -------
        dict
            A PDAL pipeline dictionary describing the steps to generate the DEM.
        """
        # Build the base point cloud pipeline using the provided parameters
        pointcloud_pipeline = self.build_pdal_pipeline_from_file(filename, extent, filterNoise, reclassify, savePointCloud, outCRS, pc_outName, pc_outType)
        
        # Prepare the base pipeline dictionary
        dem_pipeline = {
            "pipeline": pointcloud_pipeline
        }

        # Add appropriate stages based on DEM type
        if demType == 'dsm':
            # Directly add the DSM writer stage
            dem_pipeline['pipeline'].append({
                "type": "writers.gdal",
                "filename": f"{dem_outName}.{dem_outExt}",
                "gdaldriver": driver,
                "nodata": -9999,
                "output_type": gridMethod,
                "resolution": float(dem_resolution),
                "radius": 2*float(dem_resolution),
                "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES",
                "override_srs": outCRS
            })
        
        elif demType == 'dtm':
            # Add a filter to keep only ground points
            dem_pipeline['pipeline'].append({
                "type": "filters.range",
                "limits": "Classification[2:2]"
            })

            # Add the DTM writer stage
            dem_pipeline['pipeline'].append({
                "type": "writers.gdal",
                "filename": f"{dem_outName}.{dem_outExt}",
                "gdaldriver": driver,
                "nodata": -9999,
                "output_type": gridMethod,
                "resolution": float(dem_resolution),
                "radius": 2*float(dem_resolution),
                "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES",
                "override_srs": outCRS
            })
        else:
            raise Exception("demType must be 'dsm' or 'dtm'.")
        
        return dem_pipeline

    # ----------------------- AWS EPT pipeline helpers -----------------
    @staticmethod
    def build_aws_pdal_pipeline(extent_epsg3857, property_ids, pc_resolution, data_source, filterNoise = False,
                            reclassify = False, savePointCloud = True, outCRS = 'EPSG:3857', pc_outName = 'filter_test', 
                            pc_outType = 'laz'):
        """
        Build a PDAL pipeline for downloading and processing AWS-hosted EPT data.

        Given a list of dataset identifiers, this function constructs a
        ``readers.ept`` stage for each ID, clips the point cloud to the
        provided polygon in EPSG:3857 coordinates, and optionally filters
        noise, reclassifies ground points, reprojects to a target CRS, and
        writes the resulting point cloud to disk.  The function supports
        USGS and NOAA data sources; other providers are not currently
        accepted.

        Parameters
        ----------
        extent_epsg3857 : shapely.geometry.Polygon
            AOI polygon expressed in Web Mercator (EPSG:3857) coordinates.
        property_ids : list of str
            Identifiers for the EPT datasets to read from S3 buckets or
            NOAA's Digital Coast STAC.
        pc_resolution : float
            Sampling resolution for the ``readers.ept`` stage.  Larger values
            will subsample the point cloud more aggressively.
        data_source : {'usgs', 'noaa'}
            Indicates which repository to fetch the data from.  For USGS,
            the EPT URL is constructed directly; for NOAA, the STAC
            catalog is queried to obtain the EPT link.
        filterNoise : bool, default False
            Remove noise classes 7 and 18 prior to further processing.
        reclassify : bool, default False
            Apply SMRF classification and select only ground points.
        savePointCloud : bool, default True
            Write the reprojected point cloud to disk.
        outCRS : str, default 'EPSG:3857'
            CRS to which the point cloud should be reprojected.
        pc_outName : str, default 'filter_test'
            Base filename used when saving the output point cloud.
        pc_outType : {'las', 'laz'}, default 'laz'
            File extension for the output point cloud.  ``'laz'`` triggers
            compression via LASzip.

        Returns
        -------
        dict
            A PDAL pipeline dictionary describing the EPT read and optional
            processing stages.
        """
        readers = []
        for id in property_ids:
            if data_source == 'usgs':
                url = f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{id}/ept.json"
            elif data_source == 'noaa':
                stac_url = f"https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/entwine/stac/DigitalCoast_mission_{id}.json"
                response = requests.get(stac_url)
                data = response.json()
                url = data['assets']['ept']['href']
            else:
                raise ValueError("Invalid dataset source. Must be 'usgs' or 'noaa'.")

            reader = {
                "type": "readers.ept",
                "filename": str(url),
                "polygon": str(extent_epsg3857),
                "requests": 3,
                "resolution": pc_resolution
            }
            readers.append(reader)
            
        pointcloud_pipeline = {
                "pipeline":
                    readers
        }
        
        if filterNoise == True:
            
            filter_stage = {
                "type":"filters.range",
                "limits":"Classification![7:7], Classification![18:18]"
            }
            
            pointcloud_pipeline['pipeline'].append(filter_stage)
        
        if reclassify == True:
            
            remove_classes_stage = {
                "type":"filters.assign",
                "value":"Classification = 0"
            }
            
            classify_ground_stage = {
                "type":"filters.smrf"
            }
            
            reclass_stage = {
                "type":"filters.range",
                "limits":"Classification[2:2]"
            }

            
            pointcloud_pipeline['pipeline'].append(remove_classes_stage)
            pointcloud_pipeline['pipeline'].append(classify_ground_stage)
            pointcloud_pipeline['pipeline'].append(reclass_stage)
            
        reprojection_stage = {
            "type":"filters.reprojection",
            "out_srs":outCRS,
        }
        
        pointcloud_pipeline['pipeline'].append(reprojection_stage)
        
        if savePointCloud == True:
            
            if pc_outType == 'las':
                savePC_stage = {
                    "type": "writers.las",
                    "filename": str(pc_outName)+'.'+ str(pc_outType),
                }
            elif pc_outType == 'laz':    
                savePC_stage = {
                    "type": "writers.las",
                    "compression": "laszip",
                    "filename": str(pc_outName)+'.'+ str(pc_outType),
                }
            else:
                raise Exception("pc_outType must be 'las' or 'laz'.")

            pointcloud_pipeline['pipeline'].append(savePC_stage)
            
        return pointcloud_pipeline
    
    def make_DEM_pipeline_aws(self, extent_epsg3857, property_ids, pc_resolution, dem_resolution, data_source = "usgs",
                        filterNoise = True, reclassify = True, savePointCloud = False, outCRS = 'EPSG:3857',
                        pc_outName = 'filter_test', pc_outType = 'laz', demType = 'dtm', gridMethod = 'idw',
                        dem_outName = 'dem_test', dem_outExt = 'tif', driver = "GTiff"):
        """Build a PDAL pipeline to create a DEM from AWS-hosted point cloud data.

        This method wraps :func:`build_aws_pdal_pipeline` and appends additional
        stages to generate a Digital Terrain Model (DTM) or Digital Surface
        Model (DSM) from Entwine Point Tiles.  Depending on the ``demType``
        argument, it optionally filters ground returns before gridding.

        Parameters
        ----------
        extent_epsg3857 : shapely.geometry.Polygon
            Cropping polygon in EPSG:3857 coordinates.
        property_ids : list of str
            Identifiers for the point cloud datasets to read.
        pc_resolution : float
            Resolution used when selecting points from the EPT.
        dem_resolution : float
            Output pixel size for the DEM.
        data_source : {'usgs', 'noaa'}, default 'usgs'
            Which provider the EPT data originates from.
        filterNoise : bool, default True
            Remove noise classes (7 and 18) before gridding.
        reclassify : bool, default True
            Apply ground classification via SMRF before gridding.
        savePointCloud : bool, default False
            Save the intermediate point cloud to disk.
        outCRS : str, default 'EPSG:3857'
            Target CRS for the output DEM.
        pc_outName : str, default 'filter_test'
            Base filename for the intermediate point cloud.
        pc_outType : {'las', 'laz'}, default 'laz'
            Format for the intermediate point cloud.
        demType : {'dtm', 'dsm'}, default 'dtm'
            Type of DEM to generate: ``'dtm'`` for ground-only or ``'dsm'``
            for first-return surfaces.
        gridMethod : str, default 'idw'
            Gridding algorithm to use in the GDAL writer stage.
        dem_outName : str, default 'dem_test'
            Base name for the output DEM (without extension).
        dem_outExt : str, default 'tif'
            File extension for the output DEM.
        driver : str, default 'GTiff'
            GDAL driver used to write the DEM.

        Returns
        -------
        dict
            A PDAL pipeline dictionary describing the steps to produce the DEM.
        """

        dem_pipeline = self.build_aws_pdal_pipeline(extent_epsg3857, property_ids, pc_resolution, data_source,
                                                filterNoise, reclassify, savePointCloud, outCRS, pc_outName, pc_outType)
        
        
        if demType == 'dsm':
            dem_stage = {
                    "type":"writers.gdal",
                    "filename":str(dem_outName)+ '.' + str(dem_outExt),
                    "gdaldriver":driver,
                    "nodata":-9999,
                    "output_type":gridMethod,
                    "resolution":float(dem_resolution),
                    "gdalopts":"COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES",
                    "override_srs": outCRS
            }
        
        elif demType == 'dtm':
            groundfilter_stage = {
                    "type":"filters.range",
                    "limits":"Classification[2:2]"
            }

            dem_pipeline['pipeline'].append(groundfilter_stage)

            dem_stage = {
                    "type":"writers.gdal",
                    "filename":str(dem_outName)+ '.' + str(dem_outExt),
                    "gdaldriver":driver,
                    "nodata":-9999,
                    "output_type":gridMethod,
                    "resolution":float(dem_resolution),
                    "gdalopts":"COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES",
                    "override_srs": outCRS
            }
        
        else:
            raise Exception("demType must be 'dsm' or 'dtm'.")
            
            
        dem_pipeline['pipeline'].append(dem_stage)
        
        return dem_pipeline

    # --------------------- Native UTM from AOI bounds ------------------
    
    @staticmethod
    def native_utm_crs_from_aoi_bounds(bounds,datum):
        """
        Get the native UTM coordinate reference system from the 

        :param bounds: shapely Polygon of bounding box in EPSG:4326 CRS
        :param datum: string with datum name (e.g., "WGS84")
        :return: UTM CRS code
        """
        utm_crs_list = query_utm_crs_info(
            datum_name=datum,
            area_of_interest=AreaOfInterest(
                west_lon_degree=bounds["west"],
                south_lat_degree=bounds["south"],
                east_lon_degree=bounds["east"],
                north_lat_degree=bounds["north"],
            ),
        )
        utm_crs = CRS.from_epsg(utm_crs_list[0].code)
        return utm_crs

    
    # Add the transform_options parameter with a default value of None
    @staticmethod
    def reproject_polygon(
        polygon: Polygon | MultiPolygon,
        source_crs: int | str | CRS,
        target_crs: int | str | CRS,
        transform_options: dict = None
    ) -> Polygon | MultiPolygon:
        """Reproject a shapely Polygon or MultiPolygon to a new CRS."""
        
        # Helper function to handle the optional dictionary
        def _if_not_none(value, default):
            return default if value is None else value

        # Pass the unpacked options dictionary to the transformer
        proj_transformer = Transformer.from_crs(
            source_crs,
            target_crs,
            always_xy=True,
            **_if_not_none(transform_options, {})
        )
        from shapely.ops import transform
        return transform(proj_transformer.transform, polygon)
        
    # -------- End‑to‑end driver to download data & create DEMs---------

    def dem_download_workflow(
        self,
        folder,
        output_name,                    # Desired generic output name for files on user's local file system (w/o extension, modifiers like "_DTM", "_DSM" will be added depending on product created)
        API_Key,                        # OpenTopography Enterprise API key       
        dem_resolution = 1.0,           # Desired grid size (in meters) for output raster DEM
        dataset_type = "compare",       # Whether dataset is compare or reference dataset    
        epoch: Optional[date] = None,   # Optional epoch date for CRS, if not provided, will use current date
        filterNoise = True,             # Option to remove points from USGS Class 7 (Low Noise) and Class 18 (High Noise).
        reclassify = False,         
        savePointCloud = False,         
        pc_resolution = 0.1,            # The desired resolution of the pointcloud based on the following definition: 
                                        #        A point resolution limit to select, expressed as a grid cell edge length. 
                                        #        Units correspond to resource coordinate system units. For example, 
                                        #        for a coordinate system expressed in meters, a resolution value of 0.1 
                                        #        will select points up to a ground resolution of 100 points per square meter.
                                        #        The resulting resolution may not be exactly this value: the minimum possible 
                                        #        resolution that is at least as precise as the requested resolution will be selected. 
                                        #        Therefore the result may be a bit more precise than requested. 
                                        # Source: https://pdal.io/stages/readers.ept.html#readers-ept
        outCRS = "WGS84 UTM",           # Output coordinate reference systemt (CRS), specified by ESPG code (e.g., 3857 - Web Mercator)
        method="idw",                   # method for gap-filling
        nodata=-9999,                   # no data values
        max_dist=100,                   # max distance to consider for gap filling
        smooth_iter=0                   # number of smoothing iterations
        
    ):
        """Download point clouds and produce DEMs for a single dataset.

        This workflow coordinates the download of point cloud data (via the
        OpenTopography API, USGS, or NOAA), generation of DTM and DSM rasters
        using PDAL, and optional gap filling.  It handles coordinate
        reference system selection including epoch tags, writes outputs
        to disk, and stores the resulting file paths on the ``GetDEMs``
        instance.  Parameters mirror those of :func:`make_DEM_pipeline_from_file`
        and :func:`make_DEM_pipeline_aws` but additionally require an API
        key when fetching OT-hosted datasets.

        Parameters
        ----------
        folder : str
            Directory in which to save downloaded point clouds and generated DEMs.
        output_name : str
            Base filename used when saving outputs (suffixes ``_DTM`` and
            ``_DSM`` will be appended).
        API_Key : str
            Enterprise API key for accessing OpenTopography-hosted datasets.
        dem_resolution : float, default 1.0
            Grid spacing for the output DEMs in map units.
        dataset_type : {'compare', 'reference'}, default 'compare'
            Indicates whether the dataset being processed corresponds to the
            ``compare`` or ``reference`` survey.
        epoch : datetime.date or None, optional
            Epoch date to embed in the CRS of the output DEM.  If omitted,
            the current date is used.
        filterNoise : bool, default True
            Remove noise classes 7 and 18 when gridding.
        reclassify : bool, default False
            Apply SMRF ground classification prior to gridding.
        savePointCloud : bool, default False
            Save the filtered point cloud to disk.
        pc_resolution : float, default 0.1
            Resolution for sampling points from EPT sources (in units of the
            source CRS).  This influences the density of the downloaded
            point cloud.
        outCRS : str, default 'WGS84 UTM'
            Output CRS for the DEM.  If set to ``"WGS84 UTM"``, a native UTM
            zone is chosen based on the AOI.
        method : str, default 'idw'
            Gap-filling interpolation method (see :func:`fill_no_data`).
        nodata : int or float, default -9999
            NoData value to assign in the output rasters.
        max_dist : float, default 100
            Maximum search distance for gap filling.
        smooth_iter : int, default 0
            Number of smoothing iterations for gap filling.

        Returns
        -------
        None
            The function operates via side effects: rasters are written to
            disk and attributes on the ``GetDEMs`` object are updated to
            reference the generated DEMs.
        """

        self.initial_compare_dataset_crs = int(self.ot.compare_horizontal_crs)
        self.initial_reference_dataset_crs = int(self.ot.reference_horizontal_crs)

        self.target_compare_dataset_crs = self.native_utm_crs_from_aoi_bounds(self.da.bounds,"WGS84").to_epsg()
        self.target_reference_dataset_crs = self.native_utm_crs_from_aoi_bounds(self.da.bounds,"WGS84").to_epsg()

        self.bounds_polygon_epsg_initial_compare_crs = self.reproject_polygon(self.da.polygon["merged_polygon"], 4326, self.initial_compare_dataset_crs)
        self.bounds_polygon_epsg_initial_reference_crs = self.reproject_polygon(self.da.polygon["merged_polygon"], 4326, self.initial_reference_dataset_crs)
        
    
        if dataset_type == "compare":
            bounds_polygon_epsg_initial_crs = self.bounds_polygon_epsg_initial_compare_crs
            data_source_ = self.ot.compare_data_source
            dataset_id = self.ot.compare_property_id
            dataset_crs_ = self.target_compare_dataset_crs
            self.compare_dtm_path = folder+output_name+'_'+dataset_type+'_DTM.tif'
            self.compare_dsm_path = folder+output_name+'_'+dataset_type+'_DSM.tif'
               
        elif dataset_type == "reference":
            data_source_ = self.ot.reference_data_source
            dataset_id = self.ot.reference_property_id
            bounds_polygon_epsg_initial_crs = self.bounds_polygon_epsg_initial_reference_crs
            dataset_crs_ = self.target_reference_dataset_crs
            self.reference_dtm_path = folder+output_name+'_'+dataset_type+'_DTM.tif'
            self.reference_dsm_path = folder+output_name+'_'+dataset_type+'_DSM.tif'
            
        else:
            raise ValueError("dataset_type must be either 'compare' or 'reference'")

        if outCRS == "WGS84 UTM":
            base_crs = CRS.from_epsg(dataset_crs_)
        else:
            base_crs = CRS.from_user_input(outCRS)

        # Create a new CRS definition with the epoch
        if epoch:
            try:
                # Convert the base CRS to a PROJ string
                proj_string = base_crs.to_proj4()
                # Calculate the epoch as a decimal year
                decimal_year = epoch.year + (epoch.timetuple().tm_yday - 1) / 365.25
                # Append the epoch parameter to the PROJ string
                final_out_crs_wkt = f"{proj_string} +epoch={decimal_year:.4f}"
                print(f"✔️ Using epoch {epoch} for {dataset_type} dataset via PROJ string.")
            except Exception as e:
                warnings.warn(f"Could not convert CRS to PROJ string to add epoch: {e}. Proceeding without epoch.")
                final_out_crs_wkt = base_crs.to_wkt()
        else:
            final_out_crs_wkt = base_crs.to_wkt()
            print(f"⚠️ No epoch provided for {dataset_type} dataset.")
        
        if data_source_ == 'ot':
            # Download OpenTopography data via S3 (no Enterprise API key required)
            # This uses the publicly accessible S3 bucket at opentopography.s3.sdsc.edu
            
            if not _BOTO3_AVAILABLE:
                raise ImportError(
                    "boto3 is required for downloading OpenTopography datasets. "
                    "Install with: pip install boto3"
                )
            
            # Get the short_name for S3 bucket path
            if dataset_type == "compare":
                short_name = getattr(self.ot, 'compare_short_name', None)
            else:
                short_name = getattr(self.ot, 'reference_short_name', None)
            
            if short_name is None or pd.isna(short_name):
                # Fallback: use clean_name or property_id
                if dataset_type == "compare":
                    short_name = getattr(self.ot, 'compare_clean_name', None) or dataset_id
                else:
                    short_name = getattr(self.ot, 'reference_clean_name', None) or dataset_id
                logger.warning(f"Short name not found in catalog, using: {short_name}")
            
            logger.info(f"Downloading OpenTopography dataset via S3: {short_name}")
            
            # Setup S3 client for OpenTopography (unsigned access)
            s3_config = Config(
                signature_version=UNSIGNED,
                retries={'max_attempts': 3, 'mode': 'standard'},
                s3={'addressing_style': 'path'}
            )
            s3_client = boto3.client(
                's3', 
                endpoint_url='https://opentopography.s3.sdsc.edu', 
                config=s3_config
            )
            
            # Create output directory for point cloud tiles
            pc_output_dir = Path(folder) / f"{output_name}_{dataset_type}_tiles"
            pc_output_dir.mkdir(parents=True, exist_ok=True)
            
            downloaded_laz_files = []
            
            try:
                # FIRST: Do a targeted search for tile index files at the root level
                # These are typically named *TileIndex.zip, *tileindex.zip, *_tiles.zip, etc.
                # and are at the root of the dataset folder, not in subdirectories
                tile_index_candidates = []
                
                # List just the first level (no recursive deep dive yet)
                # Use Delimiter to get only direct children
                root_response = s3_client.list_objects_v2(
                    Bucket='pc-bulk',
                    Prefix=f"{short_name}/",
                    Delimiter='/',  # This limits to direct children only
                    MaxKeys=1000
                )
                
                # Check direct files at root level for tile index
                for obj in root_response.get('Contents', []):
                    key = obj['Key']
                    filename = Path(key).name.lower()
                    
                    # Check for tile index patterns (case-insensitive due to .lower())
                    if filename.endswith('.zip'):
                        if any(pattern in filename for pattern in [
                            'tileindex', 'tile_index', 'tiles', 'index',
                            'footprint', 'boundary', 'extent', 'grid'
                        ]):
                            tile_index_candidates.append(('zip', key))
                            logger.info(f"Found tile index candidate: {key}")
                    elif filename.endswith('.shp'):
                        if any(pattern in filename for pattern in [
                            'tileindex', 'tile_index', 'tiles', 'index',
                            'footprint', 'boundary', 'extent', 'grid'
                        ]):
                            tile_index_candidates.append(('shp', key))
                            logger.info(f"Found tile index candidate: {key}")
                
                # Also get subdirectories (CommonPrefixes)
                subdirs = [p['Prefix'] for p in root_response.get('CommonPrefixes', [])]
                if subdirs:
                    logger.info(f"Found subdirectories: {subdirs}")
                
                # Log root level files for debugging
                root_files = [Path(obj['Key']).name for obj in root_response.get('Contents', [])]
                if root_files:
                    logger.info(f"Root level files: {root_files}")
                
                aoi_geom = self.da.polygon.get("merged_polygon")
                
                # Try each tile index candidate
                tile_index_found = False
                for idx_type, tile_index_key in tile_index_candidates:
                    if tile_index_found:
                        break
                        
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir)
                            
                            if idx_type == 'zip':
                                # Download and extract zip file
                                zip_path = temp_path / 'tileindex.zip'
                                logger.info(f"Downloading tile index zip: {tile_index_key}")
                                s3_client.download_file('pc-bulk', tile_index_key, str(zip_path))
                                
                                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                    zip_ref.extractall(temp_path)
                                
                                # Find shapefile in extracted contents
                                shp_files = list(temp_path.rglob('*.shp'))
                            else:
                                # Direct shapefile - need to download all components (.shp, .shx, .dbf, .prj)
                                logger.info(f"Downloading tile index shapefile: {tile_index_key}")
                                base_key = tile_index_key[:-4]  # Remove .shp
                                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                                    try:
                                        s3_client.download_file('pc-bulk', base_key + ext, 
                                                               str(temp_path / (Path(base_key).name + ext)))
                                    except Exception:
                                        pass  # Some extensions are optional
                                shp_files = list(temp_path.glob('*.shp'))
                            
                            if shp_files:
                                logger.info(f"Reading shapefile: {shp_files[0].name}")
                                tiles_gdf = gpd.read_file(shp_files[0])
                                
                                # Log shapefile columns for debugging
                                logger.info(f"Tile index columns: {list(tiles_gdf.columns)}")
                                
                                tiles_gdf = tiles_gdf.to_crs('EPSG:4326')
                                
                                # Find intersecting tiles
                                intersecting_tiles = tiles_gdf[tiles_gdf.intersects(aoi_geom)]
                                logger.info(f"Found {len(intersecting_tiles)} tiles intersecting with AOI "
                                           f"(out of {len(tiles_gdf)} total)")
                                
                                if len(intersecting_tiles) > 0:
                                    tile_index_found = True
                                    
                                    # Log first tile's data for debugging
                                    first_tile = intersecting_tiles.iloc[0]
                                    logger.info(f"First tile data: {dict(first_tile.drop('geometry') if 'geometry' in first_tile.index else first_tile)}")
                                    
                                    # Download tiles
                                    def download_s3_tile(s3_key, local_path):
                                        try:
                                            if local_path.exists():
                                                logger.info(f"File already exists: {local_path.name}")
                                                return str(local_path)
                                            logger.info(f"Downloading: {s3_key}")
                                            s3_client.download_file('pc-bulk', s3_key, str(local_path))
                                            return str(local_path)
                                        except Exception as e:
                                            logger.error(f"Failed to download {s3_key}: {e}")
                                            return None
                                    
                                    with ThreadPoolExecutor(max_workers=5) as executor:
                                        futures = []
                                        
                                        for idx, tile in intersecting_tiles.iterrows():
                                            # Initialize variables for this tile
                                            url_val = None
                                            filename = None
                                            s3_key = None
                                            
                                            # Check URL column first - parse full S3 path from URL
                                            if 'URL' in tile.index and pd.notna(tile['URL']):
                                                url_val = str(tile['URL']).strip()
                                                logger.info(f"URL column value: {url_val}")
                                                
                                                # Parse the URL to extract S3 key
                                                # URL format: https://opentopography.s3.sdsc.edu/pc-bulk/IN_2011_2013/East_IN/Randolph/file.las
                                                # We need: IN_2011_2013/East_IN/Randolph/file.las
                                                if 'pc-bulk/' in url_val:
                                                    s3_key = url_val.split('pc-bulk/')[-1]
                                                    filename = Path(s3_key).name
                                                    logger.info(f"Extracted S3 key from URL: {s3_key}")
                                            elif 'url' in tile.index and pd.notna(tile['url']):
                                                url_val = str(tile['url']).strip()
                                                if 'pc-bulk/' in url_val:
                                                    s3_key = url_val.split('pc-bulk/')[-1]
                                                    filename = Path(s3_key).name
                                            
                                            # Check Filename column (for output naming)
                                            if not filename:
                                                if 'Filename' in tile.index and pd.notna(tile['Filename']):
                                                    filename = str(tile['Filename']).strip()
                                                elif 'filename' in tile.index and pd.notna(tile['filename']):
                                                    filename = str(tile['filename']).strip()
                                            
                                            if not filename:
                                                logger.warning(f"No filename found for tile at index {idx}")
                                                continue
                                            
                                            if not filename.lower().endswith(('.laz', '.las')):
                                                logger.debug(f"Skipping non-LAZ file: {filename}")
                                                continue
                                            
                                            # If we already have s3_key from URL parsing, use it directly
                                            # Otherwise, search for the file
                                            if not s3_key:
                                                # Try to find the file - first at root, then in subdirectories
                                                root_key = f"{short_name}/{filename}"
                                                
                                                # Check root level
                                                try:
                                                    s3_client.head_object(Bucket='pc-bulk', Key=root_key)
                                                    s3_key = root_key
                                                    logger.debug(f"Found at root: {s3_key}")
                                                except Exception:
                                                    # Try alternate extension at root
                                                    if filename.lower().endswith('.las'):
                                                        alt_filename = filename[:-4] + '.laz'
                                                    else:
                                                        alt_filename = filename[:-4] + '.las'
                                                    alt_root_key = f"{short_name}/{alt_filename}"
                                                    try:
                                                        s3_client.head_object(Bucket='pc-bulk', Key=alt_root_key)
                                                        s3_key = alt_root_key
                                                        filename = alt_filename
                                                        logger.debug(f"Found at root with alternate extension: {s3_key}")
                                                    except Exception:
                                                        # Search in subdirectories
                                                        logger.info(f"Not at root, searching subdirs: {subdirs}")
                                                        for subdir in subdirs:
                                                            # subdir is like "IN_2011_2013/East_IN/"
                                                            # We want to try "IN_2011_2013/East_IN/filename.las"
                                                            subdir_key = f"{subdir}{filename}"
                                                            logger.info(f"Trying: {subdir_key}")
                                                            try:
                                                                s3_client.head_object(Bucket='pc-bulk', Key=subdir_key)
                                                                s3_key = subdir_key
                                                                logger.info(f"Found in subdirectory: {s3_key}")
                                                                break
                                                            except Exception as e:
                                                                # Try alternate extension (.las <-> .laz)
                                                                if filename.lower().endswith('.las'):
                                                                    alt_filename = filename[:-4] + '.laz'
                                                                else:
                                                                    alt_filename = filename[:-4] + '.las'
                                                                alt_key = f"{subdir}{alt_filename}"
                                                                try:
                                                                    s3_client.head_object(Bucket='pc-bulk', Key=alt_key)
                                                                    s3_key = alt_key
                                                                    filename = alt_filename  # Update filename for output
                                                                    logger.info(f"Found with alternate extension: {s3_key}")
                                                                    break
                                                                except Exception:
                                                                    logger.info(f"Not found at {subdir_key} or {alt_key}")
                                                                    continue
                                            
                                            if s3_key:
                                                output_path = pc_output_dir / filename
                                                futures.append(executor.submit(download_s3_tile, s3_key, output_path))
                                            else:
                                                logger.warning(f"Could not find {filename} in bucket")
                                        
                                        for future in as_completed(futures):
                                            result = future.result()
                                            if result:
                                                downloaded_laz_files.append(result)
                                                for field in ['Filename', 'filename', 'file', 'name', 'file_name', 
                                                             'tilename', 'File', 'Name']:
                                                    if field in tile.index:
                                                        val = tile[field]
                                                        if pd.notna(val) and str(val).strip():
                                                            filename = Path(str(val)).name
                                                            break
                                            
                                else:
                                    logger.warning(f"No tiles intersect AOI in {Path(tile_index_key).name}")
                            else:
                                logger.warning(f"No shapefile found in {tile_index_key}")
                    except Exception as e:
                        logger.warning(f"Failed to process tile index {tile_index_key}: {e}")
                        continue
                
                # Fallback if no tile index worked
                if not tile_index_found:
                    if not tile_index_candidates:
                        logger.warning(f"No tile index files found for {short_name}")
                    
                    # Now we need to list LAZ files and try filename-based filtering
                    logger.info("Falling back to filename-based tile filtering...")
                    
                    # Get the dataset's horizontal CRS from catalog
                    if dataset_type == "compare":
                        dataset_epsg = self.ot.compare_horizontal_crs
                    else:
                        dataset_epsg = self.ot.reference_horizontal_crs
                    
                    logger.info(f"Dataset horizontal CRS: EPSG:{dataset_epsg}")
                    
                    # Transform AOI to dataset CRS for intersection testing
                    try:
                        dataset_epsg_int = int(dataset_epsg) if dataset_epsg else None
                    except (ValueError, TypeError):
                        dataset_epsg_int = None
                        logger.warning(f"Could not parse EPSG code: {dataset_epsg}")
                    
                    if not dataset_epsg_int:
                        logger.error("No valid EPSG code available, cannot spatially filter tiles")
                        raise RuntimeError(f"Cannot download OT dataset without valid CRS. EPSG={dataset_epsg}")
                    
                    # List all LAZ files (including subdirectories)
                    laz_files_in_bucket = []
                    logger.info("Listing all LAZ files in bucket (this may take a moment)...")
                    
                    paginator = s3_client.get_paginator('list_objects_v2')
                    page_iterator = paginator.paginate(
                        Bucket='pc-bulk',
                        Prefix=f"{short_name}/",
                        PaginationConfig={'MaxItems': 50000}  # Higher limit for large datasets
                    )
                    
                    for page in page_iterator:
                        for obj in page.get('Contents', []):
                            key = obj['Key']
                            if key.lower().endswith(('.laz', '.las')):
                                laz_files_in_bucket.append(key)
                    
                    logger.info(f"Found {len(laz_files_in_bucket)} LAZ files in bucket")
                    
                    if not laz_files_in_bucket:
                        raise RuntimeError(f"No LAZ files found in bucket {short_name}/")
                    
                    # Show sample filenames for debugging
                    sample_files = [Path(k).name for k in laz_files_in_bucket[:5]]
                    logger.info(f"Sample filenames: {sample_files}")
                    
                    # Transform AOI to dataset CRS
                    try:
                        transformer = Transformer.from_crs(
                            "EPSG:4326", f"EPSG:{dataset_epsg_int}", 
                            always_xy=True
                        )
                        # Transform AOI bounds to dataset CRS
                        aoi_bounds_4326 = aoi_geom.bounds  # (minx, miny, maxx, maxy)
                        aoi_min_transformed = transformer.transform(aoi_bounds_4326[0], aoi_bounds_4326[1])
                        aoi_max_transformed = transformer.transform(aoi_bounds_4326[2], aoi_bounds_4326[3])
                        
                        aoi_minx = min(aoi_min_transformed[0], aoi_max_transformed[0])
                        aoi_miny = min(aoi_min_transformed[1], aoi_max_transformed[1])
                        aoi_maxx = max(aoi_min_transformed[0], aoi_max_transformed[0])
                        aoi_maxy = max(aoi_min_transformed[1], aoi_max_transformed[1])
                        
                        logger.info(f"AOI in EPSG:{dataset_epsg_int}: "
                                   f"({aoi_minx:.0f}, {aoi_miny:.0f}) - ({aoi_maxx:.0f}, {aoi_maxy:.0f})")
                        
                        # Try multiple filename patterns
                        # Pattern 1: XXXXXX_YYYYYYYY.laz (standard UTM coords)
                        # Pattern 2: prefix_XXXXXXYYYYYY_suffix.laz (combined coords)
                        # Pattern 3: Any 6-8 digit numbers that could be coords
                        
                        coord_patterns = [
                            # Standard: 490000_4332000.laz
                            re.compile(r'(\d{6})_(\d{7})\.la[sz]$', re.IGNORECASE),
                            # With prefix: ot_in2012_04901935_12.las -> extract 0490, 1935
                            re.compile(r'_(\d{3})(\d{5})_\d+\.la[sz]$', re.IGNORECASE),
                            # State plane style: look for any large numbers
                            re.compile(r'(\d{5,7})_(\d{5,8})\.la[sz]$', re.IGNORECASE),
                        ]
                        
                        matching_files = []
                        tile_size = 1000  # Assume 1km tiles
                        
                        for key in laz_files_in_bucket:
                            filename = Path(key).name
                            
                            for pattern in coord_patterns:
                                match = pattern.search(filename)
                                if match:
                                    try:
                                        # Handle different coordinate scales
                                        coord1 = float(match.group(1))
                                        coord2 = float(match.group(2))
                                        
                                        # Heuristic: if coords are too small, they might need scaling
                                        # UTM eastings are typically 100,000-900,000
                                        # UTM northings are typically 0-10,000,000
                                        if coord1 < 1000:
                                            coord1 *= 1000  # Scale up
                                        if coord2 < 100000:
                                            coord2 *= 1000  # Scale up
                                        
                                        tile_easting = coord1
                                        tile_northing = coord2
                                        
                                        # Check if tile intersects AOI
                                        tile_minx = tile_easting
                                        tile_miny = tile_northing
                                        tile_maxx = tile_easting + tile_size
                                        tile_maxy = tile_northing + tile_size
                                        
                                        if (tile_maxx >= aoi_minx and tile_minx <= aoi_maxx and
                                            tile_maxy >= aoi_miny and tile_miny <= aoi_maxy):
                                            matching_files.append(key)
                                            break  # Don't try other patterns
                                    except (ValueError, IndexError):
                                        continue
                        
                        if matching_files:
                            logger.info(f"Found {len(matching_files)} tiles intersecting AOI")
                            
                            # Download matching tiles
                            for key in matching_files:
                                filename = Path(key).name
                                output_path = pc_output_dir / filename
                                try:
                                    if not output_path.exists():
                                        logger.info(f"Downloading: {key}")
                                        s3_client.download_file('pc-bulk', key, str(output_path))
                                    downloaded_laz_files.append(str(output_path))
                                except Exception as e:
                                    logger.error(f"Failed to download {key}: {e}")
                        else:
                            logger.warning(f"No tiles found intersecting AOI after coordinate parsing")
                            logger.warning(f"Checked {len(laz_files_in_bucket)} files")
                            logger.warning(f"This dataset may require manual tile index creation or different coordinate parsing")
                            
                    except Exception as e:
                        logger.error(f"Failed to transform AOI to dataset CRS: {e}")
                        raise
                
            except Exception as e:
                logger.error(f"Error downloading OpenTopography dataset {short_name}: {e}")
                raise
            
            if not downloaded_laz_files:
                raise RuntimeError(f"No LAZ files downloaded for OpenTopography dataset {short_name}")
            
            logger.info(f"Downloaded {len(downloaded_laz_files)} LAZ tiles")
            
            # Now create DEMs from the downloaded tiles
            # If multiple tiles, we need to merge them first or process together
            if len(downloaded_laz_files) == 1:
                input_laz = downloaded_laz_files[0]
            else:
                # Merge tiles into a single LAZ file using PDAL
                merged_laz = folder + output_name + '_' + dataset_type + '_merged.laz'
                
                merge_pipeline = {
                    "pipeline": [
                        {"type": "readers.las", "filename": f} for f in downloaded_laz_files
                    ] + [
                        {"type": "filters.merge"},
                        {"type": "writers.las", "filename": merged_laz}
                    ]
                }
                
                logger.info(f"Merging {len(downloaded_laz_files)} tiles into {merged_laz}")
                merge_pipe = pdal.Pipeline(json.dumps(merge_pipeline))
                merge_pipe.execute()
                input_laz = merged_laz
            
            # Create DTM
            ot_dtm_pipeline = self.make_DEM_pipeline_from_file(
                input_laz, bounds_polygon_epsg_initial_crs, dem_resolution,
                filterNoise=filterNoise, reclassify=reclassify, savePointCloud=savePointCloud, 
                outCRS=final_out_crs_wkt,
                pc_outName=folder+output_name, pc_outType='laz', demType='dtm', gridMethod='idw', 
                dem_outName=folder+output_name+'_'+dataset_type+'_DTM', dem_outExt='tif', driver="GTiff"
            )
            ot_dtm_pipeline = pdal.Pipeline(json.dumps(ot_dtm_pipeline))
            ot_dtm_pipeline.execute_streaming(chunk_size=1000000)
            dtm_path = folder+output_name+'_'+dataset_type+'_DTM.tif'
            self.cleanup_raster_nodata(dtm_path, nodata=nodata)
            logger.info(f"Created DTM: {dtm_path}")

            # Create DSM
            ot_dsm_pipeline = self.make_DEM_pipeline_from_file(
                input_laz, bounds_polygon_epsg_initial_crs, dem_resolution,
                filterNoise=filterNoise, reclassify=reclassify, savePointCloud=savePointCloud, 
                outCRS=final_out_crs_wkt,
                pc_outName=folder+output_name, pc_outType='laz', demType='dsm', gridMethod='max', 
                dem_outName=folder+output_name+'_'+dataset_type+'_DSM', dem_outExt='tif', driver="GTiff"
            )
            ot_dsm_pipeline = pdal.Pipeline(json.dumps(ot_dsm_pipeline))
            ot_dsm_pipeline.execute_streaming(chunk_size=1000000)
            dsm_path = folder+output_name+'_'+dataset_type+'_DSM.tif'
            self.cleanup_raster_nodata(dsm_path, nodata=nodata)
            logger.info(f"Created DSM: {dsm_path}")
        
        elif data_source_ == "usgs":
            usgs_dtm_pipeline = self.make_DEM_pipeline_aws(bounds_polygon_epsg_initial_crs, [dataset_id], pc_resolution, dem_resolution, data_source = "usgs",
                    filterNoise = False, reclassify = False, savePointCloud = False, outCRS = final_out_crs_wkt,
                    pc_outName = folder+output_name+'_'+dataset_type, pc_outType = 'laz', demType = 'dtm', gridMethod = 'idw', 
                    dem_outName = folder+output_name+'_'+dataset_type+'_DTM', dem_outExt = 'tif', driver = "GTiff")

            usgs_dtm_pipeline = pdal.Pipeline(json.dumps(usgs_dtm_pipeline))
            usgs_dtm_pipeline.execute_streaming(chunk_size=1000000)
            dtm_path = folder+output_name+'_'+dataset_type+'_DTM.tif'
            self.cleanup_raster_nodata(dtm_path, nodata=nodata)
            
            usgs_dsm_pipeline = self.make_DEM_pipeline_aws(bounds_polygon_epsg_initial_crs, [dataset_id], pc_resolution, dem_resolution, data_source = "usgs",
                            filterNoise = False, reclassify = False, savePointCloud = False, outCRS = final_out_crs_wkt,
                            pc_outName = folder+output_name+'_'+dataset_type, pc_outType = 'laz', demType = 'dsm', gridMethod = 'max', 
                            dem_outName = folder+output_name+'_'+dataset_type+'_DSM', dem_outExt = 'tif', driver = "GTiff")

            usgs_dsm_pipeline = pdal.Pipeline(json.dumps(usgs_dsm_pipeline))
            usgs_dsm_pipeline.execute_streaming(chunk_size=1000000)
            dsm_path = folder+output_name+'_'+dataset_type+'_DSM.tif'
            self.cleanup_raster_nodata(dsm_path, nodata=nodata)

        elif data_source_ == "noaa":
            noaa_dtm_pipeline = self.make_DEM_pipeline_aws(bounds_polygon_epsg_initial_crs, [dataset_id], pc_resolution, dem_resolution, data_source = "noaa",
                    filterNoise = False, reclassify = False, savePointCloud = False, outCRS = final_out_crs_wkt,
                    pc_outName = folder+output_name+'_'+dataset_type, pc_outType = 'laz', demType = 'dtm', gridMethod = 'idw', 
                    dem_outName = folder+output_name+'_'+dataset_type+'_DTM', dem_outExt = 'tif', driver = "GTiff")

            noaa_dtm_pipeline = pdal.Pipeline(json.dumps(noaa_dtm_pipeline))
            noaa_dtm_pipeline.execute_streaming(chunk_size=1000000)
            dtm_path = folder+output_name+'_'+dataset_type+'_DTM.tif'
            self.cleanup_raster_nodata(dtm_path, nodata=nodata)
            
            noaa_dsm_pipeline = self.make_DEM_pipeline_aws(bounds_polygon_epsg_initial_crs, [dataset_id], pc_resolution, dem_resolution, data_source = "noaa",
                            filterNoise = False, reclassify = False, savePointCloud = False, outCRS = final_out_crs_wkt,
                            pc_outName = folder+output_name+'_'+dataset_type, pc_outType = 'laz', demType = 'dsm', gridMethod = 'max', 
                            dem_outName = folder+output_name+'_'+dataset_type+'_DSM', dem_outExt = 'tif', driver = "GTiff")

            noaa_dsm_pipeline = pdal.Pipeline(json.dumps(noaa_dsm_pipeline))
            noaa_dsm_pipeline.execute_streaming(chunk_size=1000000)
            dsm_path = folder+output_name+'_'+dataset_type+'_DSM.tif'
            self.cleanup_raster_nodata(dsm_path, nodata=nodata)

        else:
            raise ValueError(f"Unknown data source: {data_source_}")
        
        # Store metadata about the created DEMs for later integration
        # IMPORTANT: Merge with existing metadata to preserve user overrides 
        # (e.g., units_override from set_compare_units())
        if dataset_type == "compare":
            # Preserve any existing overrides
            existing = getattr(self, '_compare_metadata', {})
            preserved_keys = {
                'units_override': existing.get('units_override'),
                'unit_info_override': existing.get('unit_info_override'),
                'geoid_model_override': existing.get('geoid_model_override'),
            }
            self._compare_metadata = {
                'epoch': epoch,
                'epoch_decimal': _date_to_decimal_year(epoch),
                'horizontal_crs': dataset_crs_,
                'vertical_crs_string': self.ot.compare_vertical_crs,
                'data_source': data_source_,
                'dataset_id': dataset_id,
                'dem_resolution': dem_resolution,
                'gridding_method_dtm': 'idw',
                'gridding_method_dsm': 'max',
            }
            # Restore preserved overrides
            for key, value in preserved_keys.items():
                if value is not None:
                    self._compare_metadata[key] = value
        else:
            # Preserve any existing overrides
            existing = getattr(self, '_reference_metadata', {})
            preserved_keys = {
                'units_override': existing.get('units_override'),
                'unit_info_override': existing.get('unit_info_override'),
                'geoid_model_override': existing.get('geoid_model_override'),
            }
            self._reference_metadata = {
                'epoch': epoch,
                'epoch_decimal': _date_to_decimal_year(epoch),
                'horizontal_crs': dataset_crs_,
                'vertical_crs_string': self.ot.reference_vertical_crs,
                'data_source': data_source_,
                'dataset_id': dataset_id,
                'dem_resolution': dem_resolution,
                'gridding_method_dtm': 'idw',
                'gridding_method_dsm': 'max',
            }
            # Restore preserved overrides
            for key, value in preserved_keys.items():
                if value is not None:
                    self._reference_metadata[key] = value

    def set_compare_geoid(self, geoid_model: str):
        """
        Override the geoid model for the compare dataset.
        
        Use this when the catalog doesn't have the correct geoid info
        or when you know the specific geoid used.
        
        Parameters
        ----------
        geoid_model : str
            Geoid model name (e.g., 'geoid09', 'geoid12b', 'geoid18')
            
        Example
        -------
        >>> gdems.set_compare_geoid('geoid09')
        >>> pair = gdems.get_dtm_pair()  # Will use geoid09 for compare
        """
        if not hasattr(self, '_compare_metadata'):
            self._compare_metadata = {}
        
        # Update vertical_crs_string to include geoid
        current = self._compare_metadata.get('vertical_crs_string', 'NAVD88')
        if geoid_model.lower() not in str(current).lower():
            self._compare_metadata['vertical_crs_string'] = f"NAVD88 ({geoid_model})"
        
        # Also store explicit override
        self._compare_metadata['geoid_model_override'] = geoid_model
        logger.info(f"Set compare geoid model to: {geoid_model}")

    def set_reference_geoid(self, geoid_model: str):
        """
        Override the geoid model for the reference dataset.
        
        Parameters
        ----------
        geoid_model : str
            Geoid model name (e.g., 'geoid09', 'geoid12b', 'geoid18')
        """
        if not hasattr(self, '_reference_metadata'):
            self._reference_metadata = {}
        
        current = self._reference_metadata.get('vertical_crs_string', 'NAVD88')
        if geoid_model.lower() not in str(current).lower():
            self._reference_metadata['vertical_crs_string'] = f"NAVD88 ({geoid_model})"
        
        self._reference_metadata['geoid_model_override'] = geoid_model
        logger.info(f"Set reference geoid model to: {geoid_model}")

    def set_compare_units(self, units: str):
        """
        Override the vertical units for the compare dataset.
        
        Use this when the catalog doesn't have the correct unit info
        or when you know the specific units used.
        
        Parameters
        ----------
        units : str
            Unit name: 'meter', 'foot', 'us_survey_foot', or common aliases
            
        Example
        -------
        >>> gdems.set_compare_units('meter')
        >>> gdems.set_compare_units('ftUS')  # US survey foot
        >>> pair = gdems.get_dtm_pair()  # Will use specified units for compare
        """
        try:
            normalized_unit = _normalize_unit_name(units)
        except ValueError as e:
            raise ValueError(f"Invalid units '{units}': {e}")
        
        if not hasattr(self, '_compare_metadata'):
            self._compare_metadata = {}
        
        self._compare_metadata['units_override'] = normalized_unit
        
        # Also store UnitInfo if available
        unit_info = _get_unit_info(normalized_unit)
        if unit_info is not None:
            self._compare_metadata['unit_info_override'] = unit_info
            logger.info(f"Set compare vertical units to: {unit_info.display_name}")
        else:
            logger.info(f"Set compare vertical units to: {normalized_unit}")

    def set_reference_units(self, units: str):
        """
        Override the vertical units for the reference dataset.
        
        Parameters
        ----------
        units : str
            Unit name: 'meter', 'foot', 'us_survey_foot', or common aliases
        """
        try:
            normalized_unit = _normalize_unit_name(units)
        except ValueError as e:
            raise ValueError(f"Invalid units '{units}': {e}")
        
        if not hasattr(self, '_reference_metadata'):
            self._reference_metadata = {}
        
        self._reference_metadata['units_override'] = normalized_unit
        
        # Also store UnitInfo if available
        unit_info = _get_unit_info(normalized_unit)
        if unit_info is not None:
            self._reference_metadata['unit_info_override'] = unit_info
            logger.info(f"Set reference vertical units to: {unit_info.display_name}")
        else:
            logger.info(f"Set reference vertical units to: {normalized_unit}")

    # =========================================================================
    # Integration with Domain Classes (Raster, PointCloud, RasterPair, etc.)
    # =========================================================================

    def _load_dem_as_raster(
        self,
        dem_path: str,
        rtype: str,
        epoch: Optional[Union[date, datetime, float]] = None,
        vertical_crs_string: Optional[str] = None,
        horizontal_crs: Optional[int] = None,
        gridding_method: Optional[str] = None,
        data_source: Optional[str] = None,
        dataset_id: Optional[str] = None,
        geoid_model_override: Optional[str] = None,
        units_override: Optional[str] = None,
    ) -> "Raster":
        """
        Load a DEM file as a Raster object with full metadata.
        
        This method creates a Raster object from a file path and populates
        it with metadata from the OpenTopography catalog and PDAL pipeline.
        
        Parameters
        ----------
        dem_path : str
            Path to the DEM GeoTIFF file
        rtype : str
            Raster type ('dtm' or 'dsm')
        epoch : date, datetime, or float, optional
            Acquisition epoch (will be converted to decimal year)
        vertical_crs_string : str, optional
            Vertical CRS string from catalog (e.g., "NAVD88 (Geoid 12B)")
        horizontal_crs : int, optional
            Horizontal CRS EPSG code
        gridding_method : str, optional
            Interpolation method used during DEM creation ('idw', 'max', etc.)
        data_source : str, optional
            Data source ('ot', 'usgs', 'noaa')
        dataset_id : str, optional
            Dataset identifier
        geoid_model_override : str, optional
            Explicit geoid model to use, overriding any parsed value
        units_override : str, optional
            Explicit vertical units ('meter', 'foot', 'us_survey_foot')
            
        Returns
        -------
        Raster
            Loaded raster with CRS history and metadata
            
        Raises
        ------
        ImportError
            If Raster class is not available
        FileNotFoundError
            If the DEM file doesn't exist
        """
        if not _DOMAIN_CLASSES_AVAILABLE:
            raise ImportError(
                "Domain classes (Raster, etc.) not available. "
                "Please ensure raster.py is in the Python path."
            )
        
        if not os.path.exists(dem_path):
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
        # Convert epoch to decimal year
        if isinstance(epoch, (date, datetime)):
            epoch_decimal = _date_to_decimal_year(epoch)
        elif isinstance(epoch, (int, float)):
            epoch_decimal = float(epoch)
        else:
            epoch_decimal = None
        
        # Parse vertical CRS info
        vert_info = _parse_vertical_crs_string(vertical_crs_string)
        
        # Apply geoid override if provided
        effective_geoid = geoid_model_override or vert_info.get('geoid_model')
        
        # Apply units override if provided, get UnitInfo if available
        effective_units = units_override or vert_info.get('units') or 'meter'  # Default to meters
        effective_unit_info = None
        
        if _UNIT_UTILS_AVAILABLE:
            if units_override:
                effective_unit_info = lookup_unit(units_override)
            elif vert_info.get('unit_info') is not None:
                effective_unit_info = vert_info['unit_info']
            else:
                effective_unit_info = METER  # Default to meters
        
        # Build metadata dictionary
        metadata = {
            'epoch': epoch_decimal,
            'geoid_model': effective_geoid,
            'vertical_units': effective_units,
            'data_source': data_source,
            'dataset_id': dataset_id,
            'gridding_method': gridding_method,
            'vertical_datum_string': vertical_crs_string,
        }
        
        # Load raster with Raster.from_file
        raster = Raster.from_file(dem_path, rtype=rtype, metadata=metadata)
        
        # Set additional attributes from parsed vertical info
        if vert_info.get('is_orthometric') is not None:
            raster.is_orthometric = vert_info['is_orthometric']
        
        # Set geoid model (using override if provided)
        if effective_geoid:
            raster.original_geoid_model = effective_geoid
            raster.current_geoid_model = effective_geoid
        
        # Set vertical units - use UnitInfo if available
        if effective_unit_info is not None and hasattr(raster, 'vertical_unit'):
            # New enhanced approach with UnitInfo
            raster.original_vertical_unit = effective_unit_info
            raster.current_vertical_unit = effective_unit_info
            raster.original_vertical_units = effective_unit_info.display_name
            raster.current_vertical_units = effective_unit_info.display_name
        else:
            # Fallback to string-based units
            raster.vertical_units = effective_units
            raster.original_vertical_units = effective_units
            raster.current_vertical_units = effective_units
        
        # Record the interpolation used during DEM creation
        if raster.crs_history is not None and gridding_method:
            try:
                raster.crs_history.record_interpolation_entry(
                    operation_type='creation',
                    method=gridding_method,
                    description=f"Point cloud gridding to {rtype.upper()} using {gridding_method}",
                    source_file=None,  # Point cloud source unknown
                    target_file=dem_path,
                    target_resolution=getattr(raster, 'resolution', None),
                    parameters={
                        'data_source': data_source,
                        'dataset_id': dataset_id,
                        'raster_type': rtype,
                    },
                )
            except Exception:
                pass  # Graceful degradation if CRSHistory doesn't support this
        
        return raster

    def get_compare_dtm(self) -> "Raster":
        """
        Get the compare DTM as a Raster object with full metadata.
        
        Returns
        -------
        Raster
            The compare DTM with CRS history and metadata from the catalog
            
        Raises
        ------
        AttributeError
            If compare DTM has not been created yet
        """
        if not hasattr(self, 'compare_dtm_path') or self.compare_dtm_path is None:
            raise AttributeError(
                "Compare DTM not yet created. Run dem_download_workflow() first."
            )
        
        meta = getattr(self, '_compare_metadata', {})
        return self._load_dem_as_raster(
            dem_path=self.compare_dtm_path,
            rtype='dtm',
            epoch=meta.get('epoch'),
            vertical_crs_string=meta.get('vertical_crs_string'),
            horizontal_crs=meta.get('horizontal_crs'),
            gridding_method=meta.get('gridding_method_dtm', 'idw'),
            data_source=meta.get('data_source'),
            dataset_id=meta.get('dataset_id'),
            geoid_model_override=meta.get('geoid_model_override'),
            units_override=meta.get('units_override'),
        )

    def get_compare_dsm(self) -> "Raster":
        """
        Get the compare DSM as a Raster object with full metadata.
        
        Returns
        -------
        Raster
            The compare DSM with CRS history and metadata
        """
        if not hasattr(self, 'compare_dsm_path') or self.compare_dsm_path is None:
            raise AttributeError(
                "Compare DSM not yet created. Run dem_download_workflow() first."
            )
        
        meta = getattr(self, '_compare_metadata', {})
        return self._load_dem_as_raster(
            dem_path=self.compare_dsm_path,
            rtype='dsm',
            epoch=meta.get('epoch'),
            vertical_crs_string=meta.get('vertical_crs_string'),
            horizontal_crs=meta.get('horizontal_crs'),
            gridding_method=meta.get('gridding_method_dsm', 'max'),
            data_source=meta.get('data_source'),
            dataset_id=meta.get('dataset_id'),
            geoid_model_override=meta.get('geoid_model_override'),
            units_override=meta.get('units_override'),
        )

    def get_reference_dtm(self) -> "Raster":
        """
        Get the reference DTM as a Raster object with full metadata.
        
        Returns
        -------
        Raster
            The reference DTM with CRS history and metadata
        """
        if not hasattr(self, 'reference_dtm_path') or self.reference_dtm_path is None:
            raise AttributeError(
                "Reference DTM not yet created. Run dem_download_workflow() first."
            )
        
        meta = getattr(self, '_reference_metadata', {})
        return self._load_dem_as_raster(
            dem_path=self.reference_dtm_path,
            rtype='dtm',
            epoch=meta.get('epoch'),
            vertical_crs_string=meta.get('vertical_crs_string'),
            horizontal_crs=meta.get('horizontal_crs'),
            gridding_method=meta.get('gridding_method_dtm', 'idw'),
            data_source=meta.get('data_source'),
            dataset_id=meta.get('dataset_id'),
            geoid_model_override=meta.get('geoid_model_override'),
            units_override=meta.get('units_override'),
        )

    def get_reference_dsm(self) -> "Raster":
        """
        Get the reference DSM as a Raster object with full metadata.
        
        Returns
        -------
        Raster
            The reference DSM with CRS history and metadata
        """
        if not hasattr(self, 'reference_dsm_path') or self.reference_dsm_path is None:
            raise AttributeError(
                "Reference DSM not yet created. Run dem_download_workflow() first."
            )
        
        meta = getattr(self, '_reference_metadata', {})
        return self._load_dem_as_raster(
            dem_path=self.reference_dsm_path,
            rtype='dsm',
            epoch=meta.get('epoch'),
            vertical_crs_string=meta.get('vertical_crs_string'),
            horizontal_crs=meta.get('horizontal_crs'),
            gridding_method=meta.get('gridding_method_dsm', 'max'),
            data_source=meta.get('data_source'),
            dataset_id=meta.get('dataset_id'),
            geoid_model_override=meta.get('geoid_model_override'),
            units_override=meta.get('units_override'),
        )

    def get_dtm_pair(self) -> "RasterPair":
        """
        Get a RasterPair of the compare and reference DTMs.
        
        The returned RasterPair is ready for differencing and analysis.
        The compare DTM is raster1, reference DTM is raster2, following
        the convention that difference = raster2 - raster1.
        
        Returns
        -------
        RasterPair
            Pair of DTMs ready for compute_difference()
            
        Example
        -------
        >>> gdems = GetDEMs(da, ot)
        >>> gdems.dem_download_workflow(..., dataset_type="compare", ...)
        >>> gdems.dem_download_workflow(..., dataset_type="reference", ...)
        >>> pair = gdems.get_dtm_pair()
        >>> result = pair.compute_difference(verbose=True)
        """
        if not _DOMAIN_CLASSES_AVAILABLE:
            raise ImportError(
                "Domain classes (RasterPair, etc.) not available. "
                "Please ensure rasterpair.py is in the Python path."
            )
        
        compare_dtm = self.get_compare_dtm()
        reference_dtm = self.get_reference_dtm()
        
        return RasterPair(raster1=compare_dtm, raster2=reference_dtm)

    def get_dtm_pair_with_rasters(
        self, 
        compare_dtm: "Raster" = None, 
        reference_dtm: "Raster" = None
    ) -> "RasterPair":
        """
        Get a RasterPair using provided or auto-loaded DTM rasters.
        
        This method allows you to pass pre-modified Raster objects
        (e.g., after calling add_metadata() to set geoid models).
        
        Parameters
        ----------
        compare_dtm : Raster, optional
            The compare DTM. If None, loads from get_compare_dtm()
        reference_dtm : Raster, optional
            The reference DTM. If None, loads from get_reference_dtm()
            
        Returns
        -------
        RasterPair
            Pair of DTMs ready for compute_difference()
            
        Example
        -------
        >>> compare = gdems.get_compare_dtm()
        >>> compare.add_metadata(geoid_model='geoid09')
        >>> pair = gdems.get_dtm_pair_with_rasters(compare_dtm=compare)
        """
        if not _DOMAIN_CLASSES_AVAILABLE:
            raise ImportError("Domain classes not available.")
        
        if compare_dtm is None:
            compare_dtm = self.get_compare_dtm()
        if reference_dtm is None:
            reference_dtm = self.get_reference_dtm()
        
        return RasterPair(raster1=compare_dtm, raster2=reference_dtm)

    def get_dsm_pair(self) -> "RasterPair":
        """
        Get a RasterPair of the compare and reference DSMs.
        
        Returns
        -------
        RasterPair
            Pair of DSMs ready for compute_difference()
        """
        if not _DOMAIN_CLASSES_AVAILABLE:
            raise ImportError(
                "Domain classes (RasterPair, etc.) not available. "
                "Please ensure rasterpair.py is in the Python path."
            )
        
        compare_dsm = self.get_compare_dsm()
        reference_dsm = self.get_reference_dsm()
        
        return RasterPair(raster1=compare_dsm, raster2=reference_dsm)

    def load_pointcloud(
        self,
        pc_path: str,
        epoch: Optional[Union[date, datetime, float]] = None,
        vertical_crs_string: Optional[str] = None,
        units_override: Optional[str] = None,
    ) -> "PointCloud":
        """
        Load a point cloud file as a PointCloud object with metadata.
        
        Parameters
        ----------
        pc_path : str
            Path to the LAZ/LAS file
        epoch : date, datetime, or float, optional
            Acquisition epoch
        vertical_crs_string : str, optional
            Vertical CRS string from catalog
        units_override : str, optional
            Override vertical units (e.g., 'meter', 'foot', 'us_survey_foot')
            
        Returns
        -------
        PointCloud
            Loaded point cloud with metadata
        """
        if not _DOMAIN_CLASSES_AVAILABLE:
            raise ImportError(
                "Domain classes (PointCloud, etc.) not available. "
                "Please ensure pointcloud.py is in the Python path."
            )
        
        if not os.path.exists(pc_path):
            raise FileNotFoundError(f"Point cloud file not found: {pc_path}")
        
        # Create and load point cloud
        pc = PointCloud(pc_path)
        pc.from_file()
        
        # Set epoch if provided
        if epoch is not None:
            if isinstance(epoch, (date, datetime)):
                pc.epoch = _date_to_decimal_year(epoch)
            elif isinstance(epoch, (int, float)):
                pc.epoch = float(epoch)
        
        # Parse and set vertical CRS info
        if vertical_crs_string:
            vert_info = _parse_vertical_crs_string(vertical_crs_string)
            if vert_info.get('is_orthometric') is not None:
                pc.is_orthometric = vert_info['is_orthometric']
            if vert_info.get('geoid_model'):
                pc.geoid_model = vert_info['geoid_model']
            
            # Set unit info from catalog string if not overridden
            if not units_override and _UNIT_UTILS_AVAILABLE:
                unit_info = vert_info.get('unit_info')
                if unit_info is not None and hasattr(pc, 'vertical_unit'):
                    pc.vertical_unit = unit_info
                    pc.vertical_units = unit_info.display_name
        
        # Apply units override if provided
        if units_override and _UNIT_UTILS_AVAILABLE:
            unit_info = lookup_unit(units_override)
            if unit_info is not None and hasattr(pc, 'vertical_unit'):
                pc.vertical_unit = unit_info
                pc.vertical_units = unit_info.display_name
        
        return pc

    def get_all_rasters(self) -> Dict[str, "Raster"]:
        """
        Get all created DEMs as a dictionary of Raster objects.
        
        Returns
        -------
        dict
            Dictionary with keys like 'compare_dtm', 'compare_dsm',
            'reference_dtm', 'reference_dsm', mapping to Raster objects.
            Only includes DEMs that have been created.
        """
        result = {}
        
        if hasattr(self, 'compare_dtm_path') and self.compare_dtm_path:
            try:
                result['compare_dtm'] = self.get_compare_dtm()
            except Exception:
                pass
        
        if hasattr(self, 'compare_dsm_path') and self.compare_dsm_path:
            try:
                result['compare_dsm'] = self.get_compare_dsm()
            except Exception:
                pass
        
        if hasattr(self, 'reference_dtm_path') and self.reference_dtm_path:
            try:
                result['reference_dtm'] = self.get_reference_dtm()
            except Exception:
                pass
        
        if hasattr(self, 'reference_dsm_path') and self.reference_dsm_path:
            try:
                result['reference_dsm'] = self.get_reference_dsm()
            except Exception:
                pass
        
        return result

    def print_metadata_summary(self) -> None:
        """
        Print a summary of metadata for all created DEMs.
        
        Displays epoch, CRS, geoid model, and other key metadata
        in a formatted table.
        """
        print("\n" + "=" * 70)
        print("DEM METADATA SUMMARY")
        print("=" * 70)
        
        for name in ['compare', 'reference']:
            meta = getattr(self, f'_{name}_metadata', None)
            dtm_path = getattr(self, f'{name}_dtm_path', None)
            dsm_path = getattr(self, f'{name}_dsm_path', None)
            
            if meta:
                print(f"\n{name.upper()} Dataset:")
                print(f"  Epoch (decimal year): {meta.get('epoch_decimal')}")
                print(f"  Epoch (date):         {meta.get('epoch')}")
                print(f"  Horizontal CRS:       EPSG:{meta.get('horizontal_crs')}")
                print(f"  Vertical CRS:         {meta.get('vertical_crs_string')}")
                print(f"  Data Source:          {meta.get('data_source')}")
                print(f"  Dataset ID:           {meta.get('dataset_id')}")
                print(f"  DEM Resolution:       {meta.get('dem_resolution')} m")
                print(f"  DTM Gridding Method:  {meta.get('gridding_method_dtm')}")
                print(f"  DSM Gridding Method:  {meta.get('gridding_method_dsm')}")
                if dtm_path:
                    print(f"  DTM Path:             {dtm_path}")
                if dsm_path:
                    print(f"  DSM Path:             {dsm_path}")
        
        print("\n" + "=" * 70)