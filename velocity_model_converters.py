# velocity_model_converters.py
"""
Format converters for velocity/deformation models.

Handles conversion of various source formats to PROJ-compatible GeoTIFF:
  - UCERF3 OpenSHA → GeoTIFF
  - USGS NSHM NetCDF → GeoTIFF  
  - GEM GSRM ASCII → GeoTIFF
  - EarthScope point data → GeoTIFF
  - Generic NetCDF → GeoTIFF
  - Generic ASCII/CSV → GeoTIFF
"""

from __future__ import annotations

import json
import os
import re
import struct
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS as RioCRS
except ImportError:
    rasterio = None

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    from scipy.interpolate import griddata
except ImportError:
    griddata = None


# ============================================================================
# Base converter class
# ============================================================================

class VelocityConverter:
    """Base class for velocity model format converters."""
    
    @staticmethod
    def convert(
        input_path: Path,
        output_path: Path,
        **kwargs,
    ) -> Path:
        """
        Convert source format to PROJ-compatible GeoTIFF.
        
        Args:
            input_path: Source file/directory
            output_path: Output GeoTIFF path
            **kwargs: Format-specific options
            
        Returns:
            Path to created GeoTIFF
        """
        raise NotImplementedError


# ============================================================================
# UCERF3 OpenSHA converter
# ============================================================================

class UCERF3Converter(VelocityConverter):
    """
    Convert UCERF3 deformation model to GeoTIFF.
    
    UCERF3 provides:
      - Fault slip rates (in OpenSHA FaultSystemSolution format)
      - Off-fault strain rates on 0.1° grid
      
    We extract the gridded strain rates and convert to velocities.
    """
    
    @staticmethod
    def convert(
        input_path: Path,
        output_path: Path,
        model_variant: str = "MeanUCERF3",
        **kwargs,
    ) -> Path:
        """
        Convert UCERF3 to GeoTIFF velocity grid.
        
        Args:
            input_path: Path to UCERF3 zip or extracted directory
            output_path: Output GeoTIFF path
            model_variant: Which UCERF3 variant ('MeanUCERF3', 'GEOLOGIC', etc.)
        """
        if rasterio is None:
            raise ImportError("rasterio required for UCERF3 conversion")
        
        input_path = Path(input_path)
        
        # Extract if zip
        if input_path.suffix == '.zip':
            extract_dir = input_path.parent / input_path.stem
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(input_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            work_dir = extract_dir
        else:
            work_dir = input_path
        
        # Find strain rate files
        # UCERF3 format: off-fault strain on 0.1 degree grid
        # Typically: *_strain_rates.txt or similar
        strain_files = list(work_dir.glob(f"*{model_variant}*strain*.txt"))
        if not strain_files:
            strain_files = list(work_dir.glob("*strain*.txt"))
        
        if not strain_files:
            raise FileNotFoundError(
                f"No strain rate files found in {work_dir}. "
                f"Expected *strain*.txt"
            )
        
        strain_file = strain_files[0]
        
        # Parse UCERF3 strain rate file
        # Format: lon, lat, exx, eyy, exy (strain rate tensor components)
        data = np.loadtxt(strain_file, skiprows=1)
        
        if data.shape[1] < 5:
            raise ValueError(
                f"Expected at least 5 columns (lon, lat, exx, eyy, exy), "
                f"got {data.shape[1]}"
            )
        
        lons = data[:, 0]
        lats = data[:, 1]
        exx = data[:, 2]  # E-W strain rate (1/yr)
        eyy = data[:, 3]  # N-S strain rate (1/yr)
        exy = data[:, 4]  # Shear strain rate (1/yr)
        
        # Convert strain rates to velocities
        # This is simplified - proper conversion requires:
        # 1. Integration of strain field
        # 2. Boundary conditions (plate motion)
        # 3. Rotation consideration
        
        # For now, approximate using strain × characteristic length
        # This gives relative velocities, not absolute
        R_earth = 6371000  # meters
        deg_to_rad = np.pi / 180
        lat_rad = lats * deg_to_rad
        
        # Approximate velocity from strain (simplified)
        # v = strain_rate × distance
        # Use local earth radius for characteristic scale
        dx = 0.1 * deg_to_rad * R_earth * np.cos(lat_rad)  # E-W distance
        dy = 0.1 * deg_to_rad * R_earth  # N-S distance
        
        # Velocity components (mm/yr)
        ve = exx * dx * 1000  # East velocity
        vn = eyy * dy * 1000  # North velocity
        
        # For vertical: assume negligible from horizontal strain
        # (would need full 3D model for proper vertical)
        vu = np.zeros_like(ve)
        
        # Create regular grid
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        
        # UCERF3 uses 0.1 degree spacing
        resolution = 0.1
        nx = int((lon_max - lon_min) / resolution) + 1
        ny = int((lat_max - lat_min) / resolution) + 1
        
        lon_grid = np.linspace(lon_min, lon_max, nx)
        lat_grid = np.linspace(lat_min, lat_max, ny)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Interpolate to regular grid
        if griddata is None:
            raise ImportError("scipy required for gridding")
        
        print(f"Gridding UCERF3 data: {len(lons)} points → {nx}x{ny} grid")
        
        ve_grid = griddata((lons, lats), ve, (lon_mesh, lat_mesh), method='linear')
        vn_grid = griddata((lons, lats), vn, (lon_mesh, lat_mesh), method='linear')
        vu_grid = np.zeros_like(ve_grid)
        
        # Write to GeoTIFF
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=ny,
            width=nx,
            count=3,
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform,
            compress='deflate',
        ) as dst:
            dst.write(ve_grid.astype(np.float32), 1)
            dst.write(vn_grid.astype(np.float32), 2)
            dst.write(vu_grid.astype(np.float32), 3)
            
            dst.set_band_description(1, 'east_velocity')
            dst.set_band_description(2, 'north_velocity')
            dst.set_band_description(3, 'up_velocity')
            
            dst.update_tags(1, units='mm/year')
            dst.update_tags(2, units='mm/year')
            dst.update_tags(3, units='mm/year')
            
            dst.update_tags(
                source='UCERF3',
                model_variant=model_variant,
                conversion_note='Velocities approximated from strain rates',
            )
        
        print(f"Created UCERF3 velocity grid: {output_path}")
        return Path(output_path)


# ============================================================================
# USGS NSHM NetCDF converter
# ============================================================================

class USGSNSHMConverter(VelocityConverter):
    """
    Convert USGS NSHM 2023 geodetic models to GeoTIFF.
    
    NSHM data release includes multiple model variants:
      - Pollitz block model
      - Zeng smoothed model
      - Shen model
      - Evans model
    """
    
    @staticmethod
    def convert(
        input_path: Path,
        output_path: Path,
        model_variant: str = "pollitz",
        **kwargs,
    ) -> Path:
        """
        Convert USGS NSHM NetCDF to GeoTIFF.
        
        Args:
            input_path: NetCDF file or directory
            output_path: Output GeoTIFF
            model_variant: Which model ('pollitz', 'zeng', 'shen', 'evans')
        """
        if xr is None:
            raise ImportError("xarray required for NetCDF conversion")
        if rasterio is None:
            raise ImportError("rasterio required for GeoTIFF writing")
        
        input_path = Path(input_path)
        
        # Find NetCDF file
        if input_path.is_dir():
            nc_files = list(input_path.glob(f"*{model_variant}*.nc"))
            if not nc_files:
                nc_files = list(input_path.glob("*.nc"))
            if not nc_files:
                raise FileNotFoundError(f"No NetCDF files in {input_path}")
            nc_file = nc_files[0]
        else:
            nc_file = input_path
        
        # Load NetCDF
        print(f"Loading {nc_file}")
        ds = xr.open_dataset(nc_file)
        
        # Extract velocity components
        # NSHM format may vary - try common variable names
        var_names = {
            've': ['velocity_east', 've', 'east_velocity', 'veast'],
            'vn': ['velocity_north', 'vn', 'north_velocity', 'vnorth'],
            'vu': ['velocity_up', 'vu', 'up_velocity', 'vup', 'vertical_velocity'],
        }
        
        def find_var(names: List[str]) -> Optional[str]:
            for name in names:
                if name in ds.variables:
                    return name
            return None
        
        ve_var = find_var(var_names['ve'])
        vn_var = find_var(var_names['vn'])
        vu_var = find_var(var_names['vu'])
        
        if not ve_var or not vn_var:
            raise ValueError(
                f"Could not find velocity variables in NetCDF. "
                f"Available: {list(ds.variables.keys())}"
            )
        
        ve = ds[ve_var].values
        vn = ds[vn_var].values
        vu = ds[vu_var].values if vu_var else np.zeros_like(ve)
        
        # Get coordinates
        # Try common names
        lon_var = find_var(['lon', 'longitude', 'x'])
        lat_var = find_var(['lat', 'latitude', 'y'])
        
        if not lon_var or not lat_var:
            raise ValueError(
                f"Could not find coordinate variables. "
                f"Available: {list(ds.variables.keys())}"
            )
        
        lons = ds[lon_var].values
        lats = ds[lat_var].values
        
        # Convert to mm/year if needed
        # Check units
        ve_units = ds[ve_var].attrs.get('units', 'm/year')
        if 'm/year' in ve_units.lower() or 'm/yr' in ve_units.lower():
            print(f"Converting from m/year to mm/year")
            ve *= 1000
            vn *= 1000
            vu *= 1000
        
        # Determine grid spacing and create GeoTIFF
        if lons.ndim == 1 and lats.ndim == 1:
            # 1D coordinate arrays - create mesh
            ny, nx = len(lats), len(lons)
            transform = from_bounds(lons[0], lats[0], lons[-1], lats[-1], nx, ny)
        else:
            # 2D coordinate arrays
            ny, nx = lons.shape
            lon_min, lon_max = lons.min(), lons.max()
            lat_min, lat_max = lats.min(), lats.max()
            transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)
        
        # Write GeoTIFF
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=ny,
            width=nx,
            count=3,
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform,
            compress='deflate',
        ) as dst:
            dst.write(ve.astype(np.float32), 1)
            dst.write(vn.astype(np.float32), 2)
            dst.write(vu.astype(np.float32), 3)
            
            dst.set_band_description(1, 'east_velocity')
            dst.set_band_description(2, 'north_velocity')
            dst.set_band_description(3, 'up_velocity')
            
            dst.update_tags(1, units='mm/year')
            dst.update_tags(2, units='mm/year')
            dst.update_tags(3, units='mm/year')
            
            dst.update_tags(
                source='USGS_NSHM_2023',
                model_variant=model_variant,
            )
        
        ds.close()
        
        print(f"Created USGS NSHM velocity grid: {output_path}")
        return Path(output_path)


# ============================================================================
# GEM GSRM ASCII converter
# ============================================================================

class GSRMConverter(VelocityConverter):
    """
    Convert GEM GSRM ASCII strain rate grids to velocity GeoTIFF.
    
    GSRM provides strain rate tensors globally.
    """
    
    @staticmethod
    def convert(
        input_path: Path,
        output_path: Path,
        **kwargs,
    ) -> Path:
        """
        Convert GSRM ASCII to GeoTIFF.
        
        Similar approach to UCERF3 - convert strain rates to velocities.
        """
        if rasterio is None:
            raise ImportError("rasterio required")
        
        input_path = Path(input_path)
        
        # Find strain rate files
        if input_path.is_dir():
            ascii_files = list(input_path.glob("*.txt"))
            if not ascii_files:
                ascii_files = list(input_path.glob("*.dat"))
            if not ascii_files:
                raise FileNotFoundError(f"No ASCII files in {input_path}")
            ascii_file = ascii_files[0]
        else:
            ascii_file = input_path
        
        # Parse GSRM format
        # Typical: lon, lat, exx, eyy, exy, [uncertainties]
        data = np.loadtxt(ascii_file, skiprows=1)
        
        lons = data[:, 0]
        lats = data[:, 1]
        exx = data[:, 2]
        eyy = data[:, 3]
        exy = data[:, 4]
        
        # Convert to velocities (same approach as UCERF3)
        R_earth = 6371000
        deg_to_rad = np.pi / 180
        lat_rad = lats * deg_to_rad
        
        # Estimate grid spacing
        unique_lons = np.unique(lons)
        resolution = np.median(np.diff(unique_lons))
        
        dx = resolution * deg_to_rad * R_earth * np.cos(lat_rad)
        dy = resolution * deg_to_rad * R_earth
        
        ve = exx * dx * 1000
        vn = eyy * dy * 1000
        vu = np.zeros_like(ve)
        
        # Grid
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        
        nx = int((lon_max - lon_min) / resolution) + 1
        ny = int((lat_max - lat_min) / resolution) + 1
        
        lon_grid = np.linspace(lon_min, lon_max, nx)
        lat_grid = np.linspace(lat_min, lat_max, ny)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        if griddata is None:
            raise ImportError("scipy required")
        
        ve_grid = griddata((lons, lats), ve, (lon_mesh, lat_mesh), method='linear')
        vn_grid = griddata((lons, lats), vn, (lon_mesh, lat_mesh), method='linear')
        vu_grid = np.zeros_like(ve_grid)
        
        # Write GeoTIFF
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=ny,
            width=nx,
            count=3,
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform,
            compress='deflate',
        ) as dst:
            dst.write(ve_grid.astype(np.float32), 1)
            dst.write(vn_grid.astype(np.float32), 2)
            dst.write(vu_grid.astype(np.float32), 3)
            
            dst.set_band_description(1, 'east_velocity')
            dst.set_band_description(2, 'north_velocity')
            dst.set_band_description(3, 'up_velocity')
            
            dst.update_tags(1, units='mm/year')
            dst.update_tags(2, units='mm/year')
            dst.update_tags(3, units='mm/year')
            
            dst.update_tags(source='GEM_GSRM_v2.1')
        
        print(f"Created GSRM velocity grid: {output_path}")
        return Path(output_path)


# ============================================================================
# EarthScope point data converter
# ============================================================================

class EarthScopeConverter(VelocityConverter):
    """
    Convert EarthScope/UNAVCO GNSS point velocities to gridded GeoTIFF.
    """
    
    @staticmethod
    def convert(
        input_path: Path,
        output_path: Path,
        resolution: float = 0.1,
        method: str = 'linear',
        **kwargs,
    ) -> Path:
        """
        Grid EarthScope point data.
        
        Args:
            input_path: ASCII file with station velocities
            output_path: Output GeoTIFF
            resolution: Grid spacing in degrees
            method: Interpolation method ('linear', 'cubic', 'nearest')
        """
        if griddata is None:
            raise ImportError("scipy required for gridding")
        if rasterio is None:
            raise ImportError("rasterio required")
        
        # Parse EarthScope format
        # Typical: lon, lat, ve, vn, vu, [sig_e, sig_n, sig_u], [site_code]
        data = np.loadtxt(input_path, skiprows=1)
        
        lons = data[:, 0]
        lats = data[:, 1]
        ve = data[:, 2]
        vn = data[:, 3]
        vu = data[:, 4] if data.shape[1] > 4 else np.zeros_like(ve)
        
        # Create regular grid
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        
        nx = int((lon_max - lon_min) / resolution) + 1
        ny = int((lat_max - lat_min) / resolution) + 1
        
        lon_grid = np.linspace(lon_min, lon_max, nx)
        lat_grid = np.linspace(lat_min, lat_max, ny)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        print(f"Gridding {len(lons)} GNSS stations → {nx}x{ny} grid")
        
        # Interpolate
        ve_grid = griddata((lons, lats), ve, (lon_mesh, lat_mesh), method=method)
        vn_grid = griddata((lons, lats), vn, (lon_mesh, lat_mesh), method=method)
        vu_grid = griddata((lons, lats), vu, (lon_mesh, lat_mesh), method=method)
        
        # Write GeoTIFF
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=ny,
            width=nx,
            count=3,
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform,
            compress='deflate',
        ) as dst:
            dst.write(ve_grid.astype(np.float32), 1)
            dst.write(vn_grid.astype(np.float32), 2)
            dst.write(vu_grid.astype(np.float32), 3)
            
            dst.set_band_description(1, 'east_velocity')
            dst.set_band_description(2, 'north_velocity')
            dst.set_band_description(3, 'up_velocity')
            
            dst.update_tags(1, units='mm/year')
            dst.update_tags(2, units='mm/year')
            dst.update_tags(3, units='mm/year')
            
            dst.update_tags(
                source='EarthScope_GNSS',
                interpolation_method=method,
                n_stations=len(lons),
            )
        
        print(f"Created gridded velocity model: {output_path}")
        return Path(output_path)


# ============================================================================
# Converter registry and dispatcher
# ============================================================================

CONVERTERS = {
    'ucerf3': UCERF3Converter,
    'usgs_nshm': USGSNSHMConverter,
    'gsrm': GSRMConverter,
    'earthscope': EarthScopeConverter,
}


def convert_model(
    model_name: str,
    input_path: Path,
    output_path: Optional[Path] = None,
    **kwargs,
) -> Path:
    """
    Convert velocity model to PROJ-compatible GeoTIFF.
    
    Args:
        model_name: Model identifier (determines converter)
        input_path: Source file/directory
        output_path: Output GeoTIFF (auto-generated if None)
        **kwargs: Converter-specific options
        
    Returns:
        Path to created GeoTIFF
    """
    # Determine converter
    converter_key = None
    for key in CONVERTERS:
        if key in model_name.lower():
            converter_key = key
            break
    
    if not converter_key:
        raise ValueError(
            f"No converter found for model '{model_name}'. "
            f"Available: {list(CONVERTERS.keys())}"
        )
    
    # Auto-generate output path
    if output_path is None:
        output_path = input_path.parent / f"{model_name}_velocity.tif"
    
    # Run converter
    converter = CONVERTERS[converter_key]
    return converter.convert(input_path, output_path, **kwargs)


# ============================================================================
# Convenience exports
# ============================================================================

__all__ = [
    'VelocityConverter',
    'UCERF3Converter',
    'USGSNSHMConverter',
    'GSRMConverter',
    'EarthScopeConverter',
    'convert_model',
    'CONVERTERS',
]