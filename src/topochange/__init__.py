"""Topographic change detection and uncertainty quantification.

This package provides tools for:
- Loading and comparing raster DEMs and point clouds
- CRS transformations including vertical datum conversions
- Variogram-based uncertainty analysis
- Regional uncertainty propagation
- Interactive stable area identification

Example usage:
    from topochange import Raster, RasterPair, VariogramAnalysis

    # Load two DEMs
    dem1 = Raster.from_file("dem_2019.tif")
    dem2 = Raster.from_file("dem_2023.tif")

    # Create pair and compute difference
    pair = RasterPair(dem1, dem2)
    result = pair.compute_difference()
"""

__version__ = "0.1.0"

# Core raster classes
from .raster import Raster
from .rasterpair import RasterPair

# Point cloud classes
from .pointcloud import PointCloud
from .pointcloudpair import PointCloudPair

# Uncertainty analysis
from .variogram import (
    RasterDataHandler,
    StatisticalAnalysis,
    VariogramAnalysis,
)
from .uncertainty import RegionalUncertaintyEstimator

# Stable area analysis
from .stable_area_analysis import (
    StableAreaSelector,
    StableAreaRasterizer,
    StableAreaAnalyzer,
)

# CRS utilities
from .crs_history import CRSHistory
from .pipeline_builder import CRSState, build_vertical_pipeline

__all__ = [
    # Version
    "__version__",
    # Raster
    "Raster",
    "RasterPair",
    # Point cloud
    "PointCloud",
    "PointCloudPair",
    # Uncertainty
    "RasterDataHandler",
    "StatisticalAnalysis",
    "VariogramAnalysis",
    "RegionalUncertaintyEstimator",
    # Stable areas
    "StableAreaSelector",
    "StableAreaRasterizer",
    "StableAreaAnalyzer",
    # CRS
    "CRSHistory",
    "CRSState",
    "build_vertical_pipeline",
]
