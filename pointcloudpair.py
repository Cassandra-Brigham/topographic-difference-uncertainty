# pointcloudpair.py
"""
PointCloudPair class for comparing, transforming, aligning, and differencing two point clouds.

This module provides tools for:
- Comparing CRS, epoch, geoid, and other parameters between two point clouds
- Transforming pc1 (compare) to match pc2 (reference)'s reference frame
- ICP-based alignment using small_gicp
- 3D point cloud differencing
- 2D DEM-based differencing via RasterPair

The transformation pipeline follows the order:
1. Dynamic epoch transformation (if epochs differ)
2. Horizontal CRS reprojection (if horizontal CRS differs)
3. Vertical datum transformation (if vertical kind or geoid differs)
4. ICP alignment (optional fine registration)
5. DEM creation and differencing (for 2D analysis)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

import numpy as np
# Use pdal_wrapper for Colab compatibility (falls back to native pdal locally)
try:
    from pdal_wrapper import pdal
except ImportError:
    import pdal
from pyproj import CRS as _CRS
from pyproj.crs import CompoundCRS

# Handle imports
try:
    from pointcloud import PointCloud
    from raster import Raster
    from rasterpair import RasterPair
    from crs_utils import (
        _ensure_crs_obj,
        is_3d_geographic_crs,
        extract_ellipsoidal_height_as_vertical_crs,
    )
    from unit_utils import (
        UnitInfo,
        UNKNOWN_UNIT,
        METER,
        lookup_unit,
        get_conversion_factor,
    )
except ImportError:
    from pointcloud import PointCloud
    from raster import Raster
    from rasterpair import RasterPair
    from crs_utils import (
        _ensure_crs_obj,
        is_3d_geographic_crs,
        extract_ellipsoidal_height_as_vertical_crs,
    )
    from unit_utils import (
        UnitInfo,
        UNKNOWN_UNIT,
        METER,
        lookup_unit,
        get_conversion_factor,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def _has_small_gicp() -> bool:
    """Check if small_gicp is available."""
    try:
        import small_gicp
        return True
    except ImportError:
        return False


def _crs_equivalent(crs1: Any, crs2: Any) -> bool:
    """
    Check if two CRS are equivalent.
    
    Uses EPSG comparison first, then falls back to pyproj equals().
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
        
        return obj1.equals(obj2)
    except Exception:
        return str(crs1) == str(crs2)


def _geoid_equivalent(geoid1: Optional[str], geoid2: Optional[str]) -> bool:
    """
    Check if two geoid model names are equivalent.
    
    Handles case-insensitive comparison and common naming variations.
    """
    if geoid1 is None and geoid2 is None:
        return True
    if geoid1 is None or geoid2 is None:
        return False
    
    def normalize(g):
        g = str(g).lower().strip()
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
    
    Returns
    -------
    tuple[bool, float or None]
        (are_equivalent, conversion_factor)
    """
    if unit1 is None and unit2 is None:
        return True, None
    if unit1 is None or unit2 is None:
        return False, None
    
    # Get unit names
    if hasattr(unit1, 'name'):
        name1 = unit1.name
    else:
        name1 = str(unit1).lower()
    
    if hasattr(unit2, 'name'):
        name2 = unit2.name
    else:
        name2 = str(unit2).lower()
    
    if name1 == name2:
        return True, None
    
    if name1 == "unknown" or name2 == "unknown":
        return False, None
    
    # Try to get conversion factor
    try:
        unit1_obj = lookup_unit(name1) if isinstance(unit1, str) else unit1
        unit2_obj = lookup_unit(name2) if isinstance(unit2, str) else unit2
        if unit1_obj and unit2_obj:
            factor = get_conversion_factor(unit1_obj, unit2_obj)
            return False, factor
    except Exception:
        pass
    
    return False, None


def _load_points_from_las(filename: str, max_points: Optional[int] = None) -> np.ndarray:
    """
    Load XYZ points from a LAS/LAZ file using PDAL.
    
    Parameters
    ----------
    filename : str
        Path to LAS/LAZ file
    max_points : int, optional
        Maximum number of points to load (for downsampling)
        
    Returns
    -------
    np.ndarray
        Nx3 array of XYZ coordinates
    """
    pipeline_spec = {
        "pipeline": [
            {"type": "readers.las", "filename": str(filename)},
        ]
    }
    
    if max_points is not None:
        # Add random sampling filter
        pipeline_spec["pipeline"].append({
            "type": "filters.sample",
            "radius": 0,  # Use count-based sampling
        })
    
    pipe = pdal.Pipeline(json.dumps(pipeline_spec))
    pipe.execute()
    
    arrays = pipe.arrays
    if not arrays or len(arrays) == 0:
        raise ValueError(f"No points loaded from {filename}")
    
    arr = arrays[0]
    points = np.column_stack([arr['X'], arr['Y'], arr['Z']]).astype(np.float64)
    
    if max_points is not None and len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    return points


def _save_transformed_las(
    source_filename: str,
    output_filename: str,
    transformation_matrix: np.ndarray,
) -> None:
    """
    Apply a 4x4 transformation matrix to a LAS file and save.
    
    Parameters
    ----------
    source_filename : str
        Input LAS/LAZ file
    output_filename : str
        Output LAS/LAZ file
    transformation_matrix : np.ndarray
        4x4 homogeneous transformation matrix
    """
    # Extract rotation and translation from 4x4 matrix
    R = transformation_matrix[:3, :3]
    t = transformation_matrix[:3, 3]
    
    # Build PDAL transformation matrix string (row-major, 16 values)
    matrix_str = " ".join(str(x) for x in transformation_matrix.flatten())
    
    pipeline_spec = {
        "pipeline": [
            {"type": "readers.las", "filename": str(source_filename)},
            {
                "type": "filters.transformation",
                "matrix": matrix_str,
            },
            {"type": "writers.las", "filename": str(output_filename)},
        ]
    }
    
    pipe = pdal.Pipeline(json.dumps(pipeline_spec))
    pipe.execute()


# =============================================================================
# PointCloudPair Class
# =============================================================================

@dataclass
class PointCloudPair:
    """
    Pair of point clouds for comparison, transformation, alignment, and differencing.
    
    Attributes
    ----------
    pc1 : PointCloud
        The "compare" point cloud (to be transformed)
    pc2 : PointCloud
        The "reference" point cloud (target reference frame)
    
    Notes
    -----
    By convention:
    - pc1 is the "compare" or "source" point cloud
    - pc2 is the "reference" or "target" point cloud
    - Transformations are applied to pc1 to match pc2
    - Differences are computed as pc2 - pc1 (positive = gain)
    """
    
    pc1: PointCloud  # Compare
    pc2: PointCloud  # Reference
    
    # Internal state
    _transformation_history: List[Dict[str, Any]] = field(default_factory=list)
    _pc1_transformed: Optional[PointCloud] = field(default=None, repr=False)
    _alignment_result: Optional[Dict[str, Any]] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize internal state."""
        self._transformation_history = []
        self._pc1_transformed = None
        self._alignment_result = None
    
    # =========================================================================
    # Comparison Methods
    # =========================================================================
    
    def check_all_match(self) -> Dict[str, Any]:
        """
        Check if all CRS/metadata parameters match between pc1 and pc2.
        
        Returns
        -------
        dict
            Dictionary with match status for each parameter:
            - compound_crs, horizontal_crs, vertical_crs
            - geoid, epoch, vertical_units
            - transformations_needed: list of required transformations
        """
        result = {
            'compound_crs': {'match': False, 'pc1': None, 'pc2': None},
            'horizontal_crs': {'match': False, 'pc1': None, 'pc2': None},
            'vertical_crs': {'match': False, 'pc1': None, 'pc2': None},
            'geoid': {'match': False, 'pc1': None, 'pc2': None},
            'epoch': {'match': False, 'pc1': None, 'pc2': None},
            'vertical_units': {'match': False, 'pc1': None, 'pc2': None},
            'transformations_needed': [],
        }
        
        # Compound/3D CRS
        pc1_comp = (
            getattr(self.pc1, 'current_compound_crs', None) or
            getattr(self.pc1, 'original_compound_crs', None)
        )
        pc2_comp = (
            getattr(self.pc2, 'current_compound_crs', None) or
            getattr(self.pc2, 'original_compound_crs', None)
        )
        result['compound_crs']['pc1'] = (
            pc1_comp[:100] + '...' if pc1_comp and len(str(pc1_comp)) > 100 else pc1_comp
        )
        result['compound_crs']['pc2'] = (
            pc2_comp[:100] + '...' if pc2_comp and len(str(pc2_comp)) > 100 else pc2_comp
        )
        result['compound_crs']['match'] = _crs_equivalent(pc1_comp, pc2_comp)
        
        # Horizontal CRS
        pc1_horiz = (
            getattr(self.pc1, 'current_horizontal_crs', None) or
            getattr(self.pc1, 'original_horizontal_crs', None)
        )
        pc2_horiz = (
            getattr(self.pc2, 'current_horizontal_crs', None) or
            getattr(self.pc2, 'original_horizontal_crs', None)
        )
        result['horizontal_crs']['pc1'] = (
            pc1_horiz[:100] + '...' if pc1_horiz and len(str(pc1_horiz)) > 100 else pc1_horiz
        )
        result['horizontal_crs']['pc2'] = (
            pc2_horiz[:100] + '...' if pc2_horiz and len(str(pc2_horiz)) > 100 else pc2_horiz
        )
        result['horizontal_crs']['match'] = _crs_equivalent(pc1_horiz, pc2_horiz)
        if not result['horizontal_crs']['match']:
            result['transformations_needed'].append('horizontal_crs')
        
        # Vertical CRS
        pc1_vert = (
            getattr(self.pc1, 'current_vertical_crs', None) or
            getattr(self.pc1, 'original_vertical_crs', None)
        )
        pc2_vert = (
            getattr(self.pc2, 'current_vertical_crs', None) or
            getattr(self.pc2, 'original_vertical_crs', None)
        )
        result['vertical_crs']['pc1'] = (
            pc1_vert[:100] + '...' if pc1_vert and len(str(pc1_vert)) > 100 else pc1_vert
        )
        result['vertical_crs']['pc2'] = (
            pc2_vert[:100] + '...' if pc2_vert and len(str(pc2_vert)) > 100 else pc2_vert
        )
        result['vertical_crs']['match'] = _crs_equivalent(pc1_vert, pc2_vert)
        
        # Geoid model
        pc1_geoid = getattr(self.pc1, 'geoid_model', None)
        pc2_geoid = getattr(self.pc2, 'geoid_model', None)
        result['geoid']['pc1'] = pc1_geoid
        result['geoid']['pc2'] = pc2_geoid
        result['geoid']['match'] = _geoid_equivalent(pc1_geoid, pc2_geoid)
        
        # Check if vertical datum transformation is needed
        pc1_ortho = getattr(self.pc1, 'is_orthometric', None)
        pc2_ortho = getattr(self.pc2, 'is_orthometric', None)
        if not result['vertical_crs']['match'] or not result['geoid']['match']:
            if pc1_ortho != pc2_ortho or not result['geoid']['match']:
                result['transformations_needed'].append('vertical_datum')
        
        # Epoch
        pc1_epoch = getattr(self.pc1, 'epoch', None)
        pc2_epoch = getattr(self.pc2, 'epoch', None)
        result['epoch']['pc1'] = pc1_epoch
        result['epoch']['pc2'] = pc2_epoch
        if pc1_epoch is not None and pc2_epoch is not None:
            result['epoch']['match'] = abs(pc1_epoch - pc2_epoch) < 0.001  # ~8 hours
        else:
            result['epoch']['match'] = pc1_epoch is None and pc2_epoch is None
        if not result['epoch']['match'] and pc1_epoch is not None and pc2_epoch is not None:
            result['transformations_needed'].append('epoch')
        
        # Vertical units
        pc1_vunit = getattr(self.pc1, 'vertical_unit', UNKNOWN_UNIT)
        pc2_vunit = getattr(self.pc2, 'vertical_unit', UNKNOWN_UNIT)
        pc1_vunit_name = pc1_vunit.name if hasattr(pc1_vunit, 'name') else str(pc1_vunit)
        pc2_vunit_name = pc2_vunit.name if hasattr(pc2_vunit, 'name') else str(pc2_vunit)
        result['vertical_units']['pc1'] = pc1_vunit_name
        result['vertical_units']['pc2'] = pc2_vunit_name
        units_match, _ = _units_equivalent(pc1_vunit, pc2_vunit)
        result['vertical_units']['match'] = units_match
        if not units_match:
            result['transformations_needed'].append('vertical_units')
        
        return result
    
    def print_comparison(self) -> None:
        """Print a human-readable comparison of the two point clouds."""
        comparison = self.check_all_match()
        
        print("\n" + "=" * 60)
        print("PointCloudPair Comparison")
        print("=" * 60)
        
        print(f"\nCompare (pc1):   {Path(self.pc1.filename).name}")
        print(f"Reference (pc2): {Path(self.pc2.filename).name}")
        
        print(f"\n{'Parameter':<20} {'Match':<8} {'PC1':<20} {'PC2':<20}")
        print("-" * 70)
        
        # Horizontal CRS - get directly from point clouds, not from truncated comparison dict
        pc1_horiz = (
            getattr(self.pc1, 'current_horizontal_crs', None) or
            getattr(self.pc1, 'original_horizontal_crs', None)
        )
        pc2_horiz = (
            getattr(self.pc2, 'current_horizontal_crs', None) or
            getattr(self.pc2, 'original_horizontal_crs', None)
        )
        
        pc1_str = "None"
        pc2_str = "None"
        
        if pc1_horiz is not None:
            try:
                crs_obj = _ensure_crs_obj(pc1_horiz)
                epsg = crs_obj.to_epsg()
                if epsg:
                    pc1_str = f"EPSG:{epsg}"
                elif crs_obj.name:
                    # Truncate long names
                    name = crs_obj.name
                    pc1_str = name[:18] + ".." if len(name) > 20 else name
                else:
                    pc1_str = "Custom"
            except Exception:
                pc1_str = "Unknown"
        
        if pc2_horiz is not None:
            try:
                crs_obj = _ensure_crs_obj(pc2_horiz)
                epsg = crs_obj.to_epsg()
                if epsg:
                    pc2_str = f"EPSG:{epsg}"
                elif crs_obj.name:
                    name = crs_obj.name
                    pc2_str = name[:18] + ".." if len(name) > 20 else name
                else:
                    pc2_str = "Custom"
            except Exception:
                pc2_str = "Unknown"
        
        match_str = "✓" if comparison['horizontal_crs']['match'] else "✗"
        print(f"{'Horizontal CRS':<20} {match_str:<8} {pc1_str:<20} {pc2_str:<20}")
        
        # Vertical CRS - get directly from point clouds
        pc1_vert = (
            getattr(self.pc1, 'current_vertical_crs', None) or
            getattr(self.pc1, 'original_vertical_crs', None)
        )
        pc2_vert = (
            getattr(self.pc2, 'current_vertical_crs', None) or
            getattr(self.pc2, 'original_vertical_crs', None)
        )
        
        # Determine vertical type
        def get_vertical_type(vert_crs, pc) -> str:
            if vert_crs is None:
                return "None"
            
            vert_str = str(vert_crs).lower()
            
            # Check for ellipsoidal
            if "ellipsoidal" in vert_str:
                return "Ellipsoidal"
            
            # Check is_orthometric attribute
            is_ortho = getattr(pc, 'is_orthometric', None)
            if is_ortho is True:
                return "Orthometric"
            elif is_ortho is False:
                return "Ellipsoidal"
            
            # Try to extract from CRS
            try:
                crs_obj = _ensure_crs_obj(vert_crs)
                epsg = crs_obj.to_epsg()
                if epsg:
                    # Known orthometric EPSG codes
                    if epsg in (5703, 5866, 6647, 8228):  # NAVD88, CGVD2013, etc.
                        return "Orthometric"
                    return f"EPSG:{epsg}"
                if crs_obj.name:
                    name = crs_obj.name
                    if "navd" in name.lower() or "orthometric" in name.lower():
                        return "Orthometric"
                    return name[:18] + ".." if len(name) > 20 else name
            except Exception:
                pass
            
            return "Unknown"
        
        pc1_vert_str = get_vertical_type(pc1_vert, self.pc1)
        pc2_vert_str = get_vertical_type(pc2_vert, self.pc2)
        
        match_str = "✓" if comparison['vertical_crs']['match'] else "✗"
        print(f"{'Vertical CRS':<20} {match_str:<8} {pc1_vert_str:<20} {pc2_vert_str:<20}")
        
        # Geoid
        match_str = "✓" if comparison['geoid']['match'] else "✗"
        pc1_geoid = comparison['geoid']['pc1'] or "None"
        pc2_geoid = comparison['geoid']['pc2'] or "None"
        # Truncate long geoid names
        pc1_geoid = pc1_geoid[:18] + ".." if len(pc1_geoid) > 20 else pc1_geoid
        pc2_geoid = pc2_geoid[:18] + ".." if len(pc2_geoid) > 20 else pc2_geoid
        print(f"{'Geoid Model':<20} {match_str:<8} {pc1_geoid:<20} {pc2_geoid:<20}")
        
        # Epoch
        match_str = "✓" if comparison['epoch']['match'] else "✗"
        pc1_epoch = f"{comparison['epoch']['pc1']:.4f}" if comparison['epoch']['pc1'] else "None"
        pc2_epoch = f"{comparison['epoch']['pc2']:.4f}" if comparison['epoch']['pc2'] else "None"
        print(f"{'Epoch':<20} {match_str:<8} {pc1_epoch:<20} {pc2_epoch:<20}")
        
        # Units
        match_str = "✓" if comparison['vertical_units']['match'] else "✗"
        print(f"{'Vertical Units':<20} {match_str:<8} {comparison['vertical_units']['pc1']:<20} {comparison['vertical_units']['pc2']:<20}")
        
        print("-" * 70)
        
        if comparison['transformations_needed']:
            print(f"\nTransformations needed: {', '.join(comparison['transformations_needed'])}")
        else:
            print(f"\n✓ Point clouds are fully aligned!")
        
        if self._transformation_history:
            print(f"\nTransformation steps applied:")
            for i, step in enumerate(self._transformation_history, 1):
                print(f"  {i}. {step.get('step', 'unknown')}")
        
        print("=" * 60 + "\n")

    
    # =========================================================================
    # Transformation Methods
    # =========================================================================
    
    def transform_compare_to_match_reference(
        self,
        skip_epoch: bool = False,
        skip_horizontal: bool = False,
        skip_vertical: bool = False,
        skip_units: bool = False,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> PointCloud:
        """
        Transform pc1 (compare) to match pc2 (reference)'s reference frame.
        
        All transformations are composed into a SINGLE PDAL pipeline pass
        for efficiency.
        """
        import sys
        
        comparison = self.check_all_match()
        
        if verbose:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print("PointCloudPair: Transform compare to match reference", file=sys.stderr)
            print(f"{'=' * 60}", file=sys.stderr)
            print(f"Transformations needed: {comparison['transformations_needed']}", file=sys.stderr)
        
        self._transformation_history = []
        
        # Get target parameters from reference (pc2)
        target_epoch = getattr(self.pc2, 'epoch', None)
        target_horiz_crs = (
            getattr(self.pc2, 'current_horizontal_crs', None) or
            getattr(self.pc2, 'original_horizontal_crs', None)
        )
        target_vert_crs = (
            getattr(self.pc2, 'current_vertical_crs', None) or
            getattr(self.pc2, 'original_vertical_crs', None)
        )
        target_geoid = getattr(self.pc2, 'geoid_model', None)
        
        # Determine vertical kinds
        source_is_ortho = getattr(self.pc1, 'is_orthometric', None)
        target_is_ortho = getattr(self.pc2, 'is_orthometric', None)
        
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
        source_geoid = getattr(self.pc1, 'geoid_model', None)
        
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
            src_epoch = getattr(self.pc1, 'epoch', None)
            if needs_epoch:
                print(f"  Epoch: {src_epoch:.4f} → {target_epoch:.4f}", file=sys.stderr)
            if needs_vertical:
                print(f"  Vertical: {source_vertical_kind} → {target_vertical_kind}", file=sys.stderr)
                print(f"  Geoid: {source_geoid} → {target_geoid}", file=sys.stderr)
            if needs_horizontal:
                print(f"  Horizontal CRS reprojection needed", file=sys.stderr)
        
        # SINGLE warp_pointcloud call with ALL parameters
        if needs_epoch or needs_vertical or needs_horizontal:
            if verbose:
                print(f"\nExecuting combined transformation pipeline...", file=sys.stderr)
            
            # Build output filename
            src_path = Path(self.pc1.filename)
            suffix_parts = []
            if needs_epoch:
                suffix_parts.append(f"epoch{target_epoch:.2f}".replace(".", "p"))
            if needs_vertical:
                suffix_parts.append(f"{target_vertical_kind[:4]}")
            if needs_horizontal:
                suffix_parts.append("reproj")
            suffix = "_".join(suffix_parts) if suffix_parts else "transformed"
            output_path = src_path.with_name(src_path.stem + f"_{suffix}" + src_path.suffix)
            
            current = self.pc1.warp_pointcloud(
                # Epoch parameters
                dynamic_target_epoch=target_epoch if needs_epoch else None,
                # Vertical parameters  
                source_vertical_kind=source_vertical_kind if needs_vertical else None,
                target_vertical_kind=target_vertical_kind if needs_vertical else None,
                source_geoid_model=source_geoid if needs_vertical else None,
                target_geoid_model=target_geoid if needs_vertical else None,
                # Horizontal parameters
                target_horizontal_crs=target_horiz_crs if needs_horizontal else None,
                # Output
                output_path=output_path,
                overwrite=overwrite,
            )
            
            self._transformation_history.append({
                'step': 'combined_transform',
                'needs_epoch': needs_epoch,
                'needs_vertical': needs_vertical,
                'needs_horizontal': needs_horizontal,
                'source_epoch': getattr(self.pc1, 'epoch', None),
                'target_epoch': target_epoch,
                'source_vertical_kind': source_vertical_kind,
                'target_vertical_kind': target_vertical_kind,
                'output_file': current.filename,
            })
            
            if verbose:
                print(f"  ✓ Combined transformation complete", file=sys.stderr)
        else:
            current = self.pc1
            if verbose:
                print(f"\nNo transformations needed.", file=sys.stderr)
        
        # Update metadata to match reference
        current.add_metadata(
            horizontal_CRS=target_horiz_crs,
            vertical_CRS=target_vert_crs,
            geoid_model=target_geoid,
            epoch=target_epoch,
        )
        
        # Cache result
        self._pc1_transformed = current
        
        if verbose:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print("Transformation pipeline complete", file=sys.stderr)
            print(f"Output: {current.filename}", file=sys.stderr)
            print(f"{'=' * 60}\n", file=sys.stderr)
        
        return current
    
    def warp_pointclouds_to_common_crs(
        self,
        target_pc: str = "pc2",
        target_crs_proj: Optional[Any] = None,
        verbose: bool = True,
    ) -> "PointCloudPair":
        """
        Warp point clouds to a common reference frame.
        
        By default, transforms pc1 to match pc2's reference frame.
        
        Parameters
        ----------
        target_pc : str, {"pc1", "pc2"}
            Which point cloud to use as the reference
            - "pc1": Transform pc2 to match pc1
            - "pc2": Transform pc1 to match pc2 (default)
        target_crs_proj : Any, optional
            If provided, warp both to this CRS (horizontal only)
        verbose : bool
            Print progress messages
            
        Returns
        -------
        PointCloudPair
            New PointCloudPair with transformed point clouds
        """
        if target_crs_proj is not None:
            # Warp both to explicit target CRS (horizontal only)
            warped_pc1 = self.pc1.warp_pointcloud(target_horizontal_crs=target_crs_proj)
            warped_pc2 = self.pc2.warp_pointcloud(target_horizontal_crs=target_crs_proj)
            return PointCloudPair(warped_pc1, warped_pc2)
        
        if target_pc == "pc2":
            # Transform pc1 to match pc2 (reference)
            transformed_pc1 = self.transform_compare_to_match_reference(verbose=verbose)
            return PointCloudPair(transformed_pc1, self.pc2)
        
        elif target_pc == "pc1":
            # Swap and transform pc2 to match pc1
            swapped_pair = PointCloudPair(self.pc2, self.pc1)
            transformed_pc2 = swapped_pair.transform_compare_to_match_reference(verbose=verbose)
            return PointCloudPair(self.pc1, transformed_pc2)
        
        else:
            raise ValueError(f"target_pc must be 'pc1' or 'pc2', got {target_pc!r}")
    
    # =========================================================================
    # Alignment Methods (ICP via small_gicp)
    # =========================================================================
    
    def align_point_clouds(
        self,
        method: str = "gicp",
        downsample_resolution: float = 0.5,
        max_correspondence_distance: float = 1.0,
        max_iterations: int = 50,
        transformation_epsilon: float = 1e-6,
        num_threads: int = 4,
        apply_transform: bool = True,
        output_path: Optional[str] = None,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Align pc1 to pc2 using ICP registration via small_gicp.
        
        Point clouds are automatically centered before registration to avoid
        numerical issues with large UTM coordinates.
        """
        if not _has_small_gicp():
            raise ImportError(
                "small_gicp is required for point cloud alignment. "
                "Install with: pip install small_gicp"
            )
        
        import small_gicp
        
        if verbose:
            print(f"\n{'=' * 60}")
            print("Point Cloud Alignment (small_gicp)")
            print(f"{'=' * 60}")
            print(f"Method: {method.upper()}")
            print(f"Downsample resolution: {downsample_resolution} m")
            print(f"Max correspondence distance: {max_correspondence_distance} m")
        
        # Use transformed pc1 if available, otherwise original
        source_pc = self._pc1_transformed or self.pc1
        target_pc = self.pc2
        
        # Load points
        if verbose:
            print(f"\nLoading source points from: {Path(source_pc.filename).name}")
        source_points = _load_points_from_las(source_pc.filename)
        
        if verbose:
            print(f"Loading target points from: {Path(target_pc.filename).name}")
        target_points = _load_points_from_las(target_pc.filename)
        
        if verbose:
            print(f"Source points: {len(source_points):,}")
            print(f"Target points: {len(target_points):,}")
        
        # =========================================================================
        # CENTER POINT CLOUDS to avoid voxel coordinate overflow
        # =========================================================================
        # Compute centroid of combined point clouds for consistent centering
        all_points = np.vstack([source_points, target_points])
        centroid = np.mean(all_points, axis=0)
        
        if verbose:
            print(f"\nCentering point clouds (centroid: [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}])")
        
        # Center both point clouds
        source_centered = source_points - centroid
        target_centered = target_points - centroid
        
        if verbose:
            src_range = np.ptp(source_centered, axis=0)
            tgt_range = np.ptp(target_centered, axis=0)
            print(f"Source range after centering: X={src_range[0]:.1f}, Y={src_range[1]:.1f}, Z={src_range[2]:.1f}")
            print(f"Target range after centering: X={tgt_range[0]:.1f}, Y={tgt_range[1]:.1f}, Z={tgt_range[2]:.1f}")
        
        # Create small_gicp point clouds from CENTERED coordinates
        source_cloud = small_gicp.PointCloud(source_centered)
        target_cloud = small_gicp.PointCloud(target_centered)
        
        # Preprocess (downsampling, normal estimation, KdTree)
        if verbose:
            print(f"\nPreprocessing point clouds...")
        
        source_cloud, source_tree = small_gicp.preprocess_points(
            source_cloud,
            downsampling_resolution=downsample_resolution,
            num_threads=num_threads,
        )
        target_cloud, target_tree = small_gicp.preprocess_points(
            target_cloud,
            downsampling_resolution=downsample_resolution,
            num_threads=num_threads,
        )
        
        if verbose:
            print(f"Source points after downsampling: {source_cloud.size():,}")
            print(f"Target points after downsampling: {target_cloud.size():,}")
        
        # Select registration type (pass as string, not enum)
        method_upper = method.upper()
        if method_upper not in ["GICP", "VGICP", "ICP", "PLANE_ICP"]:
            raise ValueError(f"Unknown method: {method}. Use 'gicp', 'vgicp', 'icp', or 'plane_icp'.")

        # Run registration
        if verbose:
            print(f"\nRunning {method_upper} registration...")

        result = small_gicp.align(
            target_cloud,
            source_cloud,
            target_tree,
            registration_type=method_upper,
            max_correspondence_distance=max_correspondence_distance,
            num_threads=num_threads,
        )
        
        # =========================================================================
        # CONVERT TRANSFORMATION back to original coordinate system
        # =========================================================================
        # The transformation T_centered is computed in centered coordinates.
        # To apply it to original coordinates:
        #   p_aligned = T_centered @ (p_original - centroid) + centroid
        #   p_aligned = T_centered @ p_original - T_centered @ centroid + centroid
        #
        # In matrix form, the transformation in original coordinates is:
        #   T_original = Translate(centroid) @ T_centered @ Translate(-centroid)
        
        T_centered = result.T_target_source  # 4x4 transformation in centered coords
        
        # Build translation matrices
        T_to_origin = np.eye(4)
        T_to_origin[:3, 3] = -centroid
        
        T_from_origin = np.eye(4)
        T_from_origin[:3, 3] = centroid
        
        # Compose: T_original = T_from_origin @ T_centered @ T_to_origin
        T_original = T_from_origin @ T_centered @ T_to_origin
        
        if verbose:
            print(f"\nTransformation in centered coordinates:")
            print(T_centered)
            print(f"\nTransformation in original coordinates:")
            print(T_original)
        
        # Compute fitness metrics using ORIGINAL coordinates
        source_transformed = (T_original[:3, :3] @ source_points.T).T + T_original[:3, 3]
        
        # Use KD-tree for nearest neighbor distances
        from scipy.spatial import cKDTree
        target_tree_scipy = cKDTree(target_points)
        distances, _ = target_tree_scipy.query(source_transformed, k=1)
        
        inlier_mask = distances < max_correspondence_distance
        fitness = np.sum(inlier_mask) / len(distances)
        rmse = np.sqrt(np.mean(distances[inlier_mask] ** 2)) if np.any(inlier_mask) else float('inf')
        
        # Extract rotation and translation for reporting
        R = T_original[:3, :3]
        t = T_original[:3, 3]
        
        # Compute rotation angle (magnitude of axis-angle representation)
        rotation_angle_rad = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        rotation_angle_deg = np.degrees(rotation_angle_rad)
        
        alignment_result = {
            'transformation': T_original,
            'transformation_centered': T_centered,
            'centroid': centroid,
            'converged': result.converged if hasattr(result, 'converged') else True,
            'iterations': result.iterations if hasattr(result, 'iterations') else None,
            'fitness': fitness,
            'rmse': rmse,
            'num_correspondences': int(np.sum(inlier_mask)),
            'method': method,
            'downsample_resolution': downsample_resolution,
            'max_correspondence_distance': max_correspondence_distance,
            'translation': t,
            'rotation_angle_deg': rotation_angle_deg,
        }
        
        if verbose:
            print(f"\nAlignment Results:")
            print(f"  Converged: {alignment_result['converged']}")
            print(f"  Fitness (inlier ratio): {fitness:.4f}")
            print(f"  RMSE: {rmse:.4f} m")
            print(f"  Inlier correspondences: {alignment_result['num_correspondences']:,}")
            print(f"  Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
            print(f"  Rotation: {rotation_angle_deg:.4f}°")
        
        # Apply transformation if requested
        if apply_transform:
            if output_path is None:
                src_path = Path(source_pc.filename)
                output_path = str(src_path.with_name(src_path.stem + "_aligned" + src_path.suffix))
            
            if os.path.exists(output_path) and not overwrite:
                raise FileExistsError(f"Output file exists and overwrite=False: {output_path}")
            
            if verbose:
                print(f"\nApplying transformation to: {output_path}")
            
            _save_transformed_las(source_pc.filename, output_path, T_original)
            
            # Load aligned point cloud
            aligned_pc = PointCloud(output_path)
            aligned_pc.from_file()
            
            # Copy metadata from source
            aligned_pc.add_metadata(
                compound_CRS=source_pc.current_compound_crs or source_pc.original_compound_crs,
                horizontal_CRS=source_pc.current_horizontal_crs or source_pc.original_horizontal_crs,
                vertical_CRS=source_pc.current_vertical_crs or source_pc.original_vertical_crs,
                geoid_model=source_pc.geoid_model,
                epoch=source_pc.epoch,
            )
            
            alignment_result['aligned_pc'] = aligned_pc
            alignment_result['output_file'] = output_path
            
            # Update internal state
            self._pc1_transformed = aligned_pc
            
            self._transformation_history.append({
                'step': 'icp_alignment',
                'method': method,
                'fitness': fitness,
                'rmse': rmse,
                'translation': t.tolist(),
                'rotation_deg': rotation_angle_deg,
                'output_file': output_path,
            })
        
        self._alignment_result = alignment_result
        
        if verbose:
            print(f"\n{'=' * 60}\n")
        
        return alignment_result
# ```

    
#     def align_point_clouds(
#         self,
#         method: str = "gicp",
#         downsample_resolution: float = 0.5,
#         max_correspondence_distance: float = 1.0,
#         max_iterations: int = 50,
#         transformation_epsilon: float = 1e-6,
#         num_threads: int = 4,
#         apply_transform: bool = True,
#         output_path: Optional[str] = None,
#         overwrite: bool = True,
#         verbose: bool = True,
#     ) -> Dict[str, Any]:
#         """
#         Align pc1 to pc2 using ICP registration via small_gicp.
        
#         This performs fine registration to correct for small misalignments
#         that remain after CRS transformations (e.g., survey errors, datum shifts).
        
#         Parameters
#         ----------
#         method : str, {"gicp", "vgicp", "icp"}
#             Registration algorithm:
#             - "gicp": Generalized ICP (recommended)
#             - "vgicp": Voxelized GICP (faster for large clouds)
#             - "icp": Standard ICP
#         downsample_resolution : float
#             Voxel size for downsampling during registration (meters)
#         max_correspondence_distance : float
#             Maximum distance for point correspondences (meters)
#         max_iterations : int
#             Maximum ICP iterations
#         transformation_epsilon : float
#             Convergence threshold for transformation change
#         num_threads : int
#             Number of threads for parallel processing
#         apply_transform : bool
#             If True, create a new transformed point cloud file
#         output_path : str, optional
#             Output file path for aligned point cloud
#         overwrite : bool
#             Overwrite existing output file
#         verbose : bool
#             Print progress messages
            
#         Returns
#         -------
#         dict
#             Alignment results:
#             - transformation: 4x4 transformation matrix
#             - converged: whether ICP converged
#             - iterations: number of iterations
#             - fitness: alignment fitness score (inlier ratio)
#             - rmse: root mean square error
#             - aligned_pc: aligned PointCloud (if apply_transform=True)
#         """
#         if not _has_small_gicp():
#             raise ImportError(
#                 "small_gicp is required for point cloud alignment. "
#                 "Install with: pip install small_gicp"
#             )
        
#         import small_gicp
        
#         if verbose:
#             print(f"\n{'=' * 60}")
#             print("Point Cloud Alignment (small_gicp)")
#             print(f"{'=' * 60}")
#             print(f"Method: {method.upper()}")
#             print(f"Downsample resolution: {downsample_resolution} m")
#             print(f"Max correspondence distance: {max_correspondence_distance} m")
        
#         # Use transformed pc1 if available, otherwise original
#         source_pc = self._pc1_transformed or self.pc1
#         target_pc = self.pc2
        
#         # Load points
#         if verbose:
#             print(f"\nLoading source points from: {Path(source_pc.filename).name}")
#         source_points = _load_points_from_las(source_pc.filename)
        
#         if verbose:
#             print(f"Loading target points from: {Path(target_pc.filename).name}")
#         target_points = _load_points_from_las(target_pc.filename)
        
#         if verbose:
#             print(f"Source points: {len(source_points):,}")
#             print(f"Target points: {len(target_points):,}")
        
#         # Create small_gicp point clouds
#         source_cloud = small_gicp.PointCloud(source_points)
#         target_cloud = small_gicp.PointCloud(target_points)
        
#         # Preprocess (downsampling, normal estimation, KdTree)
#         if verbose:
#             print(f"\nPreprocessing point clouds...")
        
#         source_cloud, source_tree = small_gicp.preprocess_points(
#             source_cloud,
#             downsampling_resolution=downsample_resolution,
#             num_threads=num_threads,
#         )
#         target_cloud, target_tree = small_gicp.preprocess_points(
#             target_cloud,
#             downsampling_resolution=downsample_resolution,
#             num_threads=num_threads,
#         )
        
#         if verbose:
#             print(f"Source points after downsampling: {source_cloud.size():,}")
#             print(f"Target points after downsampling: {target_cloud.size():,}")
        
#         # Select registration type
#         if method.lower() == "gicp":
#             reg_type = small_gicp.RegistrationType.GICP
#         elif method.lower() == "vgicp":
#             reg_type = small_gicp.RegistrationType.VGICP
#         elif method.lower() == "icp":
#             reg_type = small_gicp.RegistrationType.ICP
#         else:
#             raise ValueError(f"Unknown method: {method}. Use 'gicp', 'vgicp', or 'icp'.")
        
#         # Run registration
#         if verbose:
#             print(f"\nRunning {method.upper()} registration...")
        
#         result = small_gicp.align(
#             target_cloud,
#             source_cloud,
#             target_tree,
#             reg_type=reg_type,
#             max_correspondence_distance=max_correspondence_distance,
#             num_threads=num_threads,
#         )
        
#         # Extract transformation matrix
#         T = result.T_target_source  # 4x4 transformation matrix
        
#         # Compute fitness metrics
#         # Transform source points and compute distances to target
#         source_transformed = (T[:3, :3] @ source_points.T).T + T[:3, 3]
        
#         # Use KD-tree for nearest neighbor distances
#         from scipy.spatial import cKDTree
#         target_tree_scipy = cKDTree(target_points)
#         distances, _ = target_tree_scipy.query(source_transformed, k=1)
        
#         inlier_mask = distances < max_correspondence_distance
#         fitness = np.sum(inlier_mask) / len(distances)
#         rmse = np.sqrt(np.mean(distances[inlier_mask] ** 2)) if np.any(inlier_mask) else float('inf')
        
#         alignment_result = {
#             'transformation': T,
#             'converged': result.converged if hasattr(result, 'converged') else True,
#             'iterations': result.iterations if hasattr(result, 'iterations') else None,
#             'fitness': fitness,
#             'rmse': rmse,
#             'num_correspondences': int(np.sum(inlier_mask)),
#             'method': method,
#             'downsample_resolution': downsample_resolution,
#             'max_correspondence_distance': max_correspondence_distance,
#         }
        
#         if verbose:
#             print(f"\nAlignment Results:")
#             print(f"  Converged: {alignment_result['converged']}")
#             print(f"  Fitness (inlier ratio): {fitness:.4f}")
#             print(f"  RMSE: {rmse:.4f} m")
#             print(f"  Inlier correspondences: {alignment_result['num_correspondences']:,}")
#             print(f"\nTransformation matrix:")
#             print(T)
        
#         # Apply transformation if requested
#         if apply_transform:
#             if output_path is None:
#                 src_path = Path(source_pc.filename)
#                 output_path = str(src_path.with_name(src_path.stem + "_aligned" + src_path.suffix))
            
#             if os.path.exists(output_path) and not overwrite:
#                 raise FileExistsError(f"Output file exists and overwrite=False: {output_path}")
            
#             if verbose:
#                 print(f"\nApplying transformation to: {output_path}")
            
#             _save_transformed_las(source_pc.filename, output_path, T)
            
#             # Load aligned point cloud
#             aligned_pc = PointCloud(output_path)
#             aligned_pc.from_file()
            
#             # Copy metadata from source
#             aligned_pc.add_metadata(
#                 compound_CRS=source_pc.current_compound_crs or source_pc.original_compound_crs,
#                 horizontal_CRS=source_pc.current_horizontal_crs or source_pc.original_horizontal_crs,
#                 vertical_CRS=source_pc.current_vertical_crs or source_pc.original_vertical_crs,
#                 geoid_model=source_pc.geoid_model,
#                 epoch=source_pc.epoch,
#             )
            
#             alignment_result['aligned_pc'] = aligned_pc
#             alignment_result['output_file'] = output_path
            
#             # Update internal state
#             self._pc1_transformed = aligned_pc
            
#             self._transformation_history.append({
#                 'step': 'icp_alignment',
#                 'method': method,
#                 'fitness': fitness,
#                 'rmse': rmse,
#                 'output_file': output_path,
#             })
        
#         self._alignment_result = alignment_result
        
#         if verbose:
#             print(f"\n{'=' * 60}\n")
        
#         return alignment_result
    
    # =========================================================================
    # DEM Creation Methods
    # =========================================================================
    
    def create_dem_pair(
        self,
        dem_type: str = "dtm",
        resolution: float = 1.0,
        interpolation: str = "idw",
        use_transformed: bool = True,
        output_dir: Optional[str] = None,
        overwrite: bool = True,
        verbose: bool = True,
        **dem_kwargs,
    ) -> Tuple[Raster, Raster]:
        """
        Create DEMs from both point clouds.
        
        Parameters
        ----------
        dem_type : str, {"dtm", "dsm"}
            Type of DEM to create
        resolution : float
            Output resolution in map units (typically meters)
        interpolation : str
            Interpolation method ("tin", "idw", "min", "max", "mean", etc.)
        use_transformed : bool
            If True, use the transformed pc1 (if available)
        output_dir : str, optional
            Directory for output files (default: same as input files)
        overwrite : bool
            Overwrite existing output files
        verbose : bool
            Print progress messages
        **dem_kwargs
            Additional arguments passed to PointCloud.create_dem()
            
        Returns
        -------
        tuple[Raster, Raster]
            (dem1, dem2) - DEMs created from pc1 and pc2
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Creating {dem_type.upper()} pair")
            print(f"{'=' * 60}")
            print(f"Resolution: {resolution} m")
            print(f"Interpolation: {interpolation}")
        
        # Select source for pc1
        pc1_source = self._pc1_transformed if use_transformed and self._pc1_transformed else self.pc1
        
        # Determine output paths
        if output_dir is None:
            dir1 = Path(pc1_source.filename).parent
            dir2 = Path(self.pc2.filename).parent
        else:
            dir1 = dir2 = Path(output_dir)
            dir1.mkdir(parents=True, exist_ok=True)
        
        out1 = dir1 / f"{Path(pc1_source.filename).stem}_{dem_type}_{resolution}m.tif"
        out2 = dir2 / f"{Path(self.pc2.filename).stem}_{dem_type}_{resolution}m.tif"
        
        if verbose:
            print(f"\nCreating DEM from pc1: {Path(pc1_source.filename).name}")
        
        dem1 = pc1_source.create_dem(
            output_path=str(out1),
            dem_type=dem_type,
            resolution=resolution,
            interpolation=interpolation,
            **dem_kwargs,
        )
        
        if verbose:
            print(f"Creating DEM from pc2: {Path(self.pc2.filename).name}")
        
        dem2 = self.pc2.create_dem(
            output_path=str(out2),
            dem_type=dem_type,
            resolution=resolution,
            interpolation=interpolation,
            **dem_kwargs,
        )
        
        # Copy epoch and CRS info to DEMs
        dem1.epoch = getattr(pc1_source, 'epoch', None)
        dem2.epoch = getattr(self.pc2, 'epoch', None)
        
        if verbose:
            print(f"\nDEM1: {out1}")
            print(f"DEM2: {out2}")
            print(f"{'=' * 60}\n")
        
        return dem1, dem2
    
    def create_dtm_pair(self, **kwargs) -> Tuple[Raster, Raster]:
        """Create DTM (bare earth) pair. Convenience wrapper for create_dem_pair()."""
        return self.create_dem_pair(dem_type="dtm", **kwargs)
    
    def create_dsm_pair(self, **kwargs) -> Tuple[Raster, Raster]:
        """Create DSM (surface) pair. Convenience wrapper for create_dem_pair()."""
        return self.create_dem_pair(dem_type="dsm", **kwargs)
    
    # =========================================================================
    # 3D Differencing Methods
    # =========================================================================
    
    def compute_3d_difference(
        self,
        max_distance: float = 1.0,
        use_transformed: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute point-to-point 3D differences between point clouds.
        
        For each point in pc1, finds the nearest point in pc2 and computes
        the signed vertical (Z) difference.
        
        Parameters
        ----------
        max_distance : float
            Maximum 3D distance for valid correspondences (meters)
        use_transformed : bool
            If True, use the transformed/aligned pc1
        verbose : bool
            Print progress messages
            
        Returns
        -------
        dict
            Results including:
            - differences: array of Z differences for each pc1 point
            - distances_3d: 3D distances to nearest pc2 point
            - valid_mask: boolean mask for points within max_distance
            - statistics: dict with mean, std, median, etc.
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("Computing 3D Point Cloud Difference")
            print(f"{'=' * 60}")
        
        # Select source for pc1
        pc1_source = self._pc1_transformed if use_transformed and self._pc1_transformed else self.pc1
        
        # Load points
        if verbose:
            print(f"Loading points...")
        
        points1 = _load_points_from_las(pc1_source.filename)
        points2 = _load_points_from_las(self.pc2.filename)
        
        if verbose:
            print(f"PC1 points: {len(points1):,}")
            print(f"PC2 points: {len(points2):,}")
        
        # Build KD-tree on pc2 (reference)
        from scipy.spatial import cKDTree
        
        if verbose:
            print(f"Building KD-tree...")
        
        tree2 = cKDTree(points2)
        
        # Find nearest neighbors
        if verbose:
            print(f"Finding correspondences...")
        
        distances_3d, indices = tree2.query(points1, k=1)
        
        # Compute Z differences (pc2 - pc1, positive = gain)
        z1 = points1[:, 2]
        z2_nearest = points2[indices, 2]
        z_differences = z2_nearest - z1
        
        # Apply distance filter
        valid_mask = distances_3d <= max_distance
        valid_differences = z_differences[valid_mask]
        
        # Compute statistics
        if len(valid_differences) > 0:
            statistics = {
                'count': len(valid_differences),
                'mean': float(np.mean(valid_differences)),
                'std': float(np.std(valid_differences)),
                'median': float(np.median(valid_differences)),
                'min': float(np.min(valid_differences)),
                'max': float(np.max(valid_differences)),
                'q25': float(np.percentile(valid_differences, 25)),
                'q75': float(np.percentile(valid_differences, 75)),
                'iqr': float(np.percentile(valid_differences, 75) - np.percentile(valid_differences, 25)),
                'nmad': float(1.4826 * np.median(np.abs(valid_differences - np.median(valid_differences)))),
                'valid_ratio': float(np.sum(valid_mask) / len(valid_mask)),
            }
        else:
            statistics = {
                'count': 0,
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan,
                'min': np.nan,
                'max': np.nan,
                'valid_ratio': 0.0,
            }
        
        if verbose:
            print(f"\n3D Difference Statistics:")
            print(f"  Valid points: {statistics['count']:,} ({statistics['valid_ratio']:.1%})")
            print(f"  Mean: {statistics['mean']:.4f} m")
            print(f"  Std: {statistics['std']:.4f} m")
            print(f"  Median: {statistics['median']:.4f} m")
            print(f"  NMAD: {statistics.get('nmad', np.nan):.4f} m")
            print(f"  Range: [{statistics['min']:.4f}, {statistics['max']:.4f}] m")
            print(f"{'=' * 60}\n")
        
        return {
            'differences': z_differences,
            'distances_3d': distances_3d,
            'valid_mask': valid_mask,
            'statistics': statistics,
            'max_distance': max_distance,
        }
    
    # =========================================================================
    # 2D Differencing Methods (via RasterPair)
    # =========================================================================
    
    def compute_2d_difference(
        self,
        dem_type: str = "dtm",
        resolution: float = 1.0,
        interpolation: str = "idw",
        transform_first: bool = True,
        use_transformed: bool = True,
        output_dir: Optional[str] = None,
        overwrite: bool = True,
        verbose: bool = True,
        **dem_kwargs,
    ) -> Dict[str, Any]:
        """
        Compute 2D (raster-based) elevation difference.
        
        This method:
        1. Creates DEMs from both point clouds
        2. Creates a RasterPair
        3. Uses RasterPair.compute_difference() for differencing
        
        Parameters
        ----------
        dem_type : str, {"dtm", "dsm"}
            Type of DEM to create
        resolution : float
            DEM resolution in map units
        interpolation : str
            DEM interpolation method
        transform_first : bool
            Transform compare DEM to match reference before differencing
        use_transformed : bool
            Use transformed pc1 for DEM creation
        output_dir : str, optional
            Directory for output files
        overwrite : bool
            Overwrite existing files
        verbose : bool
            Print progress messages
        **dem_kwargs
            Additional arguments for DEM creation
            
        Returns
        -------
        dict
            Results from RasterPair.compute_difference()
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("Computing 2D (DEM-based) Difference")
            print(f"{'=' * 60}")
        
        # Create DEMs
        dem1, dem2 = self.create_dem_pair(
            dem_type=dem_type,
            resolution=resolution,
            interpolation=interpolation,
            use_transformed=use_transformed,
            output_dir=output_dir,
            overwrite=overwrite,
            verbose=verbose,
            **dem_kwargs,
        )
        
        # Create RasterPair (dem1 = compare, dem2 = reference)
        raster_pair = RasterPair(dem1, dem2)
        
        if verbose:
            print("\nRasterPair comparison:")
            raster_pair.print_summary()
        
        # Compute difference using RasterPair
        result = raster_pair.compute_difference(
            transform_first=transform_first,
            interpolation_method="bilinear",
            clip_to_overlap=True,
            overwrite=overwrite,
            verbose=verbose,
        )
        
        # Add point cloud context to result
        result['dem_type'] = dem_type
        result['dem_resolution'] = resolution
        result['dem_interpolation'] = interpolation
        result['pc1_file'] = self.pc1.filename
        result['pc2_file'] = self.pc2.filename
        
        return result
    
    def compute_dtm_difference(self, **kwargs) -> Dict[str, Any]:
        """Compute DTM-based difference. Convenience wrapper."""
        return self.compute_2d_difference(dem_type="dtm", **kwargs)
    
    def compute_dsm_difference(self, **kwargs) -> Dict[str, Any]:
        """Compute DSM-based difference. Convenience wrapper."""
        return self.compute_2d_difference(dem_type="dsm", **kwargs)
    
    # =========================================================================
    # Full Pipeline Methods
    # =========================================================================
    
    def full_differencing_pipeline(
        self,
        dem_type: str = "dtm",
        resolution: float = 1.0,
        interpolation: str = "idw",
        align_icp: bool = True,
        icp_method: str = "gicp",
        icp_downsample: float = 0.5,
        skip_epoch: bool = False,
        output_dir: Optional[str] = None,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full differencing pipeline.
        
        Pipeline steps:
        1. Transform pc1 to match pc2's reference frame
        2. (Optional) ICP alignment for fine registration
        3. Create DEMs
        4. Compute difference
        
        Parameters
        ----------
        dem_type : str
            Type of DEM ("dtm" or "dsm")
        resolution : float
            DEM resolution in meters
        interpolation : str
            DEM interpolation method
        align_icp : bool
            Whether to perform ICP alignment
        icp_method : str
            ICP method ("gicp", "vgicp", "icp")
        icp_downsample : float
            Downsampling resolution for ICP
        skip_epoch : bool
            Skip epoch transformation (faster, but less accurate if epochs differ significantly)
        output_dir : str, optional
            Output directory
        overwrite : bool
            Overwrite existing files
        verbose : bool
            Print progress messages
            
        Returns
        -------
        dict
            Complete results including:
            - comparison: initial comparison results
            - transformation_history: list of applied transformations
            - alignment_result: ICP alignment results (if performed)
            - difference_result: DEM differencing results
        """
        if verbose:
            print(f"\n{'#' * 60}")
            print("# Full Differencing Pipeline")
            print(f"{'#' * 60}")
        
        results = {
            'comparison': self.check_all_match(),
            'transformation_history': [],
            'alignment_result': None,
            'difference_result': None,
        }
        
        # Step 1: Transform pc1 to match pc2
        if verbose:
            print("\n[Pipeline Step 1/4] CRS/Datum Transformation")
        
        if results['comparison']['transformations_needed']:
            self.transform_compare_to_match_reference(
                skip_epoch=skip_epoch,
                overwrite=overwrite,
                verbose=verbose,
            )
            results['transformation_history'] = self._transformation_history.copy()
        else:
            if verbose:
                print("  No transformations needed - point clouds already aligned")
        
        # Step 2: ICP Alignment
        if align_icp:
            if verbose:
                print("\n[Pipeline Step 2/4] ICP Fine Alignment")
            
            if _has_small_gicp():
                alignment = self.align_point_clouds(
                    method=icp_method,
                    downsample_resolution=icp_downsample,
                    apply_transform=True,
                    overwrite=overwrite,
                    verbose=verbose,
                )
                results['alignment_result'] = alignment
            else:
                if verbose:
                    print("  Skipping ICP - small_gicp not installed")
        else:
            if verbose:
                print("\n[Pipeline Step 2/4] ICP Alignment - Skipped")
        
        # Step 3 & 4: DEM creation and differencing
        if verbose:
            print("\n[Pipeline Step 3-4/4] DEM Creation and Differencing")
        
        diff_result = self.compute_2d_difference(
            dem_type=dem_type,
            resolution=resolution,
            interpolation=interpolation,
            use_transformed=True,
            output_dir=output_dir,
            overwrite=overwrite,
            verbose=verbose,
        )
        results['difference_result'] = diff_result
        
        if verbose:
            print(f"\n{'#' * 60}")
            print("# Pipeline Complete")
            print(f"{'#' * 60}")
            
            stats = diff_result.get('statistics', {})
            print(f"\nFinal Difference Statistics:")
            print(f"  Mean: {stats.get('mean', np.nan):.4f} m")
            print(f"  Std:  {stats.get('std', np.nan):.4f} m")
            print(f"  NMAD: {stats.get('nmad', np.nan):.4f} m")
            print(f"\nDifference raster: {diff_result.get('difference_raster', {}).filename}")
            print(f"{'#' * 60}\n")
        
        return results
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_transformed_pc1(self) -> Optional[PointCloud]:
        """Get the transformed pc1, if available."""
        return self._pc1_transformed
    
    def get_alignment_result(self) -> Optional[Dict[str, Any]]:
        """Get the ICP alignment result, if available."""
        return self._alignment_result
    
    def get_transformation_history(self) -> List[Dict[str, Any]]:
        """Get the list of transformations applied."""
        return self._transformation_history.copy()
    
    def reset(self) -> None:
        """Reset internal state (clear cached transformations)."""
        self._transformation_history = []
        self._pc1_transformed = None
        self._alignment_result = None