"""CRS transformation history tracking.

Provides the CRSHistory class for tracking all CRS and coordinate transformations
applied to rasters and point clouds, including interpolation methods used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from pyproj import CRS as _CRS

from .crs_utils import _ensure_crs_obj, crs_to_wkt2_2019, crs_to_projjson

if TYPE_CHECKING:  # for type hints only
    from .pointcloud import PointCloud
    from .raster import Raster

OwnerType = Union["PointCloud", "Raster"]


# =============================================================================
# Interpolation Method Constants
# =============================================================================

# Common interpolation methods for reference
INTERPOLATION_METHODS = {
    # Raster resampling methods (rasterio/GDAL)
    'nearest': {
        'description': 'Nearest neighbor - no interpolation, preserves original values',
        'order': 0,
        'smoothing': False,
        'preserves_values': True,
    },
    'bilinear': {
        'description': 'Bilinear interpolation - weighted average of 4 nearest pixels',
        'order': 1,
        'smoothing': True,
        'preserves_values': False,
    },
    'cubic': {
        'description': 'Cubic convolution - uses 16 nearest pixels',
        'order': 3,
        'smoothing': True,
        'preserves_values': False,
    },
    'cubic_spline': {
        'description': 'Cubic spline interpolation',
        'order': 3,
        'smoothing': True,
        'preserves_values': False,
    },
    'lanczos': {
        'description': 'Lanczos windowed sinc - high quality, 36 pixels',
        'order': None,
        'smoothing': True,
        'preserves_values': False,
    },
    'average': {
        'description': 'Average of all contributing pixels',
        'order': None,
        'smoothing': True,
        'preserves_values': False,
    },
    'mode': {
        'description': 'Most frequent value - for categorical data',
        'order': None,
        'smoothing': False,
        'preserves_values': True,
    },
    # Point cloud / DEM creation methods
    'idw': {
        'description': 'Inverse Distance Weighting',
        'order': None,
        'smoothing': True,
        'preserves_values': False,
    },
    'kriging': {
        'description': 'Kriging / Gaussian Process interpolation',
        'order': None,
        'smoothing': True,
        'preserves_values': False,
    },
    'tin': {
        'description': 'Triangulated Irregular Network linear interpolation',
        'order': 1,
        'smoothing': False,
        'preserves_values': True,  # At vertices
    },
    'natural_neighbor': {
        'description': 'Natural neighbor (Sibson) interpolation',
        'order': None,
        'smoothing': True,
        'preserves_values': True,  # At data points
    },
    'spline': {
        'description': 'Spline interpolation (various types)',
        'order': None,
        'smoothing': True,
        'preserves_values': False,
    },
    # Geoid sampling
    'map_coordinates': {
        'description': 'scipy.ndimage.map_coordinates grid sampling',
        'order': 1,  # Typically bilinear
        'smoothing': True,
        'preserves_values': False,
    },
}


@dataclass
class InterpolationEntry:
    """
    Record of an interpolation operation applied to the data.
    
    This tracks resampling during transformations, grid creation,
    alignment operations, and geoid sampling.
    """
    timestamp: datetime
    operation_type: str  # 'creation', 'transformation', 'alignment', 'vertical_datum', 'geoid_sampling'
    method: str  # 'nearest', 'bilinear', 'cubic', 'idw', 'kriging', etc.
    description: str
    
    # Optional details
    source_file: Optional[str] = None
    target_file: Optional[str] = None
    
    # Method-specific parameters (e.g., power for IDW, variogram for kriging)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Grid/resolution info
    source_resolution: Optional[float] = None
    target_resolution: Optional[float] = None
    
    # Quality indicators
    pixel_shift_magnitude: Optional[float] = None  # Max shift in pixels
    resampling_skipped: bool = False  # True if shift was sub-pixel and skipped
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'operation_type': self.operation_type,
            'method': self.method,
            'description': self.description,
            'source_file': self.source_file,
            'target_file': self.target_file,
            'parameters': self.parameters,
            'source_resolution': self.source_resolution,
            'target_resolution': self.target_resolution,
            'pixel_shift_magnitude': self.pixel_shift_magnitude,
            'resampling_skipped': self.resampling_skipped,
        }


@dataclass
class CRSEntry:
    """
    One CRS or raster-related event in the history.
    """
    timestamp: datetime
    entry_type: str  # "initial", "manual_change", "transformation", "raster_creation"
    description: str

    # CRS state
    compound_crs_proj: Optional[_CRS] = None
    horizontal_crs_proj: Optional[_CRS] = None
    vertical_crs_proj: Optional[_CRS] = None

    compound_crs_wkt: Optional[str] = None
    compound_crs_wkt2: Optional[str] = None
    compound_crs_epsg: Optional[int] = None

    horizontal_crs_wkt: Optional[str] = None
    horizontal_crs_wkt2: Optional[str] = None
    horizontal_crs_epsg: Optional[int] = None

    vertical_crs_wkt: Optional[str] = None
    vertical_crs_wkt2: Optional[str] = None
    vertical_crs_epsg: Optional[int] = None

    geoid_model: Optional[str] = None
    epoch: Optional[float] = None  # decimal year
    
    # Interpolation method used for this operation (if applicable)
    interpolation_method: Optional[str] = None

    # Any additional info (e.g., raster creation params, alignment params, etc.)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CRSHistory:
    """
    Track CRS / geoid / epoch changes, raster creation steps, and interpolation
    methods for a PointCloud or Raster object.

    This is intended to reflect the structure described in pseudocode.txt:
    - Original compound / horizontal / vertical CRS in WKT / WKT2 / EPSG
    - Current CRS variants
    - History list describing changes and transformations
    - Interpolation history tracking resampling operations
    """

    owner: OwnerType
    history: List[CRSEntry] = field(default_factory=list)
    
    # Interpolation tracking (Raster-specific, but available for all)
    interpolation_history: List[InterpolationEntry] = field(default_factory=list)

    # New: explicit linkage fields
    derived_from: Optional[str] = None
    derived_products: List[Dict[str, Any]] = field(default_factory=list)

    # Original state
    original_compound_crs_proj: Optional[_CRS] = None
    original_compound_crs_wkt: Optional[str] = None
    original_compound_crs_wkt2: Optional[str] = None
    original_compound_crs_epsg: Optional[int] = None

    original_horizontal_crs_proj: Optional[_CRS] = None
    original_horizontal_crs_wkt: Optional[str] = None
    original_horizontal_crs_wkt2: Optional[str] = None
    original_horizontal_crs_epsg: Optional[int] = None

    original_vertical_crs_proj: Optional[_CRS] = None
    original_vertical_crs_wkt: Optional[str] = None
    original_vertical_crs_wkt2: Optional[str] = None
    original_vertical_crs_epsg: Optional[int] = None

    original_geoid_model: Optional[str] = None
    original_epoch: Optional[float] = None  # decimal year

    # Current state
    current_compound_crs_proj: Optional[_CRS] = None
    current_compound_crs_wkt: Optional[str] = None
    current_compound_crs_wkt2: Optional[str] = None
    current_compound_crs_epsg: Optional[int] = None

    current_horizontal_crs_proj: Optional[_CRS] = None
    current_horizontal_crs_wkt: Optional[str] = None
    current_horizontal_crs_wkt2: Optional[str] = None
    current_horizontal_crs_epsg: Optional[int] = None

    current_vertical_crs_proj: Optional[_CRS] = None
    current_vertical_crs_wkt: Optional[str] = None
    current_vertical_crs_wkt2: Optional[str] = None
    current_vertical_crs_epsg: Optional[int] = None

    current_geoid_model: Optional[str] = None
    current_epoch: Optional[float] = None

    def __post_init__(self) -> None:
        """
        Initialize original and current CRS information from the owner.
        """
        # Compound CRS
        owner_compound = getattr(self.owner, "current_compound_crs", None) or getattr(
            self.owner, "original_compound_crs", None
        )
        if owner_compound is not None:
            try:
                comp_crs = _ensure_crs_obj(owner_compound)
            except Exception:
                comp_crs = None
        else:
            comp_crs = None

        # Horizontal CRS
        owner_horizontal = getattr(self.owner, "current_horizontal_crs", None) or getattr(
            self.owner, "original_horizontal_crs", None
        )
        if owner_horizontal is not None:
            try:
                horiz_crs = _ensure_crs_obj(owner_horizontal)
            except Exception:
                horiz_crs = None
        else:
            horiz_crs = None

        # Vertical CRS
        owner_vertical = getattr(self.owner, "current_vertical_crs", None) or getattr(
            self.owner, "original_vertical_crs", None
        )
        if owner_vertical is not None:
            try:
                vert_crs = _ensure_crs_obj(owner_vertical)
            except Exception:
                vert_crs = None
        else:
            vert_crs = None

        epoch = getattr(self.owner, "epoch", None)
        geoid_model = getattr(self.owner, "geoid_model", None)

        # Fill original
        self._set_original(comp_crs, horiz_crs, vert_crs, geoid_model, epoch)
        # Initialize current to original
        self._set_current(comp_crs, horiz_crs, vert_crs, geoid_model, epoch)

        # Record initial entry
        self.history.append(
            self._make_entry(
                entry_type="initial",
                description="Initial CRS/geoid/epoch from source dataset.",
            )
        )

    # ----- internal helpers -----

    def _set_original(
        self,
        compound: Optional[_CRS],
        horizontal: Optional[_CRS],
        vertical: Optional[_CRS],
        geoid_model: Optional[str],
        epoch: Optional[float],
    ) -> None:
        self.original_compound_crs_proj = compound
        self.original_horizontal_crs_proj = horizontal
        self.original_vertical_crs_proj = vertical
        self.original_geoid_model = geoid_model
        self.original_epoch = epoch

        if compound is not None:
            self.original_compound_crs_wkt = compound.to_wkt()
            self.original_compound_crs_wkt2 = crs_to_wkt2_2019(compound, pretty=False)
            self.original_compound_crs_epsg = compound.to_epsg()
        if horizontal is not None:
            self.original_horizontal_crs_wkt = horizontal.to_wkt()
            self.original_horizontal_crs_wkt2 = crs_to_wkt2_2019(horizontal, pretty=False)
            self.original_horizontal_crs_epsg = horizontal.to_epsg()
        if vertical is not None:
            self.original_vertical_crs_wkt = vertical.to_wkt()
            self.original_vertical_crs_wkt2 = crs_to_wkt2_2019(vertical, pretty=False)
            self.original_vertical_crs_epsg = vertical.to_epsg()

    def _set_current(
        self,
        compound: Optional[_CRS],
        horizontal: Optional[_CRS],
        vertical: Optional[_CRS],
        geoid_model: Optional[str],
        epoch: Optional[float],
    ) -> None:
        self.current_compound_crs_proj = compound
        self.current_horizontal_crs_proj = horizontal
        self.current_vertical_crs_proj = vertical
        self.current_geoid_model = geoid_model
        self.current_epoch = epoch

        if compound is not None:
            self.current_compound_crs_wkt = compound.to_wkt()
            self.current_compound_crs_wkt2 = crs_to_wkt2_2019(compound, pretty=False)
            self.current_compound_crs_epsg = compound.to_epsg()
        if horizontal is not None:
            self.current_horizontal_crs_wkt = horizontal.to_wkt()
            self.current_horizontal_crs_wkt2 = crs_to_wkt2_2019(horizontal, pretty=False)
            self.current_horizontal_crs_epsg = horizontal.to_epsg()
        if vertical is not None:
            self.current_vertical_crs_wkt = vertical.to_wkt()
            self.current_vertical_crs_wkt2 = crs_to_wkt2_2019(vertical, pretty=False)
            self.current_vertical_crs_epsg = vertical.to_epsg()

    def _make_entry(
        self,
        entry_type: str,
        description: str,
        extra: Optional[Dict[str, Any]] = None,
        interpolation_method: Optional[str] = None,
    ) -> CRSEntry:
        """
        Create a CRSEntry using the current state.
        """
        comp = self.current_compound_crs_proj
        horiz = self.current_horizontal_crs_proj
        vert = self.current_vertical_crs_proj

        return CRSEntry(
            timestamp=datetime.utcnow(),
            entry_type=entry_type,
            description=description,
            compound_crs_proj=comp,
            horizontal_crs_proj=horiz,
            vertical_crs_proj=vert,
            compound_crs_wkt=self.current_compound_crs_wkt,
            compound_crs_wkt2=self.current_compound_crs_wkt2,
            compound_crs_epsg=self.current_compound_crs_epsg,
            horizontal_crs_wkt=self.current_horizontal_crs_wkt,
            horizontal_crs_wkt2=self.current_horizontal_crs_wkt2,
            horizontal_crs_epsg=self.current_horizontal_crs_epsg,
            vertical_crs_wkt=self.current_vertical_crs_wkt,
            vertical_crs_wkt2=self.current_vertical_crs_wkt2,
            vertical_crs_epsg=self.current_vertical_crs_epsg,
            geoid_model=self.current_geoid_model,
            epoch=self.current_epoch,
            interpolation_method=interpolation_method,
            extra=extra or {},
        )

    # ----- public methods mirroring pseudocode -----

    def add_manual_change_entry(
        self,
        new_compound_crs_proj: Optional[Union[str, _CRS, Dict[str, Any]]] = None,
        new_horizontal_crs_proj: Optional[Union[str, _CRS, Dict[str, Any]]] = None,
        new_vertical_crs_proj: Optional[Union[str, _CRS, Dict[str, Any]]] = None,
        geoid_model: Optional[str] = None,
        epoch: Optional[float] = None,
        note: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """
        Record a user-driven manual CRS/geoid/epoch change.
        """
        if new_compound_crs_proj is not None:
            self.current_compound_crs_proj = _ensure_crs_obj(new_compound_crs_proj)
        if new_horizontal_crs_proj is not None:
            self.current_horizontal_crs_proj = _ensure_crs_obj(new_horizontal_crs_proj)
        if new_vertical_crs_proj is not None:
            self.current_vertical_crs_proj = _ensure_crs_obj(new_vertical_crs_proj)

        if geoid_model is not None:
            self.current_geoid_model = geoid_model
        if epoch is not None:
            self.current_epoch = float(epoch)

        # Refresh WKT/WKT2/EPSG for new current state
        self._set_current(
            self.current_compound_crs_proj,
            self.current_horizontal_crs_proj,
            self.current_vertical_crs_proj,
            self.current_geoid_model,
            self.current_epoch,
        )

        desc = note or "Manual CRS/geoid/epoch change."
        self.history.append(self._make_entry("manual_change", desc, extra=extra))

    def record_transformation_entry(
        self,
        transformation_type: str,
        source_crs_proj: Optional[Union[str, _CRS, Dict[str, Any]]],
        target_crs_proj: Optional[Union[str, _CRS, Dict[str, Any]]],
        method: Optional[str] = None,
        transformation_matrix: Optional[Any] = None,
        alignment_origin: Optional[str] = None,
        interpolation_method: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """
        Record a CRS or alignment transformation.
        
        Parameters
        ----------
        transformation_type : str
            Type of transformation (e.g., 'horizontal_reprojection', 'dynamic_epoch', 
            'vertical_datum', 'grid_alignment')
        source_crs_proj : CRS-like
            Source CRS
        target_crs_proj : CRS-like
            Target CRS
        method : str, optional
            Transformation method used
        transformation_matrix : any, optional
            Transformation matrix if applicable
        alignment_origin : str, optional
            Origin point for alignment
        interpolation_method : str, optional
            Resampling method used (e.g., 'bilinear', 'nearest', 'cubic')
        **extra
            Additional parameters to record
        """
        src = _ensure_crs_obj(source_crs_proj) if source_crs_proj is not None else None
        tgt = _ensure_crs_obj(target_crs_proj) if target_crs_proj is not None else None

        # Update current CRS to target if present
        if tgt is not None:
            self.current_compound_crs_proj = tgt
            self._set_current(
                tgt,
                self.current_horizontal_crs_proj,
                self.current_vertical_crs_proj,
                self.current_geoid_model,
                self.current_epoch,
            )

        info: Dict[str, Any] = {
            "transformation_type": transformation_type,
            "source_crs_proj": crs_to_projjson(src) if src is not None else None,
            "target_crs_proj": crs_to_projjson(tgt) if tgt is not None else None,
            "method": method,
            "alignment_origin": alignment_origin,
            "transformation_matrix": transformation_matrix,
            "interpolation_method": interpolation_method,
        }
        info.update(extra)

        # Keep an explicit list of derived products for this owner
        target_file = info.get("target_file")
        if target_file:
            self.derived_products.append(
                {
                    "path": target_file,
                    "transformation_type": transformation_type,
                    "method": method,
                    "interpolation_method": interpolation_method,
                }
            )

        desc = f"CRS/geometry transformation ({transformation_type})."
        self.history.append(self._make_entry(
            "transformation", desc, extra=info, interpolation_method=interpolation_method
        ))
        
        # Also record in interpolation history if method was specified
        if interpolation_method:
            self.record_interpolation_entry(
                operation_type=transformation_type,
                method=interpolation_method,
                description=f"Resampling during {transformation_type}",
                source_file=info.get("source_file"),
                target_file=target_file,
            )

    def record_raster_creation_entry(
        self,
        creation_parameters: Dict[str, Any],
        description: Optional[str] = None,
        interpolation_method: Optional[str] = None,
    ) -> None:
        """
        Record raster creation parameters (DEM type, interpolation, etc.)
        
        Parameters
        ----------
        creation_parameters : dict
            Parameters used to create the raster
        description : str, optional
            Description of the creation process
        interpolation_method : str, optional
            Interpolation method used during creation (e.g., 'idw', 'kriging', 'tin')
        """
        desc = description or "Raster creation / post-processing."
        extra = {"creation_parameters": creation_parameters}
        
        # Include interpolation method in creation parameters if provided
        if interpolation_method:
            creation_parameters["interpolation_method"] = interpolation_method
        
        self.history.append(self._make_entry(
            "raster_creation", desc, extra=extra, interpolation_method=interpolation_method
        ))
        
        # Record in interpolation history if method was specified
        if interpolation_method:
            self.record_interpolation_entry(
                operation_type="creation",
                method=interpolation_method,
                description=f"Raster creation using {interpolation_method}",
                parameters=creation_parameters,
            )

    # ----- Interpolation tracking methods -----

    def record_interpolation_entry(
        self,
        operation_type: str,
        method: str,
        description: Optional[str] = None,
        source_file: Optional[str] = None,
        target_file: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        source_resolution: Optional[float] = None,
        target_resolution: Optional[float] = None,
        pixel_shift_magnitude: Optional[float] = None,
        resampling_skipped: bool = False,
    ) -> None:
        """
        Record an interpolation/resampling operation.
        
        This tracks resampling that occurs during:
        - Raster creation from point clouds
        - CRS transformations (reprojection)
        - Epoch transformations
        - Grid alignment
        - Vertical datum transformations (geoid sampling)
        
        Parameters
        ----------
        operation_type : str
            Type of operation: 'creation', 'transformation', 'alignment', 
            'vertical_datum', 'geoid_sampling', 'epoch_transform'
        method : str
            Interpolation method: 'nearest', 'bilinear', 'cubic', 'idw', 
            'kriging', 'tin', 'natural_neighbor', etc.
        description : str, optional
            Human-readable description
        source_file : str, optional
            Source file path
        target_file : str, optional
            Target/output file path
        parameters : dict, optional
            Method-specific parameters (e.g., {'power': 2} for IDW)
        source_resolution : float, optional
            Source grid resolution
        target_resolution : float, optional
            Target grid resolution
        pixel_shift_magnitude : float, optional
            Maximum pixel shift (useful for detecting if resampling was needed)
        resampling_skipped : bool
            True if resampling was skipped due to sub-pixel shift
        """
        if description is None:
            description = f"{operation_type.replace('_', ' ').title()} using {method}"
        
        entry = InterpolationEntry(
            timestamp=datetime.utcnow(),
            operation_type=operation_type,
            method=method,
            description=description,
            source_file=source_file,
            target_file=target_file,
            parameters=parameters or {},
            source_resolution=source_resolution,
            target_resolution=target_resolution,
            pixel_shift_magnitude=pixel_shift_magnitude,
            resampling_skipped=resampling_skipped,
        )
        
        self.interpolation_history.append(entry)

    def get_interpolation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all interpolation operations applied to this raster.
        
        Returns
        -------
        dict
            Summary including:
            - count: Total number of interpolation operations
            - methods_used: List of unique methods used
            - operations: List of operation types
            - any_resampling_skipped: Whether any operations were skipped
            - details: List of all interpolation entries
        """
        if not self.interpolation_history:
            return {
                'count': 0,
                'methods_used': [],
                'operations': [],
                'any_resampling_skipped': False,
                'details': [],
            }
        
        methods = list(set(e.method for e in self.interpolation_history))
        operations = list(set(e.operation_type for e in self.interpolation_history))
        any_skipped = any(e.resampling_skipped for e in self.interpolation_history)
        
        return {
            'count': len(self.interpolation_history),
            'methods_used': methods,
            'operations': operations,
            'any_resampling_skipped': any_skipped,
            'details': [e.to_dict() for e in self.interpolation_history],
        }

    def get_interpolation_chain(self) -> List[str]:
        """
        Get a simple list of interpolation methods in order of application.
        
        Useful for understanding cumulative smoothing/error effects.
        
        Returns
        -------
        list of str
            Methods in chronological order (e.g., ['bilinear', 'bilinear', 'nearest'])
        """
        return [e.method for e in self.interpolation_history if not e.resampling_skipped]

    # ----- Convenience helpers -----

    def to_dict(self) -> Dict[str, Any]:
        """
        Compact JSON-like representation of the CRS history.

        - Keeps original/current EPSG + geoid + epoch.
        - History entries have a trimmed 'extra' section (no giant PROJJSON).
        - Adds file-link information:
            * 'file': the owner's filename
            * 'derived_from': original file this one was transformed from (if any)
            * 'derived_products': list of transformed outputs originating here
        - Includes interpolation history summary
        """
        owner_file = getattr(self.owner, "filename", None)

        # Determine derived_products for this object
        if self.derived_products:
            derived_products = list(self.derived_products)
        else:
            # Backward-compatible: collect children (derived products) from history
            derived_products = []
            for e in self.history:
                if e.entry_type == "transformation":
                    target_file = e.extra.get("target_file")
                    if target_file:
                        derived_products.append(
                            {
                                "path": target_file,
                                "transformation_type": e.extra.get("transformation_type"),
                                "method": e.extra.get("method"),
                                "interpolation_method": e.extra.get("interpolation_method"),
                            }
                        )

        # Determine parent file for this object, if any
        if self.derived_from is not None:
            parent_file: Optional[str] = self.derived_from
        else:
            parent_file = None
            for e in self.history:
                if e.entry_type == "transformation":
                    src_file = e.extra.get("source_file")
                    if src_file and src_file != owner_file:
                        parent_file = src_file
                        break

        def _compact_extra(e: CRSEntry) -> Dict[str, Any]:
            """
            Trim the extra dict to the most useful bits and avoid spewing
            full PROJJSON structures.
            """
            keep_keys = (
                "transformation_type",
                "method",
                "source_file",
                "target_file",
                "src_epoch",
                "dst_epoch",
                "source_kind",
                "target_kind",
                "interpolation_method",
            )
            return {k: e.extra[k] for k in keep_keys if k in e.extra}

        return {
            "file": owner_file,
            "derived_from": parent_file,
            "derived_products": derived_products,
            "original": {
                "compound_crs_epsg": self.original_compound_crs_epsg,
                "horizontal_crs_epsg": self.original_horizontal_crs_epsg,
                "vertical_crs_epsg": self.original_vertical_crs_epsg,
                "geoid_model": self.original_geoid_model,
                "epoch": self.original_epoch,
            },
            "current": {
                "compound_crs_epsg": self.current_compound_crs_epsg,
                "horizontal_crs_epsg": self.current_horizontal_crs_epsg,
                "vertical_crs_epsg": self.current_vertical_crs_epsg,
                "geoid_model": self.current_geoid_model,
                "epoch": self.current_epoch,
            },
            "history": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "entry_type": e.entry_type,
                    "description": e.description,
                    "compound_crs_epsg": e.compound_crs_epsg,
                    "horizontal_crs_epsg": e.horizontal_crs_epsg,
                    "vertical_crs_epsg": e.vertical_crs_epsg,
                    "geoid_model": e.geoid_model,
                    "epoch": e.epoch,
                    "interpolation_method": e.interpolation_method,
                    "extra": _compact_extra(e),
                }
                for e in self.history
            ],
            "interpolation": self.get_interpolation_summary(),
        }