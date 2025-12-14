# velocity_model_registry.py
"""
Enhanced velocity/deformation model registry and selection system.

Provides:
  1. Loading models from YAML registry
  2. Spatial and temporal filtering
  3. Auto-download capability
  4. User custom model support
  5. Integration with PROJ pipelines
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal
import warnings

try:
    import yaml
except ImportError:
    yaml = None

from pyproj import datadir


# ============================================================================
# Core data structures
# ============================================================================

@dataclass
class VelocityModelInfo:
    """
    Metadata about a deformation / velocity model.
    
    Attributes:
        name: Unique identifier
        label: Human-readable description
        region: Geographic region
        bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
        epoch_min: Earliest valid epoch (decimal year)
        epoch_max: Latest valid epoch (decimal year)
        central_epoch: Reference epoch (if applicable)
        kind: Type of model ('velocity', 'deformation', 'strain_rate')
        format: File format description
        proj_op: PROJ operation ('deformation', 'defmodel', or None)
        files: List of required file patterns
        download: Download configuration dict
        auto_download: Whether to auto-download if missing
        license: License information
        citations: List of citations
        notes: Additional notes
        local: Whether files are available locally
        filepath: Resolved local filepath (set after resolution)
    """
    name: str
    label: str
    region: str
    bbox: Tuple[float, float, float, float]
    epoch_min: float
    epoch_max: float
    kind: str
    format: str
    central_epoch: Optional[float] = None
    proj_op: Optional[str] = None
    files: Optional[List[str]] = None
    download: Dict[str, Any] = field(default_factory=dict)
    auto_download: bool = False
    license: Optional[str] = None
    citations: Optional[List[str]] = None
    notes: Optional[str] = None
    local: bool = False
    filepath: Optional[str] = None

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "VelocityModelInfo":
        """Create VelocityModelInfo from registry dict."""
        bbox = tuple(data["bbox"])
        if len(bbox) != 4:
            raise ValueError(f"bbox must have 4 values, got {len(bbox)}")
        
        return cls(
            name=data.get("name", name),
            label=data["label"],
            region=data["region"],
            bbox=bbox,
            epoch_min=float(data["epoch_min"]),
            epoch_max=float(data["epoch_max"]),
            central_epoch=float(data["central_epoch"]) if data.get("central_epoch") else None,
            kind=data["kind"],
            format=data["format"],
            proj_op=data.get("proj_op"),
            files=data.get("files"),
            download=data.get("download", {}),
            auto_download=bool(data.get("auto_download", False)),
            license=data.get("license"),
            citations=data.get("citations"),
            notes=data.get("notes"),
        )


# ============================================================================
# Registry loading
# ============================================================================

def load_registry(
    registry_path: Optional[Path] = None,
    include_defaults: bool = True,
) -> List[VelocityModelInfo]:
    """
    Load velocity model registry from YAML file.
    
    Args:
        registry_path: Path to YAML registry file. If None, uses default.
        include_defaults: Whether to include built-in default models.
    
    Returns:
        List of VelocityModelInfo objects
    """
    models = []
    
    # Load from YAML if available
    if registry_path or include_defaults:
        if registry_path is None:
            # Look for default registry in common locations
            search_paths = [
                Path(__file__).parent / "velocity_models_registry.yaml",
                Path.home() / ".config" / "proj" / "velocity_models_registry.yaml",
                Path("/etc/proj/velocity_models_registry.yaml"),
            ]
            for p in search_paths:
                if p.exists():
                    registry_path = p
                    break
        
        if registry_path and Path(registry_path).exists():
            if yaml is None:
                warnings.warn("PyYAML not installed, cannot load YAML registry")
            else:
                with open(registry_path) as f:
                    data = yaml.safe_load(f)
                    
                if "models" in data:
                    for model_name, model_data in data["models"].items():
                        try:
                            model = VelocityModelInfo.from_dict(model_name, model_data)
                            models.append(model)
                        except Exception as e:
                            warnings.warn(f"Failed to load model '{model_name}': {e}")
    
    # Add built-in defaults if no registry loaded or requested
    if include_defaults and not models:
        models.extend(_get_builtin_models())
    
    return models


def _get_builtin_models() -> List[VelocityModelInfo]:
    """
    Return built-in default models (fallback if registry not available).
    """
    return [
        VelocityModelInfo(
            name="nad83csrs_v7",
            label="NAD83(CSRS) v7 velocity grid",
            region="Canada",
            bbox=(-141.01, 41.67, -52.54, 83.17),
            epoch_min=1990.0,
            epoch_max=2050.0,
            central_epoch=2010.0,
            kind="velocity",
            format="GVB / GeoTIFF",
            proj_op="deformation",
            files=["NAD83v70VG.gvb", "ca_nrc_NAD83v70VG.tif"],
            download={
                "method": "cdn_direct",
                "urls": ["https://cdn.proj.org/ca_nrc_NAD83v70VG.tif"],
            },
            auto_download=True,
        ),
        VelocityModelInfo(
            name="nkg_rf17vel",
            label="NKG RF17vel Nordic velocity model",
            region="Nordic and Baltic",
            bbox=(5.0, 54.0, 32.0, 72.0),
            epoch_min=1980.0,
            epoch_max=2050.0,
            central_epoch=2000.0,
            kind="velocity",
            format="GeoTIFF",
            proj_op="deformation",
            files=["eur_nkg_nkgrf17vel.tif"],
            download={
                "method": "cdn_direct",
                "urls": ["https://cdn.proj.org/eur_nkg_nkgrf17vel.tif"],
            },
            auto_download=True,
        ),
        VelocityModelInfo(
            name="nzgd2000_defmodel",
            label="NZGD2000 deformation model",
            region="New Zealand",
            bbox=(165.0, -48.0, 180.0, -33.0),
            epoch_min=1900.0,
            epoch_max=2050.0,
            central_epoch=2000.0,
            kind="deformation",
            format="JSON + CSV",
            proj_op="defmodel",
            files=["nzgd2000-20180701.json"],
            download={
                "method": "http_scrape",
                "base_url": "https://www.geodesy.linz.govt.nz/download/",
                "pattern": "nzgd2000_deformation_model.*\\.zip",
            },
            auto_download=True,
        ),
    ]


# ============================================================================
# File resolution and download
# ============================================================================

def _proj_data_dirs() -> List[Path]:
    """Find PROJ resource directories."""
    dirs = set()
    
    dd = datadir.get_data_dir()
    if dd and os.path.isdir(dd):
        dirs.add(Path(dd))
    
    try:
        ud = datadir.get_user_data_dir()
        if ud and os.path.isdir(ud):
            dirs.add(Path(ud))
    except Exception:
        pass
    
    for env in os.environ.get("PROJ_LIB", "").split(os.pathsep):
        if env and os.path.isdir(env):
            dirs.add(Path(env))
    
    return sorted(dirs)


def find_model_file(
    model: VelocityModelInfo,
    auto_download: Optional[bool] = None,
    verbose: bool = False,
) -> Optional[str]:
    """
    Find model file in PROJ data directories or download if needed.
    
    Args:
        model: VelocityModelInfo to find
        auto_download: Override model's auto_download setting
        verbose: Print status messages
    
    Returns:
        Path to model file, or None if not found
    """
    if auto_download is None:
        auto_download = model.auto_download
    
    if not model.files:
        if verbose:
            print(f"No files specified for model '{model.name}'")
        return None
    
    # Search PROJ data directories
    proj_dirs = _proj_data_dirs()
    
    for file_pattern in model.files:
        # Check if it's a glob pattern or exact filename
        is_pattern = any(c in file_pattern for c in ['*', '?', '[', ']'])
        
        for base_dir in proj_dirs:
            if is_pattern:
                # Use glob to find matching files
                matches = list(base_dir.glob(file_pattern))
                if matches:
                    filepath = str(matches[0])
                    if verbose:
                        print(f"Found model file: {filepath}")
                    model.local = True
                    model.filepath = filepath
                    return filepath
            else:
                # Exact filename
                candidate = base_dir / file_pattern
                if candidate.exists():
                    filepath = str(candidate)
                    if verbose:
                        print(f"Found model file: {filepath}")
                    model.local = True
                    model.filepath = filepath
                    return filepath
    
    # Not found locally - try to download if allowed
    if auto_download and model.download:
        if verbose:
            print(f"Model '{model.name}' not found locally, attempting download...")
        
        downloaded = download_and_convert_model(model, verbose=verbose)
        if downloaded:
            return downloaded
    
    if verbose:
        print(f"Model '{model.name}' not available")
    
    return None


def download_model(
    model: VelocityModelInfo,
    verbose: bool = False,
    auto_convert: bool = True,
) -> Optional[str]:
    """
    Download model files based on download configuration.
    
    Args:
        model: VelocityModelInfo with download configuration
        verbose: Print download progress
        auto_convert: Automatically convert to PROJ-compatible format
    
    Returns:
        Path to downloaded file (possibly converted), or None if failed
    """
    method = model.download.get("method")
    
    if not method:
        if verbose:
            print(f"No download method specified for '{model.name}'")
        return None
    
    # Get target directory (prefer user data dir)
    target_dir = None
    try:
        ud = datadir.get_user_data_dir()
        if ud:
            target_dir = Path(ud)
            target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    
    if not target_dir:
        # Fallback to first writable PROJ dir
        for pd in _proj_data_dirs():
            if os.access(pd, os.W_OK):
                target_dir = pd
                break
    
    if not target_dir:
        if verbose:
            print("No writable PROJ data directory found")
        return None
    
    if method == "cdn_direct":
        urls = model.download.get("urls", [])
        if not urls:
            if verbose:
                print(f"No URLs specified for '{model.name}'")
            return None
        
        # Try each URL
        for url in urls:
            try:
                filename = url.split("/")[-1]
                output_path = target_dir / filename
                
                if verbose:
                    print(f"Downloading {url} -> {output_path}")
                
                urllib.request.urlretrieve(url, output_path)
                
                model.local = True
                model.filepath = str(output_path)
                
                if verbose:
                    print(f"Successfully downloaded to {output_path}")
                
                return str(output_path)
                
            except Exception as e:
                if verbose:
                    print(f"Download failed: {e}")
                continue
        
        return None
    
    elif method == "http_scrape":
        # Scrape webpage for matching files
        base_url = model.download.get("base_url")
        pattern = model.download.get("pattern")
        
        if not base_url or not pattern:
            if verbose:
                print(f"Missing base_url or pattern for '{model.name}'")
            return None
        
        try:
            with urllib.request.urlopen(base_url, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            
            # Find all matching hrefs
            pat_re = re.compile(pattern)
            hrefs = re.findall(r'href=["\']([^"\']+)["\']', html)
            matches = [h for h in hrefs if pat_re.search(h)]
            
            if not matches:
                if verbose:
                    print(f"No matching files found at {base_url}")
                return None
            
            # Download first match
            file_url = matches[0]
            if not file_url.startswith("http"):
                file_url = base_url.rstrip("/") + "/" + file_url.lstrip("/")
            
            filename = file_url.split("/")[-1]
            output_path = target_dir / filename
            
            if verbose:
                print(f"Downloading {file_url} -> {output_path}")
            
            urllib.request.urlretrieve(file_url, output_path)
            
            # Check if it needs extraction
            if model.download.get("extract") and filename.endswith(".zip"):
                if verbose:
                    print(f"Extracting {output_path}")
                with zipfile.ZipFile(output_path, 'r') as zf:
                    zf.extractall(target_dir)
                
                # Find the main file after extraction
                if model.files:
                    for file_pattern in model.files:
                        extracted = list(target_dir.glob(file_pattern))
                        if extracted:
                            output_path = extracted[0]
                            break
            
            model.local = True
            model.filepath = str(output_path)
            
            if verbose:
                print(f"Successfully downloaded to {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            if verbose:
                print(f"Download failed: {e}")
            return None
    
    elif method == "manual":
        if verbose:
            info_url = model.download.get("info_url", "N/A")
            print(f"Model '{model.name}' requires manual download.")
            print(f"See: {info_url}")
        return None
    
    else:
        if verbose:
            print(f"Unknown download method '{method}' for '{model.name}'")
        return None


def _convert_model_if_needed(
    model: VelocityModelInfo,
    downloaded_path: str,
    verbose: bool = False,
) -> str:
    """
    Check if downloaded model needs format conversion to PROJ-compatible GeoTIFF.
    
    Args:
        model: Model metadata
        downloaded_path: Path to downloaded file
        verbose: Print conversion progress
        
    Returns:
        Path to PROJ-compatible file (may be converted or original)
    """
    # Check if model needs conversion
    needs_conversion = model.download.get("requires_conversion", False)
    
    # Also check by format
    if not needs_conversion and model.format:
        non_proj_formats = [
            "NetCDF", "netcdf", "OpenSHA", "ASCII", "csv", "KMZ"
        ]
        needs_conversion = any(fmt in model.format for fmt in non_proj_formats)
    
    if not needs_conversion:
        # Already PROJ-compatible
        return downloaded_path
    
    # Try to import converter
    try:
        from velocity_model_converters import convert_model
    except ImportError:
        if verbose:
            print(f"WARNING: Model {model.name} requires conversion but "
                  f"velocity_model_converters module not available. "
                  f"Install required packages: rasterio, xarray, scipy")
        return downloaded_path
    
    # Generate output path for converted file
    downloaded = Path(downloaded_path)
    converted_path = downloaded.parent / f"{model.name}_velocity.tif"
    
    # Check if already converted
    if converted_path.exists():
        if verbose:
            print(f"Using existing converted file: {converted_path}")
        model.filepath = str(converted_path)
        return str(converted_path)
    
    # Perform conversion
    if verbose:
        print(f"Converting {model.name} to PROJ-compatible GeoTIFF...")
        print(f"  Source: {downloaded_path}")
        print(f"  Output: {converted_path}")
    
    try:
        # Get conversion options from model metadata
        conv_opts = model.download.get("conversion_options", {})
        
        result = convert_model(
            model_name=model.name,
            input_path=downloaded,
            output_path=converted_path,
            **conv_opts,
        )
        
        model.filepath = str(result)
        
        if verbose:
            print(f"✓ Conversion successful: {result}")
        
        return str(result)
        
    except Exception as e:
        if verbose:
            print(f"WARNING: Conversion failed: {e}")
            print(f"You may need to manually convert {downloaded_path}")
        return downloaded_path


def download_and_convert_model(
    model: VelocityModelInfo,
    verbose: bool = False,
) -> Optional[str]:
    """
    Download and auto-convert model to PROJ-compatible format.
    
    This is a wrapper around download_model that adds automatic conversion.
    
    Args:
        model: VelocityModelInfo with download configuration
        verbose: Print progress
        
    Returns:
        Path to PROJ-ready file, or None if failed
    """
    # Download first
    downloaded = download_model(model, verbose=verbose, auto_convert=False)
    
    if not downloaded:
        return None
    
    # Convert if needed
    return _convert_model_if_needed(model, downloaded, verbose=verbose)


# ============================================================================
# Spatial and temporal filtering
# ============================================================================

def _bbox_intersects(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    """Check if two bounding boxes intersect."""
    aminx, aminy, amaxx, amaxy = a
    bminx, bminy, bmaxx, bmaxy = b
    return not (amaxx < bminx or bmaxx < aminx or amaxy < bminy or bmaxy < aminy)


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """Calculate bbox area (simple rectangular approximation)."""
    minx, miny, maxx, maxy = bbox
    return (maxx - minx) * (maxy - miny)


def filter_models(
    models: List[VelocityModelInfo],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    epoch_range: Optional[Tuple[float, float]] = None,
    kind: Optional[str] = None,
    require_local: bool = False,
) -> List[VelocityModelInfo]:
    """
    Filter models by spatial extent, temporal coverage, and type.
    
    Args:
        models: List of models to filter
        bbox: Geographic bounding box (min_lon, min_lat, max_lon, max_lat)
        epoch_range: Time range (min_epoch, max_epoch) 
        kind: Model kind ('velocity', 'deformation', 'strain_rate')
        require_local: Only return models available locally
    
    Returns:
        Filtered list of models
    """
    filtered = models
    
    # Spatial filter
    if bbox is not None:
        filtered = [m for m in filtered if _bbox_intersects(bbox, m.bbox)]
    
    # Temporal filter
    if epoch_range is not None:
        t0, t1 = min(epoch_range), max(epoch_range)
        tol = 1e-6
        filtered = [
            m for m in filtered
            if not (m.epoch_min - tol > t0 or m.epoch_max + tol < t1)
        ]
    
    # Kind filter
    if kind is not None:
        filtered = [m for m in filtered if m.kind == kind]
    
    # Local availability filter
    if require_local:
        filtered = [m for m in filtered if find_model_file(m, auto_download=False)]
    
    return filtered


# ============================================================================
# Model selection logic
# ============================================================================

def select_velocity_model(
    bbox_4326: Tuple[float, float, float, float],
    src_epoch: float,
    dst_epoch: float,
    *,
    models: Optional[List[VelocityModelInfo]] = None,
    registry_path: Optional[Path] = None,
    choice: Optional[int] = None,
    prefer_kind: str = "velocity",
    auto_download: bool = True,
    verbose: bool = True,
) -> Tuple[VelocityModelInfo, List[VelocityModelInfo]]:
    """
    Select appropriate velocity/deformation model for transformation.
    
    Args:
        bbox_4326: Bounding box in EPSG:4326 (min_lon, min_lat, max_lon, max_lat)
        src_epoch: Source epoch (decimal year)
        dst_epoch: Destination epoch (decimal year)
        models: Pre-loaded models (if None, loads from registry)
        registry_path: Path to custom registry file
        choice: Manual index into candidates list (overrides auto-selection)
        prefer_kind: Preferred model kind ('velocity' or 'deformation')
        auto_download: Allow downloading missing models
        verbose: Print selection details
    
    Returns:
        (selected_model, candidates_list)
    
    Selection criteria (in order of priority):
      1. Spatial coverage: bbox must intersect model bbox
      2. Temporal coverage: src and dst epochs within model's valid range
      3. Local availability (if auto_download=False)
      4. Smallest bbox (most specific to AOI)
      5. Preferred kind
      6. Central epoch closest to transformation midpoint
    """
    # Load models if not provided
    if models is None:
        models = load_registry(registry_path=registry_path)
    
    if not models:
        raise ValueError("No velocity models available")
    
    # Filter by spatial and temporal coverage
    t0, t1 = min(src_epoch, dst_epoch), max(src_epoch, dst_epoch)
    candidates = filter_models(
        models,
        bbox=bbox_4326,
        epoch_range=(t0, t1),
        require_local=(not auto_download),
    )
    
    if not candidates:
        raise ValueError(
            f"No models found covering bbox={bbox_4326} and epochs [{t0}, {t1}]. "
            f"Available models: {[m.name for m in models]}"
        )
    
    # Manual choice override
    if choice is not None:
        if not (0 <= choice < len(candidates)):
            raise IndexError(
                f"choice={choice} out of range (0..{len(candidates)-1})"
            )
        selected = candidates[choice]
        reason = "user override"
    else:
        # Auto-selection scoring
        mid_epoch = 0.5 * (t0 + t1)
        
        def score_model(m: VelocityModelInfo) -> Tuple:
            """Return sort key: lower is better."""
            # 1. Prefer models with files actually available
            has_files = find_model_file(m, auto_download=False) is not None
            file_rank = 0 if has_files else 1
            
            # 2. Prefer smaller bbox (more specific)
            area = _bbox_area(m.bbox)
            
            # 3. Prefer requested kind
            kind_match = 0 if m.kind == prefer_kind else 1
            
            # 4. Prefer central epoch close to midpoint
            if m.central_epoch is not None:
                epoch_dev = abs(m.central_epoch - mid_epoch)
            else:
                # No central epoch - use midpoint of valid range
                model_mid = 0.5 * (m.epoch_min + m.epoch_max)
                epoch_dev = abs(model_mid - mid_epoch)
            
            return (file_rank, area, kind_match, epoch_dev)
        
        sorted_candidates = sorted(candidates, key=score_model)
        selected = sorted_candidates[0]
        reason = "auto-selected best match"
    
    # Ensure file is available
    if auto_download:
        filepath = find_model_file(selected, auto_download=True, verbose=verbose)
        selected.filepath = filepath  # FIX: Store the resolved filepath on the model object
        if not filepath:
            warnings.warn(
                f"Selected model '{selected.name}' files not available and "
                f"download failed. Pipeline may fail."
            )
    else:
        # Even if auto_download is False, try to find existing file
        filepath = find_model_file(selected, auto_download=False, verbose=False)
        selected.filepath = filepath  # FIX: Store the resolved filepath
    
    # Print selection report
    if verbose:
        print(f"\nVelocity models for bbox={bbox_4326}, epochs [{t0:.1f}, {t1:.1f}]:")
        print(f"  Total available: {len(models)}")
        print(f"  Matching candidates: {len(candidates)}\n")
        
        for i, m in enumerate(candidates):
            is_selected = (m is selected)
            local_str = "local" if find_model_file(m, auto_download=False) else "remote"
            mark = f"  <== {reason}" if is_selected else ""
            
            print(f"  [{i}] {m.name}")
            print(f"      {m.label}")
            print(f"      Region: {m.region}")
            print(f"      Coverage: {m.bbox}")
            print(f"      Epochs: {m.epoch_min}-{m.epoch_max} "
                  f"(central: {m.central_epoch or 'N/A'})")
            print(f"      Kind: {m.kind} | Format: {m.format}")
            print(f"      Status: {local_str}{mark}")
            if is_selected and m.filepath:
                print(f"      File: {m.filepath}")
            print()
    
    return selected, candidates


# ============================================================================
# Custom user model support
# ============================================================================

def create_custom_model(
    name: str,
    filepath: str,
    bbox: Tuple[float, float, float, float],
    epoch_min: float,
    epoch_max: float,
    central_epoch: Optional[float] = None,
    kind: str = "velocity",
    proj_op: str = "deformation",
    **kwargs,
) -> VelocityModelInfo:
    """
    Create a custom velocity model from user-provided file.
    
    Args:
        name: Unique identifier
        filepath: Path to model file (GeoTIFF, GVB, JSON, etc.)
        bbox: Coverage (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
        epoch_min: Earliest valid epoch
        epoch_max: Latest valid epoch
        central_epoch: Reference epoch
        kind: 'velocity' or 'deformation'
        proj_op: PROJ operation ('deformation' or 'defmodel')
        **kwargs: Additional VelocityModelInfo fields
    
    Returns:
        VelocityModelInfo object ready for use
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = VelocityModelInfo(
        name=name,
        label=kwargs.get("label", f"Custom model: {name}"),
        region=kwargs.get("region", "User-defined"),
        bbox=bbox,
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        central_epoch=central_epoch,
        kind=kind,
        format=kwargs.get("format", filepath.suffix),
        proj_op=proj_op,
        files=[filepath.name],
        local=True,
        filepath=str(filepath),
        **{k: v for k, v in kwargs.items() 
           if k not in ['label', 'region', 'format']},
    )
    
    return model


# ============================================================================
# Convenience exports
# ============================================================================

__all__ = [
    "VelocityModelInfo",
    "load_registry",
    "find_model_file",
    "download_model",
    "download_and_convert_model",
    "filter_models",
    "select_velocity_model",
    "create_custom_model",
]