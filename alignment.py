"""
Automatic Point Cloud Registration for Landscape Data
Integrates with PointCloud and PointCloudPair classes
Uses small_gicp for registration and PDAL for I/O and filtering
"""

import json
import numpy as np
import small_gicp
# Use pdal_wrapper for Colab compatibility (falls back to native pdal locally)
try:
    from pdal_wrapper import pdal
except ImportError:
    import pdal
from typing import Optional, Dict, Tuple, List, Union, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import warnings
from pathlib import Path
from scipy.spatial import cKDTree
import logging
import tempfile
import os

if TYPE_CHECKING:
    from pointcloud import PointCloud
    from pointcloudpair import PointCloudPair

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegistrationMethod(Enum):
    """Available registration methods"""
    ICP = "icp"  # Standard ICP
    GICP = "gicp"  # Generalized ICP
    VGICP = "vgicp"  # Voxelized GICP
    PLANE_ICP = "plane_icp"  # Plane-to-plane ICP
    COLOR_ICP = "color_icp"  # Color-weighted ICP
    ROBUST_KERNEL = "robust_kernel"  # ICP with robust kernels


class DownsamplingMethod(Enum):
    """Point cloud downsampling methods"""
    VOXEL = "voxel"
    RANDOM = "random"
    UNIFORM = "uniform"
    FPS = "fps"  # Farthest point sampling
    NONE = "none"


@dataclass
class RegistrationConfig:
    """Configuration for point cloud registration"""
    
    # General parameters
    max_correspondence_distance: Optional[float] = None  # Auto-compute if None
    max_iterations: int = 50
    convergence_criteria: float = 1e-6
    num_threads: int = -1  # -1 for auto
    
    # Downsampling parameters
    downsampling_method: DownsamplingMethod = DownsamplingMethod.VOXEL
    voxel_size: Optional[float] = None  # Auto-compute if None
    target_points: int = 100000  # Target number of points after downsampling
    
    # GICP specific
    gicp_epsilon: float = 0.001
    
    # Robust kernel parameters (for outlier handling)
    use_robust_kernel: bool = True
    robust_kernel_delta: float = 1.0
    
    # Coarse alignment parameters
    perform_coarse_alignment: bool = True
    coarse_voxel_multiplier: float = 5.0  # Voxel size multiplier for coarse alignment
    
    # Landscape-specific optimizations
    use_ground_plane_constraint: bool = True
    ground_plane_weight: float = 0.1
    estimate_scale: bool = False  # For SfM to LiDAR alignment
    
    # Filtering parameters
    use_ground_filter: bool = False
    ground_filter_params: Optional[Dict[str, Any]] = None
    outlier_removal: bool = True
    outlier_k_neighbors: int = 20
    outlier_std_multiplier: float = 2.0
    classification_filter: Optional[Union[List[int], str]] = None  # e.g., [2] for ground only
    
    # Color support
    use_color: bool = False
    color_weight: float = 0.1
    
    # Validation
    min_fitness_score: float = 0.3  # Minimum acceptable fitness score
    max_rmse: Optional[float] = None  # Maximum acceptable RMSE (auto if None)
    
    # Auto-retry parameters
    enable_auto_retry: bool = True
    max_retries: int = 3
    retry_strategies: List[str] = field(default_factory=lambda: ["increase_correspondence", "change_method", "adjust_filtering"])


class PointCloudProcessor:
    """Preprocessing utilities for point clouds using PDAL"""
    
    @staticmethod
    def estimate_optimal_voxel_size(bounds: Tuple[float, float, float, float], 
                                   total_points: int,
                                   target_points: int = 100000) -> float:
        """
        Estimate optimal voxel size for downsampling to target number of points
        
        Args:
            bounds: (minx, miny, maxx, maxy) bounds
            total_points: Total number of points
            target_points: Target number of points after downsampling
            
        Returns:
            Estimated voxel size
        """
        if total_points <= target_points:
            return 0.0  # No downsampling needed
        
        # Estimate point cloud extent
        minx, miny, maxx, maxy = bounds
        area = (maxx - minx) * (maxy - miny)
        
        if area <= 0:
            return 1.0  # Default voxel size
        
        # Points per square meter
        density = total_points / area
        
        # Target density
        target_density = target_points / area
        
        # Voxel size to achieve target density
        voxel_size = np.sqrt(1.0 / target_density) if target_density > 0 else 1.0
        
        return float(voxel_size)
    
    @staticmethod
    def filter_outliers_pdal(input_path: str, output_path: str,
                            k_neighbors: int = 20,
                            std_multiplier: float = 2.0) -> str:
        """
        Remove statistical outliers using PDAL
        
        Args:
            input_path: Input LAS/LAZ file
            output_path: Output LAS/LAZ file
            k_neighbors: Number of neighbors for outlier detection
            std_multiplier: Standard deviation multiplier
            
        Returns:
            Path to filtered file
        """
        pipeline = pdal.Pipeline(json.dumps({
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": input_path
                },
                {
                    "type": "filters.outlier",
                    "method": "statistical",
                    "mean_k": k_neighbors,
                    "multiplier": std_multiplier
                },
                {
                    "type": "writers.las",
                    "filename": output_path
                }
            ]
        }))
        pipeline.execute()
        return output_path
    
    @staticmethod
    def apply_ground_filter(input_path: str, output_path: str,
                          smrf_params: Optional[Dict[str, Any]] = None,
                          keep_only_ground: bool = True) -> str:
        """
        Apply SMRF ground classification filter
        
        Args:
            input_path: Input LAS/LAZ file
            output_path: Output LAS/LAZ file  
            smrf_params: SMRF parameters
            keep_only_ground: If True, keep only ground points; if False, just classify
            
        Returns:
            Path to filtered file
        """
        defaults = {
            "cell": 1.0,
            "scalar": 1.25,
            "slope": 0.15,
            "threshold": 0.5,
            "window": 18.0
        }
        params = {**defaults, **(smrf_params or {})}
        
        pipeline_steps = [
            {
                "type": "readers.las",
                "filename": input_path
            },
            {
                "type": "filters.smrf",
                **params
            }
        ]
        
        if keep_only_ground:
            pipeline_steps.append({
                "type": "filters.range",
                "limits": "Classification[2:2]"  # Keep only ground
            })
        
        pipeline_steps.append({
            "type": "writers.las",
            "filename": output_path
        })
        
        pipeline = pdal.Pipeline(json.dumps({"pipeline": pipeline_steps}))
        pipeline.execute()
        return output_path
    
    @staticmethod
    def filter_by_classification(input_path: str, output_path: str,
                                classifications: List[int]) -> str:
        """
        Filter points by classification codes
        
        Args:
            input_path: Input LAS/LAZ file
            output_path: Output LAS/LAZ file
            classifications: List of classification codes to keep
            
        Returns:
            Path to filtered file
        """
        limits = ",".join([f"Classification[{c}:{c}]" for c in classifications])
        
        pipeline = pdal.Pipeline(json.dumps({
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": input_path
                },
                {
                    "type": "filters.range",
                    "limits": limits
                },
                {
                    "type": "writers.las",
                    "filename": output_path
                }
            ]
        }))
        pipeline.execute()
        return output_path
    
    @staticmethod
    def downsample_voxel_pdal(input_path: str, output_path: str,
                             voxel_size: float) -> str:
        """
        Downsample using voxel grid with PDAL
        
        Args:
            input_path: Input LAS/LAZ file
            output_path: Output LAS/LAZ file
            voxel_size: Voxel size for downsampling
            
        Returns:
            Path to downsampled file
        """
        pipeline = pdal.Pipeline(json.dumps({
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": input_path
                },
                {
                    "type": "filters.voxelcenternearestneighbor",
                    "cell": voxel_size
                },
                {
                    "type": "writers.las",
                    "filename": output_path
                }
            ]
        }))
        pipeline.execute()
        return output_path
    
    @staticmethod
    def extract_points_and_colors(las_path: str, 
                                 max_points: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract points and optionally colors from LAS/LAZ file
        
        Args:
            las_path: Path to LAS/LAZ file
            max_points: Maximum number of points to load
            
        Returns:
            Points array (Nx3) and optional colors array (Nx3)
        """
        pipeline_steps = [
            {
                "type": "readers.las",
                "filename": las_path
            }
        ]
        
        if max_points is not None:
            pipeline_steps.extend([
                {"type": "filters.randomize"},
                {"type": "filters.head", "count": max_points}
            ])
        
        pipeline = pdal.Pipeline(json.dumps({"pipeline": pipeline_steps}))
        pipeline.execute()
        
        arrays = pipeline.arrays
        if not arrays:
            raise RuntimeError(f"No points loaded from {las_path}")
        
        arr = arrays[0]
        points = np.column_stack([arr["X"], arr["Y"], arr["Z"]])
        
        # Check for color channels
        colors = None
        if all(field in arr.dtype.names for field in ["Red", "Green", "Blue"]):
            # Normalize colors to 0-1 range (assuming 16-bit color)
            colors = np.column_stack([
                arr["Red"] / 65535.0,
                arr["Green"] / 65535.0,
                arr["Blue"] / 65535.0
            ])
        
        return points, colors


class RegistrationResult:
    """Container for registration results"""
    
    def __init__(self):
        self.transformation: np.ndarray = np.eye(4)
        self.rmse: float = np.inf
        self.fitness: float = 0.0
        self.num_correspondences: int = 0
        self.converged: bool = False
        self.iterations: int = 0
        self.scale: float = 1.0
        self.method_used: str = ""
        self.retry_count: int = 0
        self.metadata: Dict[str, Any] = {}
    
    def is_valid(self, config: RegistrationConfig) -> bool:
        """Check if registration result meets quality criteria"""
        max_rmse = config.max_rmse
        if max_rmse is None:
            # Auto-compute based on typical landscape scale
            max_rmse = 5.0  # meters, reasonable for landscape data
        
        return (self.fitness >= config.min_fitness_score and 
                self.rmse <= max_rmse and
                self.converged)
    
    def __repr__(self) -> str:
        return (f"RegistrationResult(method={self.method_used}, "
                f"rmse={self.rmse:.3f}, "
                f"fitness={self.fitness:.3f}, "
                f"converged={self.converged}, "
                f"iterations={self.iterations}, "
                f"retries={self.retry_count})")


class LandscapeAligner:
    """Main class for landscape point cloud alignment with PointCloud integration"""
    
    def __init__(self, config: Optional[RegistrationConfig] = None):
        """
        Initialize aligner with configuration
        
        Args:
            config: Registration configuration (uses defaults if None)
        """
        self.config = config or RegistrationConfig()
        self.processor = PointCloudProcessor()
        
    def align(self, 
             source: Union[str, 'PointCloud', np.ndarray],
             target: Union[str, 'PointCloud', np.ndarray],
             method: RegistrationMethod = RegistrationMethod.VGICP,
             initial_transform: Optional[np.ndarray] = None) -> RegistrationResult:
        """
        Align source point cloud to target with automatic retry on failure
        
        Args:
            source: Source point cloud (file path, PointCloud object, or Nx3 array)
            target: Target point cloud (file path, PointCloud object, or Nx3 array)
            method: Registration method to use
            initial_transform: Optional initial transformation
            
        Returns:
            Registration result
        """
        # Extract paths and metadata
        source_path, source_meta = self._extract_path_and_metadata(source)
        target_path, target_meta = self._extract_path_and_metadata(target)
        
        logger.info(f"Starting registration: {source_meta['name']} -> {target_meta['name']}")
        
        # Auto-compute max correspondence distance if needed
        if self.config.max_correspondence_distance is None:
            self._auto_compute_correspondence_distance(source_meta, target_meta)
        
        # Main registration with retry logic
        if self.config.enable_auto_retry:
            result = self._align_with_retry(
                source_path, target_path, 
                source_meta, target_meta,
                method, initial_transform
            )
        else:
            result = self._align_single_attempt(
                source_path, target_path,
                source_meta, target_meta, 
                method, initial_transform
            )
        
        return result
    
    def _extract_path_and_metadata(self, 
                                  data: Union[str, 'PointCloud', np.ndarray]) -> Tuple[str, Dict[str, Any]]:
        """Extract file path and metadata from various input formats"""
        metadata = {
            "name": "unknown",
            "bounds": None,
            "total_points": None,
            "units": "meters",
            "crs": None,
            "epoch": None
        }
        
        if isinstance(data, str):
            # Direct file path
            path = data
            metadata["name"] = Path(path).stem
            
            # Quick metadata extraction with PDAL
            pipeline = pdal.Pipeline(json.dumps({
                "pipeline": [
                    {"type": "readers.las", "filename": path, "count": 0}
                ]
            }))
            pipeline.execute()
            meta = json.loads(pipeline.metadata)
            las_meta = meta.get("metadata", {}).get("readers.las", {})
            
            metadata["bounds"] = (
                las_meta.get("minx"), las_meta.get("miny"),
                las_meta.get("maxx"), las_meta.get("maxy")
            )
            metadata["total_points"] = las_meta.get("count")
            
        elif hasattr(data, 'filename'):
            # PointCloud object
            path = data.filename
            metadata["name"] = Path(path).stem
            metadata["bounds"] = getattr(data, 'bounds', None)
            metadata["total_points"] = getattr(data, 'total_points', None)
            metadata["units"] = getattr(data, 'horizontal_units', 'meters')
            metadata["crs"] = getattr(data, 'current_compound_crs', None) or getattr(data, 'original_compound_crs', None)
            metadata["epoch"] = getattr(data, 'epoch', None)
            
        elif isinstance(data, np.ndarray):
            # NumPy array - need to save to temp file
            with tempfile.NamedTemporaryFile(suffix='.las', delete=False) as tmp:
                path = tmp.name
                
            # Write array to LAS using PDAL
            structured_array = np.zeros(len(data), dtype=[('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')])
            structured_array['X'] = data[:, 0]
            structured_array['Y'] = data[:, 1]
            structured_array['Z'] = data[:, 2]
            
            pipeline = pdal.Pipeline(json.dumps({
                "pipeline": [
                    {
                        "type": "writers.las",
                        "filename": path
                    }
                ]
            }), arrays=[structured_array])
            pipeline.execute()
            
            metadata["name"] = "numpy_array"
            metadata["total_points"] = len(data)
            metadata["bounds"] = (
                data[:, 0].min(), data[:, 1].min(),
                data[:, 0].max(), data[:, 1].max()
            )
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
        
        return path, metadata
    
    def _auto_compute_correspondence_distance(self, 
                                             source_meta: Dict[str, Any],
                                             target_meta: Dict[str, Any]) -> None:
        """Auto-compute max correspondence distance based on data scale and units"""
        bounds_list = []
        for meta in [source_meta, target_meta]:
            if meta["bounds"] and all(b is not None for b in meta["bounds"]):
                bounds_list.append(meta["bounds"])
        
        if not bounds_list:
            # Default for landscape data in meters
            self.config.max_correspondence_distance = 10.0
            return
        
        # Compute based on extent
        max_extent = 0
        for bounds in bounds_list:
            minx, miny, maxx, maxy = bounds
            extent = max(maxx - minx, maxy - miny)
            max_extent = max(max_extent, extent)
        
        # Adjust for units (assume meters unless specified otherwise)
        unit_scale = 1.0
        for meta in [source_meta, target_meta]:
            if meta["units"] and "foot" in meta["units"].lower():
                unit_scale = 0.3048  # Convert to meters
                break
        
        max_extent *= unit_scale
        
        # Set to 1% of max extent, clamped between 0.5 and 50 meters
        self.config.max_correspondence_distance = np.clip(
            max_extent * 0.01, 0.5, 50.0
        )
        logger.info(f"Auto-computed max correspondence distance: {self.config.max_correspondence_distance:.2f} meters")
    
    def _align_with_retry(self,
                         source_path: str,
                         target_path: str,
                         source_meta: Dict[str, Any],
                         target_meta: Dict[str, Any],
                         method: RegistrationMethod,
                         initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        """
        Align with automatic retry using different strategies
        """
        best_result = None
        strategies_tried = []
        original_method = method
        
        for retry in range(self.config.max_retries):
            # Adjust config based on retry strategy
            if retry > 0:
                strategy = self.config.retry_strategies[min(retry - 1, len(self.config.retry_strategies) - 1)]
                self._apply_retry_strategy(strategy, retry)
                strategies_tried.append(strategy)
                
                # Try different methods on retry
                if retry == 1 and method == RegistrationMethod.VGICP:
                    method = RegistrationMethod.GICP
                    logger.info("Switching from VGICP to GICP for retry")
                elif retry == 2:
                    method = RegistrationMethod.ICP
                    logger.info("Switching to standard ICP for retry")
            
            try:
                result = self._align_single_attempt(
                    source_path, target_path,
                    source_meta, target_meta,
                    method, initial_transform
                )
                result.retry_count = retry
                
                if result.is_valid(self.config):
                    logger.info(f"Registration succeeded on attempt {retry + 1}")
                    return result
                
                if best_result is None or result.rmse < best_result.rmse:
                    best_result = result
                
                logger.warning(f"Registration attempt {retry + 1} failed validation. "
                             f"RMSE: {result.rmse:.3f}, Fitness: {result.fitness:.3f}")
                
            except Exception as e:
                logger.error(f"Registration attempt {retry + 1} failed: {e}")
        
        if best_result is None:
            best_result = RegistrationResult()
            best_result.metadata["error"] = "All registration attempts failed"
            best_result.metadata["strategies_tried"] = strategies_tried
        
        logger.warning(f"Registration failed after {self.config.max_retries} attempts. "
                      f"Best RMSE: {best_result.rmse:.3f}")
        return best_result
    
    def _apply_retry_strategy(self, strategy: str, retry_num: int) -> None:
        """Apply a retry strategy by modifying config"""
        if strategy == "increase_correspondence":
            # Increase correspondence distance
            self.config.max_correspondence_distance *= 1.5
            logger.info(f"Increased max correspondence distance to {self.config.max_correspondence_distance:.2f}")
            
        elif strategy == "change_method":
            # Method change handled in calling function
            pass
            
        elif strategy == "adjust_filtering":
            # Relax filtering parameters
            if self.config.outlier_removal:
                self.config.outlier_std_multiplier *= 1.5
                logger.info(f"Relaxed outlier threshold to {self.config.outlier_std_multiplier:.1f} std")
            
            # Increase target points for better coverage
            self.config.target_points = int(self.config.target_points * 1.5)
            logger.info(f"Increased target points to {self.config.target_points}")
    
    def _align_single_attempt(self,
                             source_path: str,
                             target_path: str,
                             source_meta: Dict[str, Any],
                             target_meta: Dict[str, Any],
                             method: RegistrationMethod,
                             initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        """
        Single registration attempt with preprocessing
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Preprocessing pipeline
            processed_source = self._preprocess_pointcloud(source_path, tmpdir, "source", source_meta)
            processed_target = self._preprocess_pointcloud(target_path, tmpdir, "target", target_meta)
            
            # Load processed points
            source_points, source_colors = self.processor.extract_points_and_colors(processed_source)
            target_points, target_colors = self.processor.extract_points_and_colors(processed_target)
            
            logger.info(f"After preprocessing: {len(source_points)} source, {len(target_points)} target points")
            
            # Coarse alignment if requested
            if self.config.perform_coarse_alignment and initial_transform is None:
                initial_transform = self._coarse_alignment(source_points, target_points)
                logger.info("Coarse alignment completed")
            
            # Fine registration
            result = self._fine_registration(
                source_points, target_points,
                source_colors, target_colors,
                method, initial_transform
            )
            
            result.method_used = method.value
            
            return result
    
    def _preprocess_pointcloud(self,
                              input_path: str,
                              tmpdir: str,
                              prefix: str,
                              metadata: Dict[str, Any]) -> str:
        """
        Apply preprocessing pipeline to point cloud
        """
        current_path = input_path
        step = 0
        
        # 1. Classification filtering
        if self.config.classification_filter is not None:
            step += 1
            output_path = os.path.join(tmpdir, f"{prefix}_step{step}_classified.laz")
            
            if self.config.classification_filter == "ground":
                classifications = [2]  # Ground class
            elif isinstance(self.config.classification_filter, (list, tuple)):
                classifications = list(self.config.classification_filter)
            else:
                classifications = None
            
            if classifications:
                current_path = self.processor.filter_by_classification(
                    current_path, output_path, classifications
                )
                logger.info(f"Filtered {prefix} to classifications: {classifications}")
        
        # 2. Ground filtering with SMRF
        if self.config.use_ground_filter:
            step += 1
            output_path = os.path.join(tmpdir, f"{prefix}_step{step}_ground.laz")
            current_path = self.processor.apply_ground_filter(
                current_path, output_path, self.config.ground_filter_params
            )
            logger.info(f"Applied ground filter to {prefix}")
        
        # 3. Outlier removal
        if self.config.outlier_removal:
            step += 1
            output_path = os.path.join(tmpdir, f"{prefix}_step{step}_cleaned.laz")
            current_path = self.processor.filter_outliers_pdal(
                current_path, output_path,
                self.config.outlier_k_neighbors,
                self.config.outlier_std_multiplier
            )
            logger.info(f"Removed outliers from {prefix}")
        
        # 4. Downsampling
        if self.config.voxel_size is None and metadata["bounds"] and metadata["total_points"]:
            # Auto-compute voxel size
            voxel_size = self.processor.estimate_optimal_voxel_size(
                metadata["bounds"],
                metadata["total_points"],
                self.config.target_points
            )
        else:
            voxel_size = self.config.voxel_size or 1.0
        
        if voxel_size > 0:
            step += 1
            output_path = os.path.join(tmpdir, f"{prefix}_step{step}_downsampled.laz")
            current_path = self.processor.downsample_voxel_pdal(
                current_path, output_path, voxel_size
            )
            logger.info(f"Downsampled {prefix} with voxel size {voxel_size:.3f}")
        
        return current_path
    
    def _coarse_alignment(self, 
                         source: np.ndarray, 
                         target: np.ndarray) -> np.ndarray:
        """
        Perform coarse alignment using centroids and PCA
        """
        # Centroid alignment
        source_centroid = source.mean(axis=0)
        target_centroid = target.mean(axis=0)
        
        initial_transform = np.eye(4)
        initial_transform[:3, 3] = target_centroid - source_centroid
        
        # Optional: PCA alignment for rotation (for landscapes, often not needed)
        if self.config.use_ground_plane_constraint:
            # For landscapes, we typically don't want to rotate around vertical axis
            # Just use translation
            pass
        elif len(source) > 1000 and len(target) > 1000:
            # Use subset for PCA
            source_subset = source[::max(1, len(source)//1000)]
            target_subset = target[::max(1, len(target)//1000)]
            
            # Center points
            source_centered = source_subset - source_centroid
            target_centered = target_subset - target_centroid
            
            # Compute principal axes
            _, _, source_axes = np.linalg.svd(source_centered.T @ source_centered)
            _, _, target_axes = np.linalg.svd(target_centered.T @ target_centered)
            
            # Compute rotation to align principal axes
            rotation = target_axes.T @ source_axes
            
            # Check for reflection
            if np.linalg.det(rotation) < 0:
                target_axes[-1] *= -1
                rotation = target_axes.T @ source_axes
            
            initial_transform[:3, :3] = rotation
        
        return initial_transform
    
    def _fine_registration(self,
                          source: np.ndarray,
                          target: np.ndarray,
                          source_colors: Optional[np.ndarray],
                          target_colors: Optional[np.ndarray],
                          method: RegistrationMethod,
                          initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        """
        Perform fine registration using small_gicp
        """
        result = RegistrationResult()
        
        # Prepare point clouds for small_gicp
        source_cloud = small_gicp.PointCloud(source)
        target_cloud = small_gicp.PointCloud(target)
        
        # Add colors if available and requested
        if self.config.use_color and source_colors is not None and target_colors is not None:
            # Note: small_gicp doesn't directly support colors in registration
            # This is a placeholder for future color-based correspondence weighting
            logger.info("Color information available (not yet integrated into registration)")
        
        # Build KD trees
        source_tree = small_gicp.KdTree(source_cloud)
        target_tree = small_gicp.KdTree(target_cloud)
        
        # Estimate normals if needed
        if method in [RegistrationMethod.GICP, RegistrationMethod.VGICP, 
                     RegistrationMethod.PLANE_ICP]:
            logger.info("Estimating normals...")
            source_cloud.estimate_normals(k=30)
            target_cloud.estimate_normals(k=30)
        
        # Set initial transformation
        if initial_transform is None:
            initial_transform = np.eye(4)
        
        try:
            # Perform registration
            logger.info(f"Running {method.value} registration...")
            
            if method == RegistrationMethod.VGICP:
                # Voxelized GICP
                voxel_size = self.config.voxel_size or 1.0
                source_voxels = small_gicp.VoxelHashMap(source_cloud, voxel_size)
                target_voxels = small_gicp.VoxelHashMap(target_cloud, voxel_size)
                
                reg_result = small_gicp.align(
                    source_voxels, target_voxels,
                    init_T=initial_transform,
                    method="VGICP",
                    max_iterations=self.config.max_iterations,
                    max_correspondence_distance=self.config.max_correspondence_distance
                )
            else:
                # Map methods to small_gicp names
                method_map = {
                    RegistrationMethod.ICP: "ICP",
                    RegistrationMethod.GICP: "GICP",
                    RegistrationMethod.PLANE_ICP: "GICP",  # Use GICP with normals
                    RegistrationMethod.ROBUST_KERNEL: "ICP",
                    RegistrationMethod.COLOR_ICP: "ICP"  # Fall back to ICP
                }
                
                reg_result = small_gicp.align(
                    source_tree, target_tree,
                    init_T=initial_transform,
                    method=method_map.get(method, "ICP"),
                    max_iterations=self.config.max_iterations,
                    max_correspondence_distance=self.config.max_correspondence_distance
                )
            
            # Extract results
            result.transformation = reg_result.T
            result.converged = reg_result.converged
            result.iterations = reg_result.iterations
            
            # Compute fitness metrics
            result.rmse, result.fitness, result.num_correspondences = \
                self._compute_fitness(source, target, result.transformation)
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            result.metadata['error'] = str(e)
        
        return result
    
    def _compute_fitness(self, 
                        source: np.ndarray, 
                        target: np.ndarray,
                        transformation: np.ndarray) -> Tuple[float, float, int]:
        """
        Compute registration fitness metrics
        """
        # Transform source points
        source_h = np.hstack([source, np.ones((len(source), 1))])
        source_transformed = (source_h @ transformation.T)[:, :3]
        
        # Find correspondences
        tree = cKDTree(target)
        distances, _ = tree.query(source_transformed)
        
        # Filter by max correspondence distance
        max_dist = self.config.max_correspondence_distance or 10.0
        mask = distances < max_dist
        valid_distances = distances[mask]
        
        if len(valid_distances) == 0:
            return np.inf, 0.0, 0
        
        # Compute metrics
        rmse = np.sqrt(np.mean(valid_distances ** 2))
        fitness = len(valid_distances) / len(source)
        num_correspondences = len(valid_distances)
        
        return rmse, fitness, num_correspondences


class PointCloudPairAligner:
    """Enhanced aligner for PointCloudPair objects with full integration"""
    
    def __init__(self, config: Optional[RegistrationConfig] = None):
        """
        Initialize pair aligner
        
        Args:
            config: Registration configuration
        """
        self.config = config or RegistrationConfig()
        self.aligner = LandscapeAligner(config)
    
    def align_pair(self, 
                  pair: 'PointCloudPair',
                  which: str = "pc1_to_pc2",
                  method: RegistrationMethod = RegistrationMethod.VGICP,
                  output_path: Optional[str] = None,
                  overwrite: bool = False) -> RegistrationResult:
        """
        Align a PointCloudPair with transformation applied to the result
        
        Args:
            pair: PointCloudPair object
            which: "pc1_to_pc2" or "pc2_to_pc1" - which cloud to align to which
            method: Registration method
            output_path: Optional output path for aligned point cloud
            overwrite: Whether to overwrite existing output
            
        Returns:
            Registration result
        """
        if which == "pc1_to_pc2":
            source_pc = pair.pc1
            target_pc = pair.pc2
            source_label = "pc1"
        elif which == "pc2_to_pc1":
            source_pc = pair.pc2
            target_pc = pair.pc1
            source_label = "pc2"
        else:
            raise ValueError("which must be 'pc1_to_pc2' or 'pc2_to_pc1'")
        
        # Check CRS compatibility
        source_crs = source_pc.current_compound_crs or source_pc.original_compound_crs
        target_crs = target_pc.current_compound_crs or target_pc.original_compound_crs
        
        if source_crs != target_crs:
            logger.warning("Point clouds have different CRS. Consider warping to common CRS first.")
        
        # Perform alignment
        result = self.aligner.align(source_pc, target_pc, method)
        
        # Apply transformation if successful
        if result.converged and output_path is not None:
            self._apply_transformation(
                source_pc,
                result.transformation,
                output_path,
                overwrite,
                target_crs,
                pair=pair,
                source_label=source_label,
            )
            
            # Update CRS history if available
            if hasattr(source_pc, 'crs_history') and source_pc.crs_history is not None:
                source_pc.crs_history.record_transformation_entry(
                    transformation_type="Align",
                    source_crs_proj=source_crs,
                    target_crs_proj=target_crs,
                    method=f"{method.value} (small_gicp)",
                    transformation_matrix=result.transformation.tolist(),
                    alignment_origin=source_label,
                    note=f"RMSE: {result.rmse:.3f}, Fitness: {result.fitness:.3f}"
                )
        
        return result
    
    def _apply_transformation(self,
                            pointcloud: 'PointCloud',
                            transformation: np.ndarray,
                            output_path: str,
                            overwrite: bool = False,
                            target_crs: Optional[str] = None,
                            pair: Optional['PointCloudPair'] = None,
                            source_label: Optional[str] = None) -> None:
        """
        Apply transformation matrix to point cloud and save
        """
        if Path(output_path).exists() and not overwrite:
            raise FileExistsError(f"Output file exists: {output_path}")
        
        # Format transformation matrix for PDAL
        matrix_str = " ".join(f"{val:.12g}" for val in transformation.reshape(-1))
        
        # Use target CRS if provided, otherwise use source CRS
        crs = target_crs or pointcloud.current_compound_crs or pointcloud.original_compound_crs
        
        pipeline_spec = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": pointcloud.filename
                },
                {
                    "type": "filters.transformation",
                    "matrix": matrix_str
                },
                {
                    "type": "writers.las",
                    "filename": output_path,
                    "a_srs": crs if crs else ""
                }
            ]
        }
        
        pipeline = pdal.Pipeline(json.dumps(pipeline_spec))
        pipeline.execute()
        
        logger.info(f"Transformed point cloud saved to {output_path}")
        
        # Instantiate a new PointCloud from the transformed file
        from pointcloud import PointCloud
        aligned_pc = PointCloud.from_file(str(output_path))
        
        # Update the pair object to reference the new aligned PointCloud
        if pair is not None and source_label is not None:
            if source_label == "pc1":
                pair.pc1 = aligned_pc
            elif source_label == "pc2":
                pair.pc2 = aligned_pc


# Convenience functions
def quick_align(source: Union[str, 'PointCloud'],
               target: Union[str, 'PointCloud'],
               method: str = "vgicp",
               use_ground: bool = False) -> Tuple[np.ndarray, float]:
    """
    Quick alignment with automatic parameters
    
    Args:
        source: Source point cloud (file path or PointCloud object)
        target: Target point cloud (file path or PointCloud object)
        method: Registration method name
        use_ground: Whether to filter to ground points only
        
    Returns:
        Transformation matrix and RMSE
    """
    config = RegistrationConfig(
        use_ground_filter=use_ground,
        classification_filter=[2] if use_ground else None,
        enable_auto_retry=True
    )
    
    aligner = LandscapeAligner(config)
    method_enum = RegistrationMethod(method.lower())
    result = aligner.align(source, target, method_enum)
    
    return result.transformation, result.rmse


def robust_align(source: Union[str, 'PointCloud'],
                target: Union[str, 'PointCloud'],
                methods: List[str] = ["vgicp", "gicp", "icp"]) -> RegistrationResult:
    """
    Try multiple methods and return best result
    
    Args:
        source: Source point cloud
        target: Target point cloud
        methods: List of method names to try
        
    Returns:
        Best registration result
    """
    best_result = None
    best_rmse = np.inf
    
    for method_name in methods:
        config = RegistrationConfig(
            enable_auto_retry=True,
            max_retries=2
        )
        
        aligner = LandscapeAligner(config)
        
        try:
            method = RegistrationMethod(method_name.lower())
            result = aligner.align(source, target, method)
            
            if result.rmse < best_rmse and result.is_valid(config):
                best_result = result
                best_rmse = result.rmse
                logger.info(f"Best method so far: {method_name} with RMSE {best_rmse:.3f}")
        except Exception as e:
            logger.warning(f"Method {method_name} failed: {e}")
    
    if best_result is None:
        raise RuntimeError("All registration methods failed")
    
    return best_result
