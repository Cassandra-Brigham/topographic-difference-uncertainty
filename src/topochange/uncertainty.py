"""Propagate variogram-based uncertainty to specific regions.

Provides utilities for computing spatial (correlated + uncorrelated)
uncertainties from variogram parameters and applying them to user-defined areas.

Classes:
- RegionalUncertaintyEstimator: Propagate uncertainty to polygon regions
"""

from __future__ import annotations

import math
from typing import Sequence, Optional, Dict, Any
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box, Point
from shapely.ops import unary_union
from shapely.prepared import prep
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from pathlib import Path
import geopandas as gpd

# Import variogram classes from variogram module
try:
    from variogram import RasterDataHandler, VariogramAnalysis
except ImportError:
    from .variogram import RasterDataHandler, VariogramAnalysis



class RegionalUncertaintyEstimator:
    """
    Estimate regional uncertainty σ_A over a polygon using several methods.

    This class combines:
      - a semivariogram model (from VariogramAnalysis)
      - a raster (RasterDataHandler)
      - a polygon of interest (area_of_interest)
      - optionally stable / unstable polygons for uncorrelated noise estimation

    Uncorrelated noise handling:
      * If stable_geoms is provided:
          σ0 is estimated from those stable areas only.
      * Else if unstable_geoms is provided and derive_stable_from_unstable=True:
          stable area is taken as (raster footprint − union(unstable_geoms)).
      * Else:
          σ0 is estimated from all valid raster pixels.

      Polygon-mean uncorrelated term:
          σ_uncorr,poly = σ0 / sqrt(N_poly),
          with N_poly ≈ area_poly / (cell_size^2).

      Raster-mean uncorrelated term:
          σ_uncorr,raster = σ0 / sqrt(N_raster),
          with N_raster = number of valid raster pixels.
    """

    @staticmethod
    def _as_multipolygon(geom):
        """Helper to normalize geometry to MultiPolygon."""
        if isinstance(geom, MultiPolygon):
            return geom
        elif isinstance(geom, Polygon):
            return MultiPolygon([geom])
        else:
            raise TypeError("Geometry must be a Polygon or MultiPolygon, not {}.".format(type(geom).__name__))

    def __init__(
        self,
        raster_data_handler: RasterDataHandler,
        variogram_analysis: VariogramAnalysis,
        area_of_interest,
        stable_geoms=None,
        unstable_geoms=None,
        derive_stable_from_unstable: bool = True,
    ):
        # ------------------------------------------------------------------ #
        # 1. Store handlers and variogram info
        # ------------------------------------------------------------------ #
        self.raster_data_handler = raster_data_handler
        self.variogram_analysis = variogram_analysis

        # Central sills/ranges
        self.ranges = np.array(variogram_analysis.ranges, dtype=float)
        self.sills = np.array(variogram_analysis.sills, dtype=float)

        # Min/max sills/ranges (fallback to central if None)
        sills_min_raw = getattr(variogram_analysis, "sills_min", None)
        sills_max_raw = getattr(variogram_analysis, "sills_max", None)
        ranges_min_raw = getattr(variogram_analysis, "ranges_min", None)
        ranges_max_raw = getattr(variogram_analysis, "ranges_max", None)

        self.sills_min = np.array(
            sills_min_raw if sills_min_raw is not None else variogram_analysis.sills,
            dtype=float,
        )
        self.sills_max = np.array(
            sills_max_raw if sills_max_raw is not None else variogram_analysis.sills,
            dtype=float,
        )
        self.ranges_min = np.array(
            ranges_min_raw if ranges_min_raw is not None else variogram_analysis.ranges,
            dtype=float,
        )
        self.ranges_max = np.array(
            ranges_max_raw if ranges_max_raw is not None else variogram_analysis.ranges,
            dtype=float,
        )

        # Nuggets (central + min/max)
        self.nugget = getattr(variogram_analysis, "best_nugget", 0.0)
        self.min_nugget = getattr(variogram_analysis, "min_nugget", self.nugget)
        self.max_nugget = getattr(variogram_analysis, "max_nugget", self.nugget)

        # Store fitted parameter vectors if you need them
        self.best_model_params = getattr(variogram_analysis, "best_params", None)
        self.best_model_params_min = None   # you can fill these later if you compute them
        self.best_model_params_max = None

        # Total variance (σ²_total) from sills + nugget
        self.sigma2 = (self.nugget or 0.0) + float(np.sum(self.sills))
        self.sigma2_min = (self.min_nugget or 0.0) + float(np.sum(self.sills_min))
        self.sigma2_max = (self.max_nugget or 0.0) + float(np.sum(self.sills_max))

        # Gamma (semivariogram) functions for central, min, max models
        self.gamma_func_total = None
        self.gamma_func_total_min = None
        self.gamma_func_total_max = None

        # Optional component-wise gamma functions
        self.gamma_func_1 = None
        self.gamma_func_2 = None
        self.gamma_func_3 = None
        self.gamma_func_1_min = None
        self.gamma_func_1_max = None
        self.gamma_func_2_min = None
        self.gamma_func_2_max = None
        self.gamma_func_3_min = None
        self.gamma_func_3_max = None

        # ---- γ(h) callables using your fitted spherical model ----
        if getattr(variogram_analysis, "best_model_func", None) is not None:
            # central model
            p1, p2, p3, p_tot = self.arrange_params(
                sills=self.sills,
                ranges=self.ranges,
                nugget=self.nugget,
            )
            self.gamma_func_total = self.bind_gamma(variogram_analysis.best_model_func, p_tot)
            self.gamma_func_1 = self.bind_gamma(variogram_analysis.best_model_func, p1)
            self.gamma_func_2 = self.bind_gamma(variogram_analysis.best_model_func, p2)
            self.gamma_func_3 = self.bind_gamma(variogram_analysis.best_model_func, p3)

            # min model
            p1_min, p2_min, p3_min, p_tot_min = self.arrange_params(
                sills=self.sills_min,
                ranges=self.ranges_min,
                nugget=self.min_nugget,
            )
            # max model
            p1_max, p2_max, p3_max, p_tot_max = self.arrange_params(
                sills=self.sills_max,
                ranges=self.ranges_max,
                nugget=self.max_nugget,
            )
            self.gamma_func_total_min = self.bind_gamma(variogram_analysis.best_model_func, p_tot_min)
            self.gamma_func_total_max = self.bind_gamma(variogram_analysis.best_model_func, p_tot_max)
            self.gamma_func_1_min = self.bind_gamma(variogram_analysis.best_model_func, p1_min)
            self.gamma_func_1_max = self.bind_gamma(variogram_analysis.best_model_func, p1_max)
            self.gamma_func_2_min = self.bind_gamma(variogram_analysis.best_model_func, p2_min)
            self.gamma_func_2_max = self.bind_gamma(variogram_analysis.best_model_func, p2_max)
            self.gamma_func_3_min = self.bind_gamma(variogram_analysis.best_model_func, p3_min)
            self.gamma_func_3_max = self.bind_gamma(variogram_analysis.best_model_func, p3_max)

        # ------------------------------------------------------------------ #
        # 2. Resolve polygon of interest
        # ------------------------------------------------------------------ #
        if isinstance(area_of_interest, (str, Path)):
            gdf = gpd.read_file(area_of_interest)
            if gdf.empty:
                raise ValueError(f"No geometries found in file: {area_of_interest}.")
            polygon = gdf.unary_union
        elif isinstance(area_of_interest, (Polygon, MultiPolygon)):
            polygon = area_of_interest
        else:
            raise TypeError("area_of_interest must be a file path or shapely Polygon/MultiPolygon.")

        if isinstance(polygon, MultiPolygon):
            polygon = unary_union(polygon)
        if not isinstance(polygon, Polygon) or polygon.is_empty or polygon.area <= 0:
            raise ValueError("Area of interest must be a valid Polygon with non-zero area.")

        self.polygon = polygon
        self.area = float(polygon.area)

        # ------------------------------------------------------------------ #
        # 3. Stable / unstable geometries for uncorrelated noise σ0
        # ------------------------------------------------------------------ #
        self.stable_geom = None
        self.unstable_geom = None

        # Collect unstable if given
        if unstable_geoms is not None:
            if isinstance(unstable_geoms, (Polygon, MultiPolygon)):
                self.unstable_geom = unstable_geoms
            else:
                # Assume iterable of geometries / GeoSeries
                self.unstable_geom = unary_union(list(unstable_geoms))

        # Collect stable if given explicitly
        if stable_geoms is not None:
            if isinstance(stable_geoms, (Polygon, MultiPolygon)):
                stable_union = stable_geoms
            else:
                stable_union = unary_union(list(stable_geoms))
            self.stable_geom = self._as_multipolygon(stable_union)

        # Derive stable from inverse of unstable if requested and no explicit stable
        if self.stable_geom is None and self.unstable_geom is not None and derive_stable_from_unstable:
            # Build raster footprint
            raster_data_handler.get_detailed_area()
            raster_footprint = raster_data_handler.merged_geom
            if raster_footprint is None:
                minx, miny, maxx, maxy = raster_data_handler.bounds
                raster_footprint = box(minx, miny, maxx, maxy)
            # Stable = footprint - union(unstable)
            stable_from_inverse = raster_footprint.difference(self.unstable_geom)
            if isinstance(stable_from_inverse, (Polygon, MultiPolygon)):
                self.stable_geom = self._as_multipolygon(stable_from_inverse)

        # ------------------------------------------------------------------ #
        # 4. Storage for results (polygon & raster)
        # ------------------------------------------------------------------ #
        # Uncorrelated terms
        self.sigma0_uncorrelated = None
        self.mean_random_uncorrelated = None        # polygon-mean term
        self.mean_random_uncorrelated_raster = None # raster-mean term

        # Correlated terms – polygon
        self.total_mean_correlated_uncertainty_polygon = None
        self.total_mean_correlated_uncertainty_min_polygon = None
        self.total_mean_correlated_uncertainty_max_polygon = None
        self.mean_random_correlated_1_polygon = None
        self.mean_random_correlated_2_polygon = None
        self.mean_random_correlated_3_polygon = None
        self.mean_random_correlated_1_min_polygon = None
        self.mean_random_correlated_1_max_polygon = None
        self.mean_random_correlated_2_min_polygon = None
        self.mean_random_correlated_2_max_polygon = None
        self.mean_random_correlated_3_min_polygon = None
        self.mean_random_correlated_3_max_polygon = None

        # Correlated terms – raster
        self.total_mean_correlated_uncertainty_raster = None
        self.total_mean_correlated_uncertainty_min_raster = None
        self.total_mean_correlated_uncertainty_max_raster = None
        self.mean_random_correlated_1_raster = None
        self.mean_random_correlated_2_raster = None
        self.mean_random_correlated_3_raster = None
        self.mean_random_correlated_1_min_raster = None
        self.mean_random_correlated_1_max_raster = None
        self.mean_random_correlated_2_min_raster = None
        self.mean_random_correlated_2_max_raster = None
        self.mean_random_correlated_3_min_raster = None
        self.mean_random_correlated_3_max_raster = None

        # Total uncertainties (polygon & raster)
        self.total_mean_uncertainty_polygon = None
        self.total_mean_uncertainty_min_polygon = None
        self.total_mean_uncertainty_max_polygon = None
        self.total_mean_uncertainty_raster = None
        self.total_mean_uncertainty_min_raster = None
        self.total_mean_uncertainty_max_raster = None

    # ---------------------------------------------------------------------- #
    # Helper: bind gamma function from your variogram model
    # ---------------------------------------------------------------------- #
    @staticmethod
    def arrange_params(*, sills, ranges, nugget=None):
        """
        Build parameter lists for a fitted variogram model.

        Returns
        -------
        (params1, params2, params3, all_params)
            Where each paramsX is [C, a, (nugget?)] for a single component (or None),
            and all_params is [C1..Cn, a1..an, (nugget?)].
        """
        if len(sills) != len(ranges):
            raise ValueError("sills and ranges must have the same length.")

        # Convert to plain Python lists of scalars
        sills = list(sills)
        ranges = list(ranges)

        # Flattened total parameter vector for the fitted model
        all_params = sills + ranges
        if nugget is not None:
            all_params.append(nugget)

        # Per-component parameter lists: [C, a] or [C, a, nugget]
        param_sets = [[C, a] for C, a in zip(sills, ranges)]
        if nugget is not None:
            param_sets = [ps + [nugget] for ps in param_sets]

        # Pad to exactly 3 entries (some may be None)
        while len(param_sets) < 3:
            param_sets.append(None)

        # Returns: (params1, params2, params3, all_params)
        return (*param_sets, all_params)

    @staticmethod
    def bind_gamma(model_func, params):
        """Return γ(h) with parameters baked in (or None)."""
        if params is None:
            return None

        # model_func is either spherical_model or spherical_model_with_nugget,
        # both expect: model_func(h, *params_flat)
        return lambda h: model_func(np.asarray(h, dtype=float), *params)

    # ---------------------------------------------------------------------- #
    # Uncorrelated term σ0 from stable / inverse-stable areas
    # ---------------------------------------------------------------------- #
    def calc_mean_random_uncorrelated(self, use_stable_areas: bool = True) -> None:
        """
        Compute uncorrelated (white) noise contribution to the mean.

        Steps:
        1) Estimate per-pixel uncorrelated σ0:
            - from stable areas if available and use_stable_areas=True
            - otherwise from all valid pixels.
        2) Propagate σ0 to:
            - polygon mean:  σ0 / sqrt(N_poly)
            - raster mean:   σ0 / sqrt(N_raster)
        """
        # Work with the full 2D raster grid from rioxarray, not the 1D data_array
        da = self.raster_data_handler.rioxarray_obj
        if da is None:
            raise RuntimeError("RasterDataHandler.rioxarray_obj is None. Call load_raster() first.")

        arr = da.values
        # If there's a band dimension, squeeze it out
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        arr = np.asarray(arr, dtype=float)

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D raster array, got shape {arr.shape}.")

        # Valid pixels on the full grid
        valid_mask = np.isfinite(arr)

        # Choose pixels from which to estimate sigma_0
        mask_for_sigma = valid_mask.copy()
        if use_stable_areas and self.stable_geom is not None:
            # geometry_mask works on the full 2D grid, so out_shape must be 2D
            transform = da.rio.transform()
            # geometry_mask: True outside geometries when invert=False
            stable_outside = geometry_mask(
                [self.stable_geom],
                out_shape=arr.shape,   # <- 2D (rows, cols), NOT data_array.shape
                transform=transform,
                invert=False,
            )
            stable_mask = ~stable_outside  # True inside stable polygons
            mask_for_sigma = valid_mask & stable_mask

            # If the stable area doesn't overlap the raster, fall back to all valid pixels
            if not np.any(mask_for_sigma):
                mask_for_sigma = valid_mask

        values = arr[mask_for_sigma]
        if values.size == 0:
            raise RuntimeError("No valid values available to estimate uncorrelated sigma.")

        # Per-pixel uncorrelated sigma (RMS; robust if mean is near zero)
        sigma0 = float(np.sqrt(np.mean(values ** 2)))
        self.sigma0_uncorrelated = sigma0

        # Polygon-mean uncorrelated term: sigma_0 / sqrt(N_poly), where N_poly from area
        res = float(self.raster_data_handler.resolution)
        cell_area = res ** 2 if res > 0 else 1.0

        if getattr(self, "area", 0.0) > 0.0:
            N_poly = max(self.area / cell_area, 1.0)
            self.mean_random_uncorrelated = sigma0 / math.sqrt(N_poly)
        else:
            self.mean_random_uncorrelated = None

        # Raster-mean uncorrelated term: σ0 / sqrt(N_raster) using all valid pixels
        N_raster = int(valid_mask.sum())
        if N_raster > 0:
            self.mean_random_uncorrelated_raster = sigma0 / math.sqrt(N_raster)
        else:
            self.mean_random_uncorrelated_raster = None

    # ---------------------------------------------------------------------- #
    # Correlated parts – these should call your existing implementations
    # ---------------------------------------------------------------------- #
    def estimate_variance_mean_monte_carlo_pairs(
        self,
        domain: Polygon,
        n_pairs: int = 200_000,
        seed: Optional[int] = None,
        sigma_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        sigma2_total: Optional[float] = None,
        gamma_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> float:
        """
        Plain Monte Carlo on independent pairs.

        Homoscedastic:
            Var(mean) ≈ E[ C(‖X−Y‖) ],  with C(h)=σ²_total−γ(h)

        Heteroscedastic (provide sigma_func returning STD at (x,y)):
            Var(mean) ≈ E[ ρ(‖X−Y‖) * σ(X) * σ(Y) ],  ρ(h)=1−γ(h)/σ²_total
        """
        if n_pairs <= 0:
            raise ValueError("n_pairs must be a positive integer.")

        rng = np.random.default_rng(seed)
        da = self.raster_data_handler.rioxarray_obj
        if da is None:
            raise RuntimeError("RasterDataHandler.rioxarray_obj is None. Call load_raster() first.")

        xs = da.x.values
        ys = da.y.values
        bounds = domain.bounds
        minx, miny, maxx, maxy = bounds

        # Random points in bounding box, then keep those in polygon
        pts = []
        while len(pts) < n_pairs * 2:  # oversample; we'll pair later
            rand_x = rng.uniform(minx, maxx, size=n_pairs)
            rand_y = rng.uniform(miny, maxy, size=n_pairs)
            for x, y in zip(rand_x, rand_y):
                if domain.contains(Point(float(x), float(y))):
                    pts.append((x, y))
                if len(pts) >= n_pairs * 2:
                    break

        pts = np.array(pts)
        # Take pairs
        X = pts[:n_pairs]
        Y = pts[n_pairs:2 * n_pairs]

        # distances
        h = np.linalg.norm(X - Y, axis=1)
        if gamma_func is None:
            if self.gamma_func_total is None:
                raise RuntimeError("No gamma function available. Fit a variogram model first.")
            gamma = self.gamma_func_total(h)
        else:
            gamma = gamma_func(h)

        if sigma2_total is None:
            sigma2_total = self.sigma2

        if sigma_func is None:
            # Homoscedastic case
            cov = sigma2_total - gamma
            var_mean = float(np.mean(cov))
        else:
            # Heteroscedastic case: need ρ(h) and σ(X), σ(Y)
            rho = 1.0 - gamma / sigma2_total
            rho = np.clip(rho, -1.0, 1.0)

            # Map (x,y) to σ via sigma_func
            sigX = sigma_func(X)
            sigY = sigma_func(Y)
            var_mean = float(np.mean(rho * sigX * sigY))

        return 0.0 if var_mean < 0 else float(np.sqrt(var_mean))

    def calc_mean_random_correlated_polygon(
        self,
        n_pairs: int = 200_000,
        seed: Optional[int] = None,
        sigma_func=None,
    ) -> None:
        """
        Fill polygon correlated terms using your Monte Carlo estimator.
        """
        if self.gamma_func_total is not None:
            self.total_mean_correlated_uncertainty_polygon = self.estimate_variance_mean_monte_carlo_pairs(
                domain=self.polygon,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2,
                gamma_func=self.gamma_func_total,
            )
        if self.gamma_func_total_min is not None:
            self.total_mean_correlated_uncertainty_min_polygon = self.estimate_variance_mean_monte_carlo_pairs(
                domain=self.polygon,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2_min,
                gamma_func=self.gamma_func_total_min,
            )
        if self.gamma_func_total_max is not None:
            self.total_mean_correlated_uncertainty_max_polygon = self.estimate_variance_mean_monte_carlo_pairs(
                domain=self.polygon,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2_max,
                gamma_func=self.gamma_func_total_max,
            )

        # Optional component-wise (1,2,3) if you need them
        if self.gamma_func_1 is not None:
            self.mean_random_correlated_1_polygon = self.estimate_variance_mean_monte_carlo_pairs(
                domain=self.polygon,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2,
                gamma_func=self.gamma_func_1,
            )
        if self.gamma_func_2 is not None:
            self.mean_random_correlated_2_polygon = self.estimate_variance_mean_monte_carlo_pairs(
                domain=self.polygon,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2,
                gamma_func=self.gamma_func_2,
            )
        if self.gamma_func_3 is not None:
            self.mean_random_correlated_3_polygon = self.estimate_variance_mean_monte_carlo_pairs(
                domain=self.polygon,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2,
                gamma_func=self.gamma_func_3,
            )

    def calc_mean_random_correlated_raster(
        self,
        level_of_detail: str = "bbox",
        n_pairs: int = 200_000,
        seed: Optional[int] = None,
        sigma_func=None,
    ) -> None:
        """
        Estimate mean correlated uncertainty via raster integration / Monte Carlo.

        level_of_detail can be used to switch between using the raster bbox
        or a more detailed footprint polygon (if you implement that).
        """
        # Build raster-level domain polygon
        # Prefer detailed area if available
        self.raster_data_handler.get_detailed_area()
        raster_geom = self.raster_data_handler.merged_geom
        if raster_geom is None:
            minx, miny, maxx, maxy = self.raster_data_handler.bounds
            raster_geom = box(minx, miny, maxx, maxy)

        if self.gamma_func_total is not None:
            self.total_mean_correlated_uncertainty_raster = self.estimate_variance_mean_monte_carlo_pairs(
                domain=raster_geom,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2,
                gamma_func=self.gamma_func_total,
            )
        if self.gamma_func_total_min is not None:
            self.total_mean_correlated_uncertainty_min_raster = self.estimate_variance_mean_monte_carlo_pairs(
                domain=raster_geom,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2_min,
                gamma_func=self.gamma_func_total_min,
            )
        if self.gamma_func_total_max is not None:
            self.total_mean_correlated_uncertainty_max_raster = self.estimate_variance_mean_monte_carlo_pairs(
                domain=raster_geom,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2_max,
                gamma_func=self.gamma_func_total_max,
            )

        # Optional component-wise (1,2,3)
        if self.gamma_func_1 is not None:
            self.mean_random_correlated_1_raster = self.estimate_variance_mean_monte_carlo_pairs(
                domain=raster_geom,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2,
                gamma_func=self.gamma_func_1,
            )
        if self.gamma_func_2 is not None:
            self.mean_random_correlated_2_raster = self.estimate_variance_mean_monte_carlo_pairs(
                domain=raster_geom,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2,
                gamma_func=self.gamma_func_2,
            )
        if self.gamma_func_3 is not None:
            self.mean_random_correlated_3_raster = self.estimate_variance_mean_monte_carlo_pairs(
                domain=raster_geom,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2,
                gamma_func=self.gamma_func_3,
            )
        if self.gamma_func_1_min is not None:
            self.mean_random_correlated_1_min_polygon = self.estimate_variance_mean_monte_carlo_pairs(
                domain=self.polygon,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2_min,
                gamma_func=self.gamma_func_1_min,
            )
        if self.gamma_func_1_max is not None:
            self.mean_random_correlated_1_max_polygon = self.estimate_variance_mean_monte_carlo_pairs(
                domain=self.polygon,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2_max,
                gamma_func=self.gamma_func_1_max,
            )

        if self.gamma_func_2_min is not None:
            self.mean_random_correlated_2_min_polygon = self.estimate_variance_mean_monte_carlo_pairs(
                domain=self.polygon,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2_min,
                gamma_func=self.gamma_func_2_min,
            )
        if self.gamma_func_2_max is not None:
            self.mean_random_correlated_2_max_polygon = self.estimate_variance_mean_monte_carlo_pairs(
                domain=self.polygon,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2_max,
                gamma_func=self.gamma_func_2_max,
            )

        if self.gamma_func_3_min is not None:
            self.mean_random_correlated_3_min_polygon = self.estimate_variance_mean_monte_carlo_pairs(
                domain=self.polygon,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2_min,
                gamma_func=self.gamma_func_3_min,
            )
        if self.gamma_func_3_max is not None:
            self.mean_random_correlated_3_max_polygon = self.estimate_variance_mean_monte_carlo_pairs(
                domain=self.polygon,
                n_pairs=n_pairs,
                seed=seed,
                sigma_func=sigma_func,
                sigma2_total=self.sigma2_max,
                gamma_func=self.gamma_func_3_max,
            )

    # ---------------------------------------------------------------------- #
    # High-level wrapper: compute full uncertainty budget
    # ---------------------------------------------------------------------- #
    def calc_mean_uncertainty(
        self,
        n_pairs: int = 200_000,
        level_of_detail: str = "bbox",
        seed: Optional[int] = None,
        sigma_func=None,
        use_stable_areas_for_uncorr: bool = True,
    ) -> None:
        """
        Compute quadrature of uncorrelated + correlated terms for polygon and raster.

        Parameters
        ----------
        n_pairs : int
            Number of Monte Carlo pairs for correlated term estimates.
        level_of_detail : {"bbox", "raster"}
            How to sample the raster footprint for the raster-wide correlated term.
        seed : int, optional
            RNG seed for reproducibility.
        sigma_func : callable, optional
            Function σ(x,y) for heteroscedastic noise; if None, homoscedastic.
        use_stable_areas_for_uncorr : bool
            If True, estimate σ0 from stable areas (or inverse of unstable) if available.
        """
        # Uncorrelated part (polygon + raster)
        self.calc_mean_random_uncorrelated(use_stable_areas=use_stable_areas_for_uncorr)

        # Correlated parts
        self.calc_mean_random_correlated_polygon(
            n_pairs=n_pairs,
            seed=seed,
            sigma_func=sigma_func,
        )
        
        
        self.calc_mean_random_correlated_raster(
            level_of_detail=level_of_detail,
            n_pairs=n_pairs,
            seed=seed,
            sigma_func=sigma_func,
        )

        # Total uncertainty for polygon
        if (
            self.mean_random_uncorrelated is not None
            and self.total_mean_correlated_uncertainty_polygon is not None
        ):
            self.total_mean_uncertainty_polygon = math.sqrt(
                self.mean_random_uncorrelated ** 2
                + self.total_mean_correlated_uncertainty_polygon ** 2
            )
        if (
            self.mean_random_uncorrelated is not None
            and self.total_mean_correlated_uncertainty_min_polygon is not None
        ):
            self.total_mean_uncertainty_min_polygon = math.sqrt(
                self.mean_random_uncorrelated ** 2
                + self.total_mean_correlated_uncertainty_min_polygon ** 2
            )
        if (
            self.mean_random_uncorrelated is not None
            and self.total_mean_correlated_uncertainty_max_polygon is not None
        ):
            self.total_mean_uncertainty_max_polygon = math.sqrt(
                self.mean_random_uncorrelated ** 2
                + self.total_mean_correlated_uncertainty_max_polygon ** 2
            )

        # Total uncertainty for raster (uses raster-mean uncorrelated term)
        if (
            self.mean_random_uncorrelated_raster is not None
            and self.total_mean_correlated_uncertainty_raster is not None
        ):
            self.total_mean_uncertainty_raster = math.sqrt(
                self.mean_random_uncorrelated_raster ** 2
                + self.total_mean_correlated_uncertainty_raster ** 2
            )
        if (
            self.mean_random_uncorrelated_raster is not None
            and self.total_mean_correlated_uncertainty_min_raster is not None
        ):
            self.total_mean_uncertainty_min_raster = math.sqrt(
                self.mean_random_uncorrelated_raster ** 2
                + self.total_mean_correlated_uncertainty_min_raster ** 2
            )
        if (
            self.mean_random_uncorrelated_raster is not None
            and self.total_mean_correlated_uncertainty_max_raster is not None
        ):
            self.total_mean_uncertainty_max_raster = math.sqrt(
                self.mean_random_uncorrelated_raster ** 2
                + self.total_mean_correlated_uncertainty_max_raster ** 2
            )

    # ---------------------------------------------------------------------- #
    # Pretty-print results (matches your existing output style)
    # ---------------------------------------------------------------------- #
    def print_results(self) -> None:
        """
        Pretty-print stored results; omit lines for None values.
        """
        def _fmt(v, f=".4f"):
            return f"{v:{f}}" if v is not None else None

        def _triple(label, central, vmin, vmax, f=".4f"):
            pieces = []
            if central is not None:
                pieces.append(_fmt(central, f))
            if vmin is not None:
                pieces.append(f"min: {_fmt(vmin, f)}")
            if vmax is not None:
                pieces.append(f"max: {_fmt(vmax, f)}")
            if pieces:
                print(label + "; ".join(pieces))

        print("Variogram Analysis Results:")
        print(f"Ranges: {self.ranges}; min: {self.ranges_min}; max: {self.ranges_max}")
        print(f"Sills: {self.sills}; min: {self.sills_min}; max: {self.sills_max}")
        if self.best_model_params is not None:
            print(f"Best Model Parameters: {self.best_model_params}")

        print("\nUncertainty Results for Polygon of interest:")
        print(f"Polygon Area (m²): {self.area:.2f}")

        if self.mean_random_uncorrelated is not None:
            print(f"Mean Random Uncorrelated Uncertainty: {self.mean_random_uncorrelated:.4f}")

        _triple("Mean Random Correlated 1: ",
                getattr(self, "mean_random_correlated_1_polygon", None),
                getattr(self, "mean_random_correlated_1_min_polygon", None),
                getattr(self, "mean_random_correlated_1_max_polygon", None))
        _triple("Mean Random Correlated 2: ",
                getattr(self, "mean_random_correlated_2_polygon", None),
                getattr(self, "mean_random_correlated_2_min_polygon", None),
                getattr(self, "mean_random_correlated_2_max_polygon", None))
        _triple("Mean Random Correlated 3: ",
                getattr(self, "mean_random_correlated_3_polygon", None),
                getattr(self, "mean_random_correlated_3_min_polygon", None),
                getattr(self, "mean_random_correlated_3_max_polygon", None))

        _triple("Total Mean Correlated Uncertainty (Polygon): ",
                self.total_mean_correlated_uncertainty_polygon,
                self.total_mean_correlated_uncertainty_min_polygon,
                self.total_mean_correlated_uncertainty_max_polygon)

        _triple("Total Mean Uncertainty (Polygon): ",
                self.total_mean_uncertainty_polygon,
                self.total_mean_uncertainty_min_polygon,
                self.total_mean_uncertainty_max_polygon)

        print("\nUncertainty Results for Raster:")
        bbox_area = float(self.raster_data_handler.bbox.area) if getattr(self.raster_data_handler, "bbox", None) is not None else np.nan
        print(f"Coarse raster Area (m²): {bbox_area:.2f}")

        _triple("Mean Random Correlated 1 (Raster): ",
                getattr(self, "mean_random_correlated_1_raster", None),
                getattr(self, "mean_random_correlated_1_min_raster", None),
                getattr(self, "mean_random_correlated_1_max_raster", None))
        _triple("Mean Random Correlated 2 (Raster): ",
                getattr(self, "mean_random_correlated_2_raster", None),
                getattr(self, "mean_random_correlated_2_min_raster", None),
                getattr(self, "mean_random_correlated_2_max_raster", None))
        _triple("Mean Random Correlated 3 (Raster): ",
                getattr(self, "mean_random_correlated_3_raster", None),
                getattr(self, "mean_random_correlated_3_min_raster", None),
                getattr(self, "mean_random_correlated_3_max_raster", None))

        _triple("Total Mean Correlated Uncertainty (Raster): ",
                self.total_mean_correlated_uncertainty_raster,
                self.total_mean_correlated_uncertainty_min_raster,
                self.total_mean_correlated_uncertainty_max_raster)

        _triple("Total Mean Uncertainty (Raster): ",
                self.total_mean_uncertainty_raster,
                self.total_mean_uncertainty_min_raster,
                self.total_mean_uncertainty_max_raster)