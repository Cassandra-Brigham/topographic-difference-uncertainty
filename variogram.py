"""Analyze vertical differencing uncertainty using variogram methods.

Provides utilities for:
- Loading and sampling raster data for variogram analysis
- Fitting parametric semivariogram models (spherical, with/without nugget)
- Bootstrap resampling for parameter confidence intervals
- Efficient Numba kernels for pairwise distance calculations

Classes:
- RasterDataHandler: Load and sample raster data
- StatisticalAnalysis: Exploratory statistics and bootstrap uncertainty
- VariogramAnalysis: Compute empirical variograms and fit models
"""

from __future__ import annotations

import math
from typing import Sequence, Optional, Dict, Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import rasterio
import rioxarray as rio
from numba import njit, prange
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon, Point, box, shape
from shapely.prepared import prep
import geopandas as gpd
from rasterio.features import shapes
from shapely.ops import unary_union


class RasterDataHandler:
    """
    Load vertical differencing raster data, subtract a vertical systematic error
    from the raster, and randomly sample raster data for further analysis.

    Attributes
    ----------
    raster_path : str
        File path to the raster data.
    unit : str
        Unit of measurement for the raster data (for plotting labels).
    resolution : float
        Nominal raster resolution (linear units).
    rioxarray_obj : rioxarray.DataArray | None
        The rioxarray object holding the raster data.
    data_array : np.ndarray | None
        Loaded raster values as a 1D array of finite pixels.
    samples : np.ndarray | None
        Sampled values from the raster.
    coords : np.ndarray | None
        Coordinates (x, y) of the sampled values.
    bbox : shapely.geometry.Polygon
        Bounding box of the raster.
    """

    def __init__(self, raster_path: str, unit: str, resolution: float):
        self.raster_path = raster_path
        self.unit = unit
        self.resolution = resolution
        self.rioxarray_obj = None
        self.data_array = None
        self.samples = None
        self.coords = None
        self.shapely_geoms = None
        self.merged_geom = None
        self.detailed_area = None

        with rasterio.open(self.raster_path) as src:
            bounds = src.bounds
            self.bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            self.bbox = box(*self.bounds)

    def get_detailed_area(self) -> None:
        """
        Compute the precise area covered by valid data in the raster by vectorizing
        the finite/nodata mask into polygon shapes and dissolving them.
        """
        with rasterio.open(self.raster_path) as src:
            data = src.read(1).astype(float)
            nodata = src.nodata
            valid = (~np.isnan(data)) if nodata is None else ((data != nodata) & ~np.isnan(data))
            geoms = shapes(valid.astype(np.uint8), mask=valid, transform=src.transform)
        self.shapely_geoms = [shape(geom) for geom, val in geoms if val == 1]
        self.merged_geom = unary_union(self.shapely_geoms)
        self.detailed_area = self.merged_geom.area

    def load_raster(self, masked: bool = True) -> None:
        """
        Load raster data and store finite values in self.data_array.

        Parameters
        ----------
        masked : bool
            If True, open as masked and coerce mask to NaN.
        """
        
        da = rio.open_rasterio(self.raster_path, masked=masked)
        if "band" in da.dims and da.sizes.get("band", 1) == 1:
            da = da.squeeze("band", drop=True)
        arr = da.values
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        nodata = da.rio.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= (arr != nodata)
        self.rioxarray_obj = da
        self.data_array = np.asarray(arr[valid], dtype=float).ravel()

    def subtract_value_from_raster(self, output_raster_path: str, value_to_subtract: float) -> None:
        """
        Subtract a specified value from the raster and write a new file.

        Parameters
        ----------
        output_raster_path : str
            Path to the output raster.
        value_to_subtract : float
            Value to subtract from each valid pixel.
        """
        with rasterio.open(self.raster_path) as src:
            data = src.read()
            nodata = src.nodata
            mask = (data != nodata) if nodata is not None else np.ones(data.shape, dtype=bool)
            data = data.astype(float)
            data[mask] -= value_to_subtract
            out_meta = src.meta.copy()
            out_meta.update({'dtype': 'float32', 'nodata': nodata})
            with rasterio.open(output_raster_path, 'w', **out_meta) as dst:
                dst.write(data)

    def plot_raster(self, plot_title: str):
        """
        Plot the loaded rioxarray DataArray with a diverging colormap.

        Raises
        ------
        RuntimeError
            If raster has not been loaded yet.
        """
        
        if self.rioxarray_obj is None:
            raise RuntimeError("Raster not loaded. Call load_raster() first.")
        rio_data = self.rioxarray_obj
        fig, ax = plt.subplots(figsize=(10, 6))
        rio_data.plot(cmap="bwr_r", ax=ax, robust=True)
        ax.set_title(plot_title, pad=30)
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.ticklabel_format(style="plain")
        ax.set_aspect('equal')
        return fig

    def sample_raster(
        self,
        area_side: float,
        samples_per_area: float,
        max_samples: int,
        *,
        seed: Optional[int] = None
    ) -> None:
        """
        Randomly sample valid pixels from the raster, storing (values, coords).

        Parameters
        ----------
        area_side : float
            Reference side length used to convert pixel area to reference-area units
            (e.g., 1000 for km² if coordinates are meters).
        samples_per_area : float
            Number of samples to draw per unit of reference area.
        max_samples : int
            Maximum total samples to draw.
        seed : int | None
            RNG seed for reproducibility.

        Raises
        ------
        ValueError
            If requested samples exceed valid pixels, or computed total is < 1.
        """
        with rasterio.open(self.raster_path) as src:
            rng = np.random.default_rng(seed)

            data = src.read(1).astype(float)
            nodata = src.nodata
            valid = np.isfinite(data)
            if nodata is not None:
                valid &= (data != nodata)

            cell_area_m2 = abs(src.res[0] * src.res[1])
            valid_rows, valid_cols = np.where(valid)
            valid_count = valid_rows.size
            cell_area_in_reference = cell_area_m2 / (area_side ** 2)
            total_samples = min(int(cell_area_in_reference * samples_per_area * valid_count), max_samples)

            
            if total_samples < 1:
                raise ValueError("Computed total_samples < 1. Increase samples_per_area or max_samples.")

            if total_samples > valid_count:
                raise ValueError("Requested samples exceed valid pixel count. Reduce samples_per_area.")

            chosen = rng.choice(valid_count, size=total_samples, replace=False)
            rows = valid_rows[chosen]
            cols = valid_cols[chosen]
            samples = data[rows, cols]
            x_coords, y_coords = src.xy(rows, cols)
            coords = np.vstack([x_coords, y_coords]).T

            mask = np.isfinite(samples)
            self.samples = samples[mask]
            self.coords = coords[mask]



class StatisticalAnalysis:
    """
    Statistical utilities for exploratory plotting and bootstrap uncertainty of the median.
    """

    def __init__(self, raster_data_handler: RasterDataHandler):
        self.raster_data_handler = raster_data_handler

    def plot_data_stats(self, filtered: bool = True):
        """
        Plot histogram of raster values with basic statistics annotated.

        Parameters
        ----------
        filtered : bool
            If True, clip to 1st–99th percentiles for visualization only.

        Returns
        -------
        matplotlib.figure.Figure
        """
        data = self.raster_data_handler.data_array
        if data is None or len(data) == 0:
            raise ValueError("No data available to plot. Call load_raster() first.")

        mean = np.mean(data)
        median = np.median(data)
        # Mode on continuous data is often not meaningful; kept for completeness
        mode_result = stats.mode(data, nan_policy="omit", keepdims=False)
        mode_vals = np.atleast_1d(mode_result.mode).astype(float)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        p1 = np.percentile(data, 1)
        p99 = np.percentile(data, 99)
        minimum = np.min(data)
        maximum = np.max(data)

        if filtered:
            data = data[(data >= p1) & (data <= p99)]

        fig, ax = plt.subplots()
        ax.hist(data, bins=60, density=False, alpha=0.6, color='g')
        ax.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
        ax.axvline(median, color='b', linestyle='dashed', linewidth=1, label='Median')
        for i, m in enumerate(mode_vals):
            ax.axvline(m, color='purple', linestyle='dashed', linewidth=1,
                       label='Mode' if i == 0 else "_nolegend_")

        mode_str = ", ".join([f"{m:.3f}" for m in mode_vals])
        textstr = "\n".join((
            f"Mean: {mean:.3f}",
            f"Median: {median:.3f}",
            f"Mode(s): {mode_str}",
            f"Min: {minimum:.3f}  Max: {maximum:.3f}",
            f"Q1: {q1:.3f}  Q3: {q3:.3f}",
        ))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        ax.set_xlabel(f'Vertical Difference ({self.raster_data_handler.unit})')
        ax.set_ylabel('Count')
        ax.set_title('Histogram of differencing results with exploratory statistics')
        ax.legend()
        plt.tight_layout()
        return fig

    def bootstrap_uncertainty_subsample(self, n_bootstrap: int = 1000, subsample_proportion: float = 0.1) -> float:
        """
        Estimate uncertainty of the median via bootstrap on random subsamples.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap resamples.
        subsample_proportion : float
            Fraction of data per resample.

        Returns
        -------
        float
            Standard deviation of bootstrap medians.
        """
        data = self.raster_data_handler.data_array
        if data is None or len(data) == 0:
            raise ValueError("No data available for bootstrap. Call load_raster() first.")

        
        subsample_size = max(1, int(round(subsample_proportion * len(data))))
        rng = np.random.default_rng()
        bootstrap_medians = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(data, size=subsample_size, replace=True)
            bootstrap_medians[i] = np.median(sample)
        return float(np.std(bootstrap_medians))


class VariogramAnalysis:
    """
    Compute empirical variograms across multiple random samples, fit spherical
    models (with optional nugget), and bootstrap parameter uncertainty.
    """

    MIN_PAIRS = 10

    def __init__(self, raster_data_handler: RasterDataHandler):
        self.raster_data_handler = raster_data_handler
        self.mean_variogram = None
        self.lags = None
        self.mean_count = None
        self.err_variogram = None
        self.fitted_variogram = None
        self.rmse = None
        self.sills = None
        self.ranges = None
        self.ranges_min = None
        self.ranges_max = None
        self.ranges_median = None
        self.err_sills = None
        self.err_ranges = None
        self.sills_min = None
        self.sills_max = None
        self.sills_median = None
        self.best_nugget = None
        self.min_nugget = None
        self.max_nugget = None
        self.median_nugget = None
        self.best_aic = None
        self.best_params = None
        self.best_model_config = None
        self.cv_mean_error_best_aic = None
        self.param_samples = None
        self.n_bins = None
        self.sigma_variogram = None
        self.best_model_func = None
        self.best_guess = None
        self.best_bounds = None
        self.all_variograms = None
        self.all_counts = None

    
    
    @staticmethod
    @njit(parallel=True)
    def bin_distances_and_squared_differences(coords, values, bin_width, max_lag_multiplier, x_extent, y_extent):
        """
        Compute and bin pairwise distances and squared differences for Matheron estimation.

        Parameters:
        -----------
        coords : np.ndarray
            Array of coordinates of shape (M, 2).
        values : np.ndarray
            Array of values of shape (M,).
        bin_edges : np.ndarray
            Array of bin edges for distance binning.

        Returns:
        --------
        bin_counts : np.ndarray
            Counts of pairs in each bin.
        binned_sum_squared_diff : np.ndarray
            Sum of squared differences for each bin.
        """
        approx_max_distance = np.sqrt(x_extent**2 + y_extent**2)
        
        if max_lag_multiplier == "max":
            max_lag = approx_max_distance
        elif max_lag_multiplier == "median":
            max_lag = 0.5 * approx_max_distance  # simple heuristic
        else:
            max_lag = float(approx_max_distance * max_lag_multiplier)
        
        # Determine bin edges using diagonal distance as maximum lag
        n_bins = int(np.ceil(max_lag / bin_width)) + 1
        bin_edges = np.arange(0, n_bins * bin_width, bin_width)
        
        M = coords.shape[0]
        max_distance = 0.0
        bin_counts = np.zeros(n_bins, dtype=np.int64)
        binned_sum_squared_diff = np.zeros(n_bins, dtype=np.float64)

        for i in prange(M):
            for j in range(i + 1, M):
                # Compute the pairwise distance
                d = 0.0
                for k in range(coords.shape[1]):
                    tmp = coords[i, k] - coords[j, k]
                    d += tmp * tmp
                dist = np.sqrt(d)
                max_distance = max(max_distance, dist)
                
                # Compute the difference
                diff = values[i] - values[j]
                
                # Compute the squared difference
                diff_squared = (diff) ** 2

                # Find the bin for this distance
                bin_idx = int(dist / bin_width)
                if 0 <= bin_idx < n_bins:
                    bin_counts[bin_idx] += 1
                    binned_sum_squared_diff[bin_idx] += diff_squared
        
        
        bin_edges = bin_edges[:n_bins]
        bin_counts = bin_counts[:n_bins]
        binned_sum_squared_diff = binned_sum_squared_diff[:n_bins]

        return n_bins, bin_counts, binned_sum_squared_diff, max_distance, max_lag

    @staticmethod
    def compute_matheron(bin_counts, ssd, min_pairs: int = 10) -> np.ndarray:
        """
        Compute Matheron semivariance γ(h) = SSD(h) / (2 N(h)) for bins with >= min_pairs.
        """
        gamma_est = np.full_like(bin_counts, np.nan, dtype=float)
        for i, (cnt, sum_sq) in enumerate(zip(bin_counts, ssd)):
            if cnt >= min_pairs:
                gamma_est[i] = sum_sq / (2.0 * cnt)
        return gamma_est

    def numba_variogram(
        self,
        area_side: float,
        samples_per_area: float,
        max_samples: int,
        bin_width: float,
        max_lag_multiplier,
        *,
        seed: Optional[int] = None
    ):
        """
        Compute one empirical variogram by sampling the raster and binning pairwise
        squared differences of values by distance.

        Returns
        -------
        bin_counts : np.ndarray
        variogram_matheron : np.ndarray
        n_bins : int
        min_distance : float
        max_distance : float
        max_lag : float
        """
        self.raster_data_handler.sample_raster(area_side, samples_per_area, max_samples, seed=seed)

        min_distance = 0.0  # retained for compatibility
        xs = self.raster_data_handler.rioxarray_obj.x.values
        ys = self.raster_data_handler.rioxarray_obj.y.values
        x_extent = float(np.max(xs) - np.min(xs))
        y_extent = float(np.max(ys) - np.min(ys))

        n_bins, bin_counts, bssd, max_distance, max_lag = self.bin_distances_and_squared_differences(
            self.raster_data_handler.coords,
            self.raster_data_handler.samples,
            bin_width,
            max_lag_multiplier,
            x_extent,
            y_extent
        )
        matheron_estimates = self.compute_matheron(bin_counts, bssd, min_pairs=self.MIN_PAIRS)
        return bin_counts, matheron_estimates, n_bins, min_distance, max_distance, max_lag

    def calculate_mean_variogram_numba(
        self,
        area_side: float,
        samples_per_area: float,
        max_samples: int,
        bin_width: float,
        max_n_bins: int,
        n_runs: int,
        max_lag_multiplier=1 / 3,
        *,
        seed: Optional[int] = None
    ) -> None:
        """
        Run multiple variogram realizations and compute the mean semivariogram
        and spread across runs.

        Parameters
        ----------
        area_side, samples_per_area, max_samples : see numba_variogram
        bin_width : float
        max_n_bins : int
        n_runs : int
        max_lag_multiplier : {"max","median"} or float
        seed : int | None
            Base seed; each run uses a child seed for reproducibility.
        """
        # Child seeds for each run to keep realizations independent but reproducible.
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(n_runs)

        all_variograms = pd.DataFrame(np.nan, index=range(n_runs), columns=range(max_n_bins))
        counts = pd.DataFrame(np.nan, index=range(n_runs), columns=range(max_n_bins))
        all_n_bins = np.zeros(n_runs, dtype=int)

        for run in range(n_runs):
            count, variogram, n_bins, _, _, _ = self.numba_variogram(
                area_side, samples_per_area, max_samples, bin_width, max_lag_multiplier,
                seed=int(child_seeds[run].generate_state(1)[0])
            )
            all_variograms.loc[run, :variogram.size - 1] = variogram
            counts.loc[run, :count.size - 1] = count
            all_n_bins[run] = n_bins

        vario_arr = all_variograms.values
        count_arr = counts.values

        with np.errstate(all='ignore'):
            mean_variogram = np.nanmean(vario_arr, axis=0)
            # use robust spread visualization width; stored as err_variogram
            err_variogram = (np.nanpercentile(vario_arr, 97.5, axis=0) -
                             np.nanpercentile(vario_arr, 2.5, axis=0)) / 2.0
            mean_count = np.nanmean(count_arr, axis=0)
            sigma_variogram = np.nanstd(vario_arr, axis=0)

        
        sigma_filtered = sigma_variogram.copy()
        sigma_filtered[sigma_filtered == 0] = np.finfo(float).eps

        valid = ~np.isnan(mean_variogram)
        self.mean_variogram = mean_variogram[valid]
        self.err_variogram = err_variogram[valid]
        self.mean_count = mean_count[valid]
        self.sigma_variogram = sigma_filtered[valid]

        n_kept = self.mean_variogram.size
        self.lags = np.linspace(bin_width / 2, bin_width * n_kept - bin_width / 2, n_kept)

        self.all_variograms = vario_arr
        self.all_counts = count_arr
        self.n_bins = int(np.nanmean(all_n_bins))

    @staticmethod
    def get_base_initial_guess(n: int, mean_variogram, lags, nugget: bool = False) -> np.ndarray:
        """
        Naive initial guess: equal sills; ranges spread linearly to max lag; optional nugget (last).
        """
        max_semivariance = np.max(mean_variogram) * 1.5
        half_max_lag = np.max(lags) / 2
        C = [max_semivariance / n] * n
        a = [((half_max_lag) / 3) * (i + 1) for i in range(n)]
        p0 = C + a + ([max_semivariance / 4] if nugget else [])
        return np.array(p0, dtype=float)

    @staticmethod
    def pure_nugget_model(h, nugget):
        """γ(h) for pure nugget: constant variance independent of h."""
        return np.full_like(h, nugget)

    @staticmethod
    def spherical_model(h, *params):
        """
        Multi-component spherical model without nugget.

        Parameters
        ----------
        h : array-like
        params : [C1..Cn, a1..an] (n sills followed by n ranges)
        """
        n = len(params) // 2
        C = params[:n]
        a = params[n:]
        model = np.zeros_like(h, dtype=float)
        for i in range(n):
            ai = a[i]
            Ci = C[i]
            mask = h <= ai
            ratio = h[mask] / ai
            model[mask] += Ci * (1.5 * ratio - 0.5 * ratio ** 3)
            model[~mask] += Ci
        return model

    def spherical_model_with_nugget(self, h, *params):
        """
        Spherical model with nugget at the END of the parameter vector.

        Parameters
        ----------
        params : [C1..Cn, a1..an, nugget]
        """
        nugget = params[-1]
        structural = params[:-1]
        return nugget + self.spherical_model(h, *structural)

    @staticmethod
    def bootstrap_fit_variogram(
        lags: np.ndarray,
        mean_vario: np.ndarray,
        sigma_vario: np.ndarray,
        model_func: Callable,
        p0: np.ndarray,
        bounds: Optional[tuple] = None,
        n_boot: int = 100,
        maxfev: int = 10000,
        *,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Parametric bootstrap for parameter uncertainty using per-lag standard deviations.

        Parameters
        ----------
        lags, mean_vario, sigma_vario : arrays
        model_func : callable
        p0 : initial params
        bounds : bounds tuple
        n_boot : int
        maxfev : int
        seed : int | None

        Returns
        -------
        np.ndarray
            Successful parameter vectors (rows).
        """
        rng = np.random.default_rng(seed)
        # Draw synthetic variograms with Gaussian noise per bin using sigma_vario.
        noise_array = rng.normal(loc=mean_vario, scale=np.where(sigma_vario > 0, sigma_vario, 0.0), size=(n_boot, len(mean_vario)))

        param_samples = []
        n_params = len(p0)
        bnds = bounds if bounds is not None else (-np.inf, np.inf)

        for n in range(n_boot):
            synth = noise_array[n, :]
            try:
                popt_synth, _ = curve_fit(
                    model_func,
                    lags,
                    synth,
                    p0=p0,
                    sigma=None,        # Using unconditional fits to the synthetic draws
                    bounds=bnds,
                    maxfev=maxfev
                )
                param_samples.append(popt_synth)
            except RuntimeError:
                param_samples.append([np.nan] * n_params)

        param_samples = np.array(param_samples)
        valid = ~np.isnan(param_samples).any(axis=1)
        return param_samples[valid]

    @staticmethod
    def _weighted_loglike_gaussian(y, yhat, sigma):
        """
        Log-likelihood for heteroscedastic Gaussian errors: sum over bins of
        -0.5 * [log(2π σ_i^2) + ((y_i - ŷ_i)^2 / σ_i^2)]
        """
        s = np.asarray(sigma, dtype=float)
        s = np.where(s <= 0, np.finfo(float).eps, s)
        resid = y - yhat
        return -0.5 * np.sum(np.log(2 * np.pi * s ** 2) + (resid ** 2) / (s ** 2))

    def cross_validate_variogram(self, model_func, p0, bounds, k: int = 5, *, seed: Optional[int] = None):
        """
        k-fold cross-validation on (lags, mean_variogram) with provided model/bounds.

        Returns
        -------
        dict | None
            {'rmse','mae','me','mse'} averaged across folds, or None if all folds fail.
        """
        rng = np.random.default_rng(seed)
        lags = self.lags
        mean_variogram = self.mean_variogram
        sigma_filtered = self.sigma_variogram

        n_bins = len(lags)
        indices = rng.permutation(n_bins)
        fold_size = max(1, n_bins // k)
        rmses, maes, mes, mses = [], [], [], []

        for i in range(k):
            valid_idx = indices[i * fold_size: (i + 1) * fold_size]
            train_idx = np.setdiff1d(indices, valid_idx)

            lags_train = lags[train_idx]
            vario_train = mean_variogram[train_idx]
            sigma_train = sigma_filtered[train_idx]

            try:
                popt, _ = curve_fit(model_func, lags_train, vario_train, p0=p0, bounds=bounds, sigma=sigma_train, absolute_sigma=True, maxfev=10000)
            except RuntimeError:
                continue

            lags_valid = lags[valid_idx]
            vario_valid = mean_variogram[valid_idx]
            predictions = model_func(lags_valid, *popt)

            errors = vario_valid - predictions
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            mae = float(np.mean(np.abs(errors)))
            me = float(np.mean(errors))
            mse = float(np.mean(errors ** 2))

            rmses.append(rmse)
            maes.append(mae)
            mes.append(me)
            mses.append(mse)

        if not rmses:
            return None

        return {'rmse': np.mean(rmses), 'mae': np.mean(maes), 'me': np.mean(mes), 'mse': np.mean(mses)}

    def fit_best_spherical_model(
        self,
        sigma_type: str = 'std',
        bounds: Optional[tuple] = None,
        method: str = 'trf',
        *,
        seed: Optional[int] = None
    ) -> None:
        """
        Fit spherical variogram models (1–3 components, optional nugget) and select the best by AIC.

        Parameters
        ----------
        sigma_type : {'std','linear','exp','sqrt','sq'}
            Bin weighting scheme used in curve_fit as sigma.
        bounds : tuple | None
            Optional (lb, ub) for parameters; if None, internal bounds are used.
        method : str
            curve_fit method (default 'trf').
        seed : int | None
            RNG seed for randomized initial guesses.

        Notes
        -----
        - Nugget parameter is ALWAYS last in the parameter vector.
        """
        rng = np.random.default_rng(seed)

        if self.all_variograms is None:
            raise RuntimeError("No variogram data. Call calculate_mean_variogram_numba() first.")

        # choose weights
        if sigma_type == 'std':
            sigma = self.sigma_variogram
        elif sigma_type == 'linear':
            sigma = 1.0 / (1.0 + self.lags)
        elif sigma_type == 'exp':
            sigma = np.exp(-self.lags)
        elif sigma_type == 'sqrt':
            sigma = 1.0 / np.sqrt(1.0 + self.lags)
        elif sigma_type == 'sq':
            sigma = 1.0 / (1.0 + self.lags ** 2)
        else:
            raise ValueError(f"Unknown sigma_type '{sigma_type}'. Use 'std', 'linear', 'exp', 'sqrt', or 'sq'.")

        best_params = None
        best_model = None
        best_func = None
        best_aic = np.inf
        best_bounds = None
        best_guess = None

        lag_max = float(np.max(self.lags)) if self.lags is not None and len(self.lags) else 1.0
        for config in (
            {'components': 1, 'nugget': False},
            {'components': 1, 'nugget': True},
            {'components': 2, 'nugget': False},
            {'components': 2, 'nugget': True},
            {'components': 3, 'nugget': False},
            {'components': 3, 'nugget': True},
        ):
            n = config['components']
            nugget = config['nugget']
            if n == 0:
                model = self.pure_nugget_model
                auto_bounds = ([0.0], [np.inf])
                p0s = [np.array([np.max(self.mean_variogram)])]
            else:
                model = self.spherical_model_with_nugget if nugget else self.spherical_model
                
                lb = [0.0] * n + [1e-6] * n + ([0.0] if nugget else [])
                ub = [np.inf] * n + [2.0 * lag_max] * n + ([np.inf] if nugget else [])
                auto_bounds = (lb, ub)

                base = self.get_base_initial_guess(n, self.mean_variogram, self.lags, nugget)
                p0s = []
                for _ in range(5):
                    perturb = (rng.random(len(base)) - 0.5) * 2.0 * 0.5  # +/-50%
                    guess = np.clip(base * (1 + perturb), 1e-6, None)
                    p0s.append(guess)

            bounds_tuple = bounds if bounds is not None else auto_bounds

            for p0 in p0s:
                try:
                    popt, _ = curve_fit(
                        model,
                        self.lags,
                        self.mean_variogram,
                        p0=p0,
                        sigma=sigma,
                        absolute_sigma=True,  
                        bounds=bounds_tuple,
                        method=method,
                        maxfev=20000
                    )
                except RuntimeError:
                    continue

                yhat = model(self.lags, *popt)
                ll = self._weighted_loglike_gaussian(self.mean_variogram, yhat, sigma)
                k = len(popt)
                aic = 2 * k - 2 * ll

                if aic < best_aic:
                    best_aic = aic
                    best_params = popt
                    best_model = config
                    best_func = model
                    best_bounds = bounds_tuple
                    best_guess = p0

        if best_params is None:
            raise RuntimeError("No valid model fit found. Check input data for NaN values.")

        self.best_params = best_params
        self.best_model_config = best_model
        self.best_model_func = best_func
        self.best_aic = best_aic
        self.best_bounds = best_bounds
        self.best_guess = best_guess
        self.fitted_variogram = (
            self.spherical_model_with_nugget if self.best_model_config['nugget']
            else self.spherical_model
        )(self.lags, *self.best_params)

        # Extract sill & range point estimates; nugget last if present
        n = self.best_model_config['components']
        if self.best_model_config['nugget']:
            self.sills = self.best_params[:n]
            self.ranges = self.best_params[n:2 * n]
            self.best_nugget = float(self.best_params[-1])
        else:
            self.sills = self.best_params[:n]
            self.ranges = self.best_params[n:2 * n]
            self.best_nugget = None

        # Prepare bounds for bootstrap consistent with nugget-last convention
        if n == 0:
            bounds_boot = ([0.0], [np.inf])
        else:
            lb = [0.0] * n + [1e-6] * n + ([0.0] if self.best_model_config['nugget'] else [])
            ub = [np.inf] * n + [2.0 * lag_max] * n + ([np.inf] if self.best_model_config['nugget'] else [])
            bounds_boot = (lb, ub)

        # Parametric bootstrap using per-bin sigma (std across runs)
        samples = self.bootstrap_fit_variogram(
            self.lags,
            self.mean_variogram,
            self.sigma_variogram,  
            self.best_model_func,
            self.best_params,
            bounds=bounds_boot,
            n_boot=500,
            maxfev=20000,
            seed=seed,
        )
        self.param_samples = samples

        # Percentiles of parameters
        if samples.size:
            if self.best_model_config['nugget']:
                nug_samps = samples[:, -1]
                samp = samples[:, :-1]
            else:
                nug_samps = None
                samp = samples

            sill_samps = samp[:, :n]
            range_samps = samp[:, n:2 * n]

            self.sills_min = np.percentile(sill_samps, 16, axis=0)
            self.sills_max = np.percentile(sill_samps, 84, axis=0)
            self.sills_median = np.percentile(sill_samps, 50, axis=0)

            self.ranges_min = np.percentile(range_samps, 16, axis=0)
            self.ranges_max = np.percentile(range_samps, 84, axis=0)
            self.ranges_median = np.percentile(range_samps, 50, axis=0)

            if nug_samps is not None:
                self.min_nugget = float(np.percentile(nug_samps, 16))
                self.max_nugget = float(np.percentile(nug_samps, 84))
                self.median_nugget = float(np.percentile(nug_samps, 50))
            else:
                self.min_nugget = self.max_nugget = self.median_nugget = None
        else:
            # fallback to point estimates
            self.sills_min = self.sills_max = self.sills_median = np.array(self.sills)
            self.ranges_min = self.ranges_max = self.ranges_median = np.array(self.ranges)
            if self.best_model_config['nugget']:
                self.min_nugget = self.max_nugget = self.median_nugget = self.best_nugget
            else:
                self.min_nugget = self.max_nugget = self.median_nugget = None

        # Compute and store cross-validation metrics on best model
        self.cv_mean_error_best_aic = self.cross_validate_variogram(
            self.best_model_func, self.best_params, self.best_bounds, k=5, seed=seed
        )

    def plot_best_spherical_model(self):
        """
        Plot mean variogram ± spread and fitted model; also show bar plot of mean pair counts.
        """
        if any(attr is None for attr in (self.mean_variogram, self.err_variogram, self.mean_count, self.lags, self.fitted_variogram)):
            raise RuntimeError("Missing variogram data. Call calculate_mean_variogram_numba() and fit_best_spherical_model() first.")

        n = min(len(self.lags), len(self.mean_variogram), len(self.err_variogram), len(self.fitted_variogram))
        lags = self.lags[:n]
        gamma = self.mean_variogram[:n]
        errs = self.err_variogram[:n]
        model = self.fitted_variogram[:n]
        counts = self.mean_count[:n]

        valid_counts = (~np.isnan(counts)) & (counts > 0)
        count_lags = lags[valid_counts]
        count_vals = counts[valid_counts]

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(10, 8), sharex=True)

        # guard single-bin bar width
        if len(lags) > 1:
            bar_width = (lags[1] - lags[0]) * 0.9
        else:
            bar_width = (lags[0] if len(lags) else 1.0) * 0.9
        axs[0].bar(count_lags, count_vals, width=bar_width, color='orange', alpha=0.5)
        axs[0].set_ylabel('Mean Count')
        axs[0].tick_params(labelbottom=False)

        axs[1].errorbar(lags, gamma, yerr=errs, fmt='o-', color='blue', label='Mean Variogram ± spread')
        axs[1].plot(lags, model, 'r-', label='Fitted Model')

        colors = ['red', 'green', 'blue']
        if self.ranges is not None and self.ranges_min is not None and self.ranges_max is not None:
            ylim = axs[1].get_ylim()
            for i, (r, rmin, rmax) in enumerate(zip(self.ranges, self.ranges_min, self.ranges_max)):
                c = colors[i % len(colors)]
                axs[1].axvline(r, color=c, linestyle='--', linewidth=1, label=f'Range {i + 1}')
                axs[1].fill_betweenx(ylim, rmin, rmax, color=c, alpha=0.2)

        if self.best_nugget is not None and self.min_nugget is not None and self.max_nugget is not None:
            axs[1].axhline(self.best_nugget, color='black', linestyle='--', linewidth=1, label='Nugget')
            axs[1].fill_between(lags, [self.min_nugget] * len(lags), [self.max_nugget] * len(lags), color='gray', alpha=0.2)

        axs[1].set_xlabel('Lag Distance')
        axs[1].set_ylabel('Semivariance')
        axs[1].legend(loc='upper right')

        rmse_str = ""
        if isinstance(self.cv_mean_error_best_aic, dict):
            rmse = self.cv_mean_error_best_aic.get('rmse', None)
            if rmse is not None:
                rmse_str = f'RMSE (CV): {rmse:.4f}'
        axs[1].set_title(rmse_str)
        plt.setp(axs[0].get_xticklabels(), visible=False)
        plt.tight_layout()
        return fig


#class RegionalUncertaintyEstimator:
