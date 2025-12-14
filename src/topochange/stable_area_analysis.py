"""Define and analyze stable/unstable areas for topographic differencing.

Provides:
- Interactive mapping widgets for drawing polygons on difference rasters
- Rasterizing polygons against raster masks
- Computing descriptive statistics within regions

Designed for use in Jupyter notebook environments.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, List, Optional, Iterable
import uuid
import statistics
import base64

import numpy as np
import pandas as pd
import rasterio
import rasterio.features as rfeatures
from rasterio.features import rasterize
import rasterio.mask
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform, unary_union
from pyproj import Transformer
import matplotlib.pyplot as plt
from ipyleaflet import Map, DrawControl, ImageOverlay, GeoJSON, WidgetControl, basemaps, ScaleControl
try:
    from ipyleaflet import LegendControl
    HAS_LEGEND = True
except Exception:
    HAS_LEGEND = False
from ipywidgets import Button, HBox, Label, Layout
from scipy import stats


class TopoMapInteractor:
    """
    Interactive map for drawing 'stable' and 'unstable' polygons on a topo-difference raster,
    with pixel-count utility, two-layer legend, labeled draw buttons, file loading, and
    optional auto-derivation of Stable = (valid raster) − (union of FOIs).
    """
    def __init__(
        self,
        topo_diff_path: Path | str,
        hillshade_path: Path | str,
        output_dir: Path | str,
        zoom: int = 15,
        map_size: tuple[str, str] = ('800px', '1300px'),
        overlay_cmap: str = 'bwr_r',
        overlay_dpi: int = 300,
        overlay_vmin: Optional[float] = None,
        overlay_vmax: Optional[float] = None,
        overlay_format: str = 'png',
        stable_path: Optional[Path | str] = None,
        unstable_path: Optional[Path | str] = None,
        stable_name_field: Optional[str] = None,
        unstable_name_field: Optional[str] = None,
        assume_input_crs: Optional[str] = 'EPSG:4326',
        auto_stable_from_unstable: bool = True,
        derive_min_area: Optional[float] = None,
        simplify_tolerance: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        topo_diff_path, hillshade_path, output_dir : Path-like
            Input rasters and output directory for overlay PNGs.
        zoom, map_size : map appearance.
        overlay_cmap, overlay_dpi, overlay_vmin, overlay_vmax : overlay styling.
            If only one of vmin/vmax is provided, the other is inferred symmetrically.
        stable_path, unstable_path : Path-like or None
            Optional polygon files to preload (shapefile/.zip, GeoPackage, GeoJSON).
        stable_name_field, unstable_name_field : str or None
            Optional attribute name for feature labels when loading files.
        assume_input_crs : str or None
            CRS to assume when an input vector file lacks a CRS (default 'EPSG:4326').
        auto_stable_from_unstable : bool
            If True, automatically set Stable = valid raster minus FOIs whenever FOIs change.
        derive_min_area : float or None
            Minimum polygon area (in raster CRS units²) to keep when deriving Stable.
        simplify_tolerance : float or None
            Optional simplification tolerance (in raster CRS units) when deriving Stable.
        """
        # Import Raster from project's raster module
        # Note: This assumes the module is importable. In actual project use:
        # from .raster import Raster
        # For standalone use, adjust the import path accordingly
        try:
            from raster import Raster
        except ImportError:
            # Fallback for when used within a package
            from .raster import Raster
        
        # Load rasters
        self.topo_diff = Raster.from_file(str(topo_diff_path))
        self.hillshade = Raster.from_file(str(hillshade_path))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for geometries and names
        self.stable_geoms: List[Polygon] = []
        self.unstable_geoms: List[Polygon] = []
        self.stable_names: List[str] = []
        self.unstable_names: List[str] = []
        self.current_category: Optional[str] = None

        # Auto-derive settings
        self.auto_stable_from_unstable = auto_stable_from_unstable
        self.derive_min_area = derive_min_area
        self.simplify_tolerance = simplify_tolerance

        # Track whether user has manually defined stable areas
        # (if True, don't auto-derive stable from unstable)
        self._user_drew_stable = False

        # Compute lat/lon bounds
        with rasterio.open(self.topo_diff.filename) as ds:
            bounds = ds.bounds
            crs = ds.crs
        transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        west, south = transformer.transform(bounds.left, bounds.bottom)
        east, north = transformer.transform(bounds.right, bounds.top)
        self.latlon_bounds = ((south, west), (north, east))

        # Overlay style
        self.overlay_cmap = overlay_cmap
        self.overlay_dpi = overlay_dpi
        self.overlay_vmin = overlay_vmin
        self.overlay_vmax = overlay_vmax
        self.overlay_format = overlay_format

        # Initial overlay image (cache-busted filename)
        self._overlay_image = self._new_overlay_image_path(format=overlay_format)
        self._generate_overlay_image(
            self._overlay_image,
            cmap=self.overlay_cmap,
            dpi=self.overlay_dpi,
            vmin=self.overlay_vmin,
            vmax=self.overlay_vmax,
            format=overlay_format,
        )
        self._overlay_data_url = self._image_to_data_url(self._overlay_image)

        # Initialize map
        center = ((north + south) / 2, (west + east) / 2)
        self.map = Map(center=center, zoom=zoom, layout=Layout(height=map_size[0], width=map_size[1]))

        # Add image overlay - try both data URL and file path
        # Some environments prefer file:// URLs
        print(f"Map bounds (lat/lon): {self.latlon_bounds}", file=sys.stderr)
        print(f"Map center: {center}", file=sys.stderr)
        print(f"Using overlay: {self._overlay_image}", file=sys.stderr)

        self.overlay_layer = ImageOverlay(url=self._overlay_data_url, bounds=self.latlon_bounds, opacity=1.0)
        self.map.add_layer(self.overlay_layer)

        # Legend
        if HAS_LEGEND:
            legend_dict = {'Stable Area': 'green', 'Feature of Interest': 'red'}
            self.map.add_control(LegendControl(legend_dict, title='Legend'))

        # GeoJSON layers
        self.geojson_stable = GeoJSON(
            data={"type": "FeatureCollection", "features": []},
            style={"color": "green", "fillColor": "green", "fillOpacity": 0.3}
        )
        self.geojson_unstable = GeoJSON(
            data={"type": "FeatureCollection", "features": []},
            style={"color": "red", "fillColor": "red", "fillOpacity": 0.3}
        )
        self.map.add_layer(self.geojson_stable)
        self.map.add_layer(self.geojson_unstable)

        # Draw control
        self.draw_control = DrawControl(polygon={"shapeOptions": {"weight": 2, "fillOpacity": 0.3}})
        for attr in ('circle', 'circlemarker', 'polyline', 'rectangle'):
            setattr(self.draw_control, attr, {})
        self.draw_control.on_draw(self._handle_draw)

        # Buttons
        self.btn_stable = Button(description='Stable', layout={'width': '80px'})
        self.btn_unstable = Button(description='Unstable', layout={'width': '80px'})
        self.btn_stable.style.button_color = 'lightgreen'
        self.btn_unstable.style.button_color = 'lightcoral'
        self.btn_stable.on_click(lambda _: self._activate_category('stable'))
        self.btn_unstable.on_click(lambda _: self._activate_category('unstable'))
        btn_box = HBox([Label(' Draw mode:'), self.btn_stable, self.btn_unstable])
        self.map.add_control(WidgetControl(widget=btn_box, position='topright'))

        # Preload polygon files (if provided)
        if stable_path is not None:
            self.load_stable_polygons(stable_path, name_field=stable_name_field, assume_crs=assume_input_crs)
        if unstable_path is not None:
            self.load_unstable_polygons(unstable_path, name_field=unstable_name_field, assume_crs=assume_input_crs)
            # Only auto-derive stable if user hasn't provided stable areas
            if self.auto_stable_from_unstable and not self._user_drew_stable:
                self.derive_stable_from_unstable(replace=True)

    # -------------------- Overlay helpers --------------------
    def _new_overlay_image_path(self, format: str = 'png') -> Path:
        stem = Path(self.topo_diff.filename).stem
        ext = 'jpg' if format == 'jpeg' else format
        return self.output_dir / f"{stem}-{uuid.uuid4().hex}.{ext}"

    def _image_to_data_url(self, path: Path | str) -> str:
        """Return a base64 data URL for an image file (works in Colab)."""
        path = Path(path)
        ext = path.suffix.lower()

        # Determine MIME type
        if ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        elif ext == '.png':
            mime_type = 'image/png'
        else:
            mime_type = 'image/png'  # default

        with open(path, "rb") as f:
            b = f.read()

        data_url = f"data:{mime_type};base64," + base64.b64encode(b).decode("ascii")
        print(f"Generated data URL: {len(data_url)} chars ({len(b)} bytes)", file=sys.stderr)
        return data_url

    def _generate_overlay_image(
        self,
        image_path: Path | str,
        *,
        cmap: str = 'bwr_r',
        dpi: int = 300,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        format: str = 'png',
    ):
        """Generate an overlay image (PNG or JPEG) of the topo_diff raster."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        # Read raster data
        with rasterio.open(self.topo_diff.filename) as src:
            data = src.read(1).astype(float)
            nodata = src.nodata
            height, width = data.shape

        print(f"Overlay: shape={data.shape}, nodata={nodata}", file=sys.stderr)

        # Mask nodata and invalid values
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        data = np.where(~np.isfinite(data), np.nan, data)

        # Check if we have any valid data
        valid_count = np.sum(~np.isnan(data))
        print(f"Valid pixels: {valid_count:,} / {data.size:,} ({100*valid_count/data.size:.1f}%)", file=sys.stderr)

        if np.all(np.isnan(data)):
            print("[WARNING] All data values are NaN/nodata. Overlay will be invisible.", file=sys.stderr)

        # Determine color limits from valid data only
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            if vmin is None and vmax is None:
                absmax = np.nanmax(np.abs(data))
                vmin_val: float = -absmax
                vmax_val: float = absmax
            elif vmin is None:
                vmin_val = -abs(float(vmax)) if vmax is not None else 1.0
                vmax_val = float(vmax) if vmax is not None else 1.0
            elif vmax is None:
                vmax_val = abs(float(vmin)) if vmin is not None else 1.0
                vmin_val = float(vmin) if vmin is not None else 1.0
            else:
                vmin_val = float(vmin)
                vmax_val = float(vmax)

            print(f"Color range: [{vmin_val:.3f}, {vmax_val:.3f}]", file=sys.stderr)
            print(f"Data range: [{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]", file=sys.stderr)
        else:
            vmin_val, vmax_val = -1.0, 1.0  # Fallback if no valid data
            print(f"No valid data. Using fallback color range [{vmin_val}, {vmax_val}].", file=sys.stderr)

        # For JPEG, we need a background color (no transparency)
        # For PNG, we can use transparency
        if format.lower() in ['jpg', 'jpeg']:
            use_transparent = False
            facecolor = 'white'
        else:
            use_transparent = True
            facecolor = 'none'

        # Create figure - simpler approach with direct pixel mapping
        fig_width_inch = width / dpi
        fig_height_inch = height / dpi

        fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)
        ax.set_position([0, 0, 1, 1])  # Fill entire figure
        ax.set_axis_off()

        # Plot with colormap
        cm = plt.get_cmap(cmap)
        if use_transparent:
            cm.set_bad(alpha=0)  # Transparent for NaN (PNG only)
        else:
            cm.set_bad(color='white')  # White background for JPEG

        im = ax.imshow(data, cmap=cm, vmin=vmin_val, vmax=vmax_val,
                       interpolation='nearest', aspect='auto', origin='upper')

        # Save image
        save_kwargs = {
            'format': 'jpeg' if format.lower() in ['jpg', 'jpeg'] else 'png',
            'bbox_inches': 'tight',
            'pad_inches': 0,
            'dpi': dpi,
            'facecolor': facecolor,
        }

        if use_transparent:
            save_kwargs['transparent'] = True

        # Note: quality parameter is handled differently for JPEG in matplotlib
        # It's passed via pil_kwargs, not directly
        if format.lower() in ['jpg', 'jpeg']:
            save_kwargs['pil_kwargs'] = {'quality': 95}

        fig.savefig(str(image_path), **save_kwargs)
        plt.close(fig)

        print(f"Saved overlay: {image_path}", file=sys.stderr)

    # -------------------- Category activation --------------------
    def _activate_category(self, category: str):
        """Set the current drawing category."""
        self.current_category = category
        if category == 'stable':
            self.draw_control.polygon = {"shapeOptions": {"color": "green", "fillColor": "green", "fillOpacity": 0.3}}
        else:
            self.draw_control.polygon = {"shapeOptions": {"color": "red", "fillColor": "red", "fillOpacity": 0.3}}
        
        # Add draw control if not already added
        if self.draw_control not in self.map.controls:
            self.map.add_control(self.draw_control)

    # -------------------- Draw handler --------------------
    def _handle_draw(self, target, action, geo_json):
        """Handle polygon drawing events."""
        if action != 'created' or self.current_category is None:
            return
        
        # Extract polygon from GeoJSON
        geom = shape(geo_json['geometry'])
        
        # Transform from WGS84 to raster CRS
        with rasterio.open(self.topo_diff.filename) as src:
            raster_crs = src.crs
        
        transformer = Transformer.from_crs('EPSG:4326', raster_crs, always_xy=True)
        poly_native = shapely_transform(transformer.transform, geom)
        
        # Store geometry
        if self.current_category == 'stable':
            self.stable_geoms.append(poly_native)
            self.stable_names.append(f"Stable_{len(self.stable_geoms)}")
            # Track that user has manually drawn stable areas
            self._user_drew_stable = True
        else:
            self.unstable_geoms.append(poly_native)
            self.unstable_names.append(f"FOI_{len(self.unstable_geoms)}")

            # Auto-derive stable areas only if:
            # 1. auto_stable_from_unstable is enabled, AND
            # 2. User has NOT manually drawn any stable areas
            if self.auto_stable_from_unstable and not getattr(self, '_user_drew_stable', False):
                self.derive_stable_from_unstable(replace=True)
        
        # Update GeoJSON layers
        self._update_geojson_layers()

    # -------------------- GeoJSON updates --------------------
    def _update_geojson_layers(self):
        """Refresh the GeoJSON layers on the map."""
        # Stable features
        stable_features = []
        for geom, name in zip(self.stable_geoms, self.stable_names):
            # Transform to WGS84 for display
            with rasterio.open(self.topo_diff.filename) as src:
                raster_crs = src.crs
            transformer = Transformer.from_crs(raster_crs, 'EPSG:4326', always_xy=True)
            geom_4326 = shapely_transform(transformer.transform, geom)
            
            stable_features.append({
                "type": "Feature",
                "geometry": mapping(geom_4326),
                "properties": {"name": name}
            })
        
        self.geojson_stable.data = {
            "type": "FeatureCollection",
            "features": stable_features
        }
        
        # Unstable features
        unstable_features = []
        for geom, name in zip(self.unstable_geoms, self.unstable_names):
            with rasterio.open(self.topo_diff.filename) as src:
                raster_crs = src.crs
            transformer = Transformer.from_crs(raster_crs, 'EPSG:4326', always_xy=True)
            geom_4326 = shapely_transform(transformer.transform, geom)
            
            unstable_features.append({
                "type": "Feature",
                "geometry": mapping(geom_4326),
                "properties": {"name": name}
            })
        
        self.geojson_unstable.data = {
            "type": "FeatureCollection",
            "features": unstable_features
        }

    # -------------------- File loading --------------------
    def load_stable_polygons(
        self,
        path: Path | str,
        name_field: Optional[str] = None,
        assume_crs: Optional[str] = 'EPSG:4326'
    ):
        """Load stable area polygons from a file."""
        gdf = gpd.read_file(path)
        
        # Handle missing CRS
        if gdf.crs is None and assume_crs is not None:
            gdf = gdf.set_crs(assume_crs)
        
        # Transform to raster CRS
        with rasterio.open(self.topo_diff.filename) as src:
            raster_crs = src.crs
        gdf = gdf.to_crs(raster_crs)
        
        # Extract geometries and names
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    self.stable_geoms.append(poly)
                    name = row[name_field] if name_field and name_field in row else f"Stable_{len(self.stable_geoms)}"
                    self.stable_names.append(str(name))
            elif isinstance(geom, Polygon):
                self.stable_geoms.append(geom)
                name = row[name_field] if name_field and name_field in row else f"Stable_{len(self.stable_geoms)}"
                self.stable_names.append(str(name))

        # Mark that user has provided stable areas (don't auto-derive)
        if self.stable_geoms:
            self._user_drew_stable = True

        self._update_geojson_layers()

    def load_unstable_polygons(
        self,
        path: Path | str,
        name_field: Optional[str] = None,
        assume_crs: Optional[str] = 'EPSG:4326'
    ):
        """Load unstable area (FOI) polygons from a file."""
        gdf = gpd.read_file(path)
        
        # Handle missing CRS
        if gdf.crs is None and assume_crs is not None:
            gdf = gdf.set_crs(assume_crs)
        
        # Transform to raster CRS
        with rasterio.open(self.topo_diff.filename) as src:
            raster_crs = src.crs
        gdf = gdf.to_crs(raster_crs)
        
        # Extract geometries and names
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    self.unstable_geoms.append(poly)
                    name = row[name_field] if name_field and name_field in row else f"FOI_{len(self.unstable_geoms)}"
                    self.unstable_names.append(str(name))
            elif isinstance(geom, Polygon):
                self.unstable_geoms.append(geom)
                name = row[name_field] if name_field and name_field in row else f"FOI_{len(self.unstable_geoms)}"
                self.unstable_names.append(str(name))
        
        self._update_geojson_layers()

    # -------------------- Auto-derive stable --------------------
    def derive_stable_from_unstable(self, replace: bool = False):
        """
        Derive stable areas as: valid raster extent minus union of unstable areas.
        
        Parameters
        ----------
        replace : bool
            If True, replace existing stable areas. If False, add to them.
        """
        if not self.unstable_geoms:
            return
        
        # Get valid data mask from raster
        with rasterio.open(self.topo_diff.filename) as src:
            data = src.read(1)
            nodata = src.nodata
            transform = src.transform
            
            # Create mask of valid pixels
            valid = np.ones(data.shape, dtype=np.uint8)
            if nodata is not None:
                valid[data == nodata] = 0
            valid[np.isnan(data)] = 0
            
            # Vectorize valid areas
            shapes_gen = rfeatures.shapes(valid, mask=valid.astype(bool), transform=transform)
            valid_polys = [shape(geom) for geom, val in shapes_gen if val == 1]
        
        if not valid_polys:
            return
        
        # Union all valid areas
        valid_union = unary_union(valid_polys)
        
        # Union all unstable areas
        unstable_union = unary_union(self.unstable_geoms)
        
        # Compute difference
        stable_derived = valid_union.difference(unstable_union)
        
        # Handle multi-part results
        if isinstance(stable_derived, MultiPolygon):
            derived_polys = list(stable_derived.geoms)
        elif isinstance(stable_derived, Polygon):
            derived_polys = [stable_derived]
        else:
            derived_polys = []
        
        # Filter by area if requested
        if self.derive_min_area is not None:
            derived_polys = [p for p in derived_polys if p.area >= self.derive_min_area]
        
        # Simplify if requested
        if self.simplify_tolerance is not None:
            derived_polys = [p.simplify(self.simplify_tolerance) for p in derived_polys]
        
        # Update stable geometries
        if replace:
            self.stable_geoms = derived_polys
            self.stable_names = [f"Stable_{i+1}" for i in range(len(derived_polys))]
        else:
            start_idx = len(self.stable_geoms)
            self.stable_geoms.extend(derived_polys)
            self.stable_names.extend([f"Stable_{start_idx + i + 1}" for i in range(len(derived_polys))])
        
        self._update_geojson_layers()

    # -------------------- Export --------------------
    def export_geodataframes(self) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Export stable and unstable geometries as GeoDataFrames in raster CRS.
        
        Returns
        -------
        gdf_stable, gdf_unstable : GeoDataFrame
            Stable and unstable area geodataframes
        """
        with rasterio.open(self.topo_diff.filename) as src:
            raster_crs = src.crs
        
        gdf_stable = gpd.GeoDataFrame(
            {'name': self.stable_names, 'geometry': self.stable_geoms},
            crs=raster_crs
        )
        
        gdf_unstable = gpd.GeoDataFrame(
            {'name': self.unstable_names, 'geometry': self.unstable_geoms},
            crs=raster_crs
        )
        
        return gdf_stable, gdf_unstable

    def export_geodataframes_latlon(self) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Export stable and unstable geometries as GeoDataFrames in WGS84.
        
        Returns
        -------
        gdf_stable, gdf_unstable : GeoDataFrame
            Stable and unstable area geodataframes in EPSG:4326
        """
        gdf_stable, gdf_unstable = self.export_geodataframes()
        return gdf_stable.to_crs('EPSG:4326'), gdf_unstable.to_crs('EPSG:4326')

    # -------------------- Overlay refresh & swap --------------------
    def refresh_overlay(
        self,
        cmap: Optional[str] = None,
        dpi: Optional[int] = None,
        *,
        recalc_bounds: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """Regenerate and swap the overlay image."""
        if cmap is not None:
            self.overlay_cmap = cmap
        if dpi is not None:
            self.overlay_dpi = dpi
        if vmin is not None or vmax is not None:
            if vmin is None and vmax is not None:
                self.overlay_vmax = float(vmax)
                self.overlay_vmin = -abs(self.overlay_vmax)
            elif vmin is not None and vmax is None:
                self.overlay_vmin = float(vmin)
                self.overlay_vmax = abs(self.overlay_vmin)
            else:
                self.overlay_vmin = vmin
                self.overlay_vmax = vmax

        if recalc_bounds:
            with rasterio.open(self.topo_diff.filename) as ds:
                bounds = ds.bounds
                crs = ds.crs
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            west, south = transformer.transform(bounds.left, bounds.bottom)
            east, north = transformer.transform(bounds.right, bounds.top)
            self.latlon_bounds = ((south, west), (north, east))

        new_image = self._new_overlay_image_path(format=self.overlay_format)
        self._generate_overlay_image(
            new_image, cmap=self.overlay_cmap, dpi=self.overlay_dpi,
            vmin=self.overlay_vmin, vmax=self.overlay_vmax,
            format=self.overlay_format
        )

        new_data_url = self._image_to_data_url(new_image)
        try:
            self.map.remove_layer(self.overlay_layer)
        except Exception:
            pass
        self.overlay_layer = ImageOverlay(url=new_data_url, bounds=self.latlon_bounds)
        self.map.add_layer(self.overlay_layer)
        self._overlay_image = new_image
        self._overlay_data_url = new_data_url

    def update_topo_diff(
        self,
        topo_diff_path: Path | str,
        *,
        cmap: Optional[str] = None,
        dpi: Optional[int] = None,
        recalc_bounds: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """Update the topographic difference raster and refresh the overlay."""
        try:
            from raster import Raster
        except ImportError:
            from .raster import Raster
        
        self.topo_diff = Raster.from_file(str(topo_diff_path))
        self.refresh_overlay(
            cmap=cmap,
            dpi=dpi,
            recalc_bounds=recalc_bounds,
            vmin=vmin,
            vmax=vmax,
        )


def descriptive_stats(values: np.ndarray) -> pd.DataFrame:
    """
    Compute descriptive statistics for a 1D array of values.
    Returns a one-row DataFrame.
    """
    data = values[~np.isnan(values)]
    if data.size == 0:
        cols = ['mean','median','mode','std','variance','min','max',
                'skewness','kurtosis','0.5_percentile','99.5_percentile']
        return pd.DataFrame([{c: np.nan for c in cols}])
    mean = float(np.mean(data))
    median = float(np.median(data))
    try:
        mode = statistics.mode(data)
    except statistics.StatisticsError:
        mode = np.nan
    std = float(np.std(data))
    var = float(np.var(data))
    minimum = float(np.min(data))
    maximum = float(np.max(data))
    skew = float(stats.skew(data))
    kurt = float(stats.kurtosis(data))
    p1, p99 = np.percentile(data, [0.5, 99.5])
    return pd.DataFrame([{
        'mean': mean,
        'median': median,
        'mode': mode,
        'std': std,
        'variance': var,
        'min': minimum,
        'max': maximum,
        'skewness': skew,
        'kurtosis': kurt,
        '0.5_percentile': float(p1),
        '99.5_percentile': float(p99)
    }])


class StableAreaRasterizer:
    """
    Rasterize stable-area polygons to mask a topographic-difference raster:
    - `rasterize_all`: one output where inside polygons = original values, outside = nodata.
    - `rasterize_each`: separate rasters per polygon.
    """
    def __init__(self, topo_diff_path: Path | str, stable_gdf, nodata: float = -9999):
        self.topo_path = Path(topo_diff_path)
        self.gdf = stable_gdf.copy()
        self.nodata = nodata

    def rasterize_all(self, output_path: Path | str) -> Path:
        """Rasterize all stable polygons into a single masked raster."""
        out_path = Path(output_path)
        with rasterio.open(self.topo_path) as src:
            profile = src.profile.copy()
            profile.update(nodata=self.nodata)
            data = src.read(1)
            mask = rasterize(
                [(geom, 1) for geom in self.gdf.geometry],
                out_shape=src.shape,
                transform=src.transform,
                fill=0,
                dtype='uint8'
            )
            out = np.where(mask == 1, data, self.nodata)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(out, 1)
        return out_path

    def rasterize_each(self, output_dir: Path | str) -> dict[int, Path]:
        """Rasterize each stable polygon into a separate raster file."""
        outdir = Path(output_dir)
        outdir.mkdir(exist_ok=True, parents=True)
        paths: dict[int, Path] = {}
        with rasterio.open(self.topo_path) as src:
            profile = src.profile.copy()
            profile.update(nodata=self.nodata)
            data = src.read(1)
            for idx, row in self.gdf.iterrows():
                geom = row.geometry
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=src.shape,
                    transform=src.transform,
                    fill=0,
                    dtype='uint8'
                )
                out = np.where(mask == 1, data, self.nodata)
                path = outdir / f"stable_area_{idx}.tif"
                with rasterio.open(path, 'w', **profile) as dst:
                    dst.write(out, 1)
                paths[idx] = path
        return paths


class StableAreaAnalyzer:
    """
    Use rasters produced by StableAreaRasterizer to compute descriptive stats:
    - `stats_all`: stats on combined-area raster.
    - `stats_each`: stats on each individual-area raster.
    """
    def __init__(self, rasterizer: StableAreaRasterizer):
        self.rasterizer = rasterizer

    def _stats_from_raster(self, path: Path | str) -> pd.DataFrame:
        """Compute descriptive statistics from a raster file."""
        with rasterio.open(path) as src:
            arr = src.read(1)
            arr = np.where(arr == src.nodata, np.nan, arr)
            flat = arr.ravel()
        return descriptive_stats(flat)

    def stats_all(self, output_path: Path | str) -> pd.DataFrame:
        """Compute statistics on the combined stable area raster."""
        out = self.rasterizer.rasterize_all(output_path)
        df = self._stats_from_raster(out)
        df.index = ['all_areas']
        return df

    def stats_each(self, output_dir: Path | str) -> pd.DataFrame:
        """Compute statistics on each individual stable area raster."""
        paths = self.rasterizer.rasterize_each(output_dir)
        records: List[pd.DataFrame] = []
        for area_id, path in paths.items():
            df = self._stats_from_raster(path)
            df['area_id'] = area_id
            records.append(df)
        result = pd.concat(records, ignore_index=True).set_index('area_id')
        return result
