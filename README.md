# topochange

Topographic change detection and uncertainty quantification using variogram-based methods.

## Installation

### From source (development)

```bash
git clone https://github.com/yourusername/topochange.git
cd topochange
pip install -e .
```

### With optional dependencies

```bash
# Interactive Jupyter widgets (ipyleaflet, ipywidgets)
pip install -e ".[interactive]"

# Point cloud support (PDAL)
pip install -e ".[pointcloud]"

# Point cloud alignment (small_gicp)
pip install -e ".[alignment]"

# All optional dependencies
pip install -e ".[all]"

# Development dependencies (pytest, black, ruff)
pip install -e ".[dev]"
```

## Quick Start

```python
from topochange import Raster, RasterPair, VariogramAnalysis

# Load two DEMs
dem1 = Raster.from_file("dem_2019.tif")
dem2 = Raster.from_file("dem_2023.tif")

# Create pair and compute difference
pair = RasterPair(dem1, dem2)
result = pair.compute_difference()
```

## Features

- Load and compare raster DEMs and point clouds
- CRS transformations including vertical datum conversions
- Variogram-based uncertainty analysis
- Regional uncertainty propagation
- Interactive stable area identification

## Requirements

- Python >= 3.9
- numpy, pandas, scipy, matplotlib
- rasterio, pyproj, geopandas, shapely, rioxarray
- numba

## License

MIT
