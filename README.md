# Beetaloo UAV-LiDAR Biodiversity Pipeline

Python reproduction of the CSIRO/GISERA report:  
*"NT Beetaloo Biodiversity LiDAR – UAV-LiDAR structural metrics, SAR and Sentinel-2 modelling"* (June 2025).

---

## Project structure

```
beetaloo_project/
├── src/
│   ├── config.py              # All paths and parameters — edit this first
│   ├── lidar_metrics.py       # LiDAR load, normalise, 23 structural metrics
│   └── remote_sensing.py      # SAR (openEO) + S2 (DEA odc-stac) access + compositing
├── notebooks/
│   ├── 01_lidar_processing.ipynb          # Point cloud → 20 m structural metric raster
│   ├── 02_sar_sentinel2_access.ipynb      # Fetch + composite SAR/S2 data
│   ├── 03_correlation_analysis.ipynb      # Spearman correlations (Figs 8–10)
│   ├── 04_qrf_modelling.ipynb             # QRF training + 10-fold CV (Table 3, Fig 12)
│   └── 05_extrapolation_timeseries.ipynb  # Wall-to-wall maps + annual time series
├── data/                      # Place your .las file here
├── outputs/                   # All generated rasters, Zarr stores, figures
└── requirements.txt
```

---

## Setup

```bash
# 1. Create and activate a conda/venv environment (Python 3.11 recommended)
conda create -n beetaloo python=3.11
conda activate beetaloo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Register your Copernicus Data Space account (free) for SAR access
#    https://dataspace.copernicus.eu — needed for openEO authentication in NB02

# 4. Edit src/config.py:
#    - Set LAS_FILE to your local .las path
#    - Adjust DATE_START / DATE_END if needed
```

---

## Running the pipeline

Run notebooks **in order** from the `notebooks/` directory:

| Notebook | What it does | Key output |
|---|---|---|
| `01_lidar_processing` | Load .las, height-normalise, compute 23 structural metrics | `outputs/lidar_structural_metrics.tif` |
| `02_sar_sentinel2_access` | Fetch SAR (openEO) + S2 (DEA), composite | `outputs/sar_composites.zarr`, `outputs/s2_composites.zarr` |
| `03_correlation_analysis` | Spearman ρ, scatter plots | `outputs/table_correlation_*.csv`, figures |
| `04_qrf_modelling` | Train QRF, 10-fold CV, accuracy & importance | `outputs/models/`, `outputs/table_accuracy_summary.csv` |
| `05_extrapolation_timeseries` | Predict full extent + annual maps | `outputs/predict_*.tif`, time-series figures |

---

## Key design choices

| Report step | Python implementation |
|---|---|
| `lasground` / height normalisation | `laspy` + coarse raster normalisation (or HeightAboveGround field if present) |
| 23 structural metrics (Table 2) | Custom cell-wise computation in `lidar_metrics.py` |
| SAR gamma0 + Lee filter (Section 3.3.3) | `openeo` `sar_backscatter(coefficient='gamma0-terrain', noise_removal=True)` |
| Sentinel-2 ARD (Section 3.4.2) | DEA Collection 3 via `odc-stac` + `pystac-client` |
| s2cloudless cloud masking | `mask_clouds_s2()` using `oa_s2cloudless` band from DEA |
| Geometric median composite | `odc.algo.xr_geomedian` (Weiszfeld fallback if not installed) |
| Quantile regression forest (R `ranger`) | `quantile-forest.RandomForestQuantileRegressor` |
| 10-fold cross-validation | `sklearn.model_selection.KFold` |

---

## Notes

- **Large extents**: enable Dask chunking in NB02 (`chunks={"x": 2048, "y": 2048, "time": 5}`) and request the openEO job as a batch job (`out_path=...`) rather than a synchronous download.
- **CRS**: if your .las file has no embedded CRS, set it manually in `config.py` (e.g. `EPSG:32753` for UTM zone 53S, which covers the Beetaloo Basin).
- **Ground classification**: if your .las does not carry `HeightAboveGround` or ground-classified returns (class 2), pre-process with `pdal` + CSF filter before running NB01.
