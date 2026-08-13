"""
Central configuration for the Beetaloo UAV-LiDAR / SAR / Sentinel-2 reproduction pipeline.

Edit PROJECT_DIR and LAS_FILE to match your local Windows setup, e.g.:

    PROJECT_DIR = Path(r"C:/Users/you/beetaloo_project")
    LAS_FILE    = Path(r"C:/Users/you/data/beetaloo_sample.las")
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (edit these for your machine)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Path to your local test .las file
LAS_FILE = DATA_DIR / "TurnerRiver2024-C1-AHD_6597715_50.laz"

# Derived/intermediate outputs
TILE_DIR = OUTPUT_DIR / "lidar_tiles"          # tiled, normalised .laz
METRICS_RASTER = OUTPUT_DIR / "lidar_structural_metrics.tif"  # 20 m multiband raster
# SAR_ZARR = OUTPUT_DIR / "sar_composites.zarr"
# S2_ZARR = OUTPUT_DIR / "s2_composites.zarr"
SAR_NC = OUTPUT_DIR / 'sar_composites.nc'
S2_NC = OUTPUT_DIR / 's2_composites.nc'
SAMPLES_GPKG = OUTPUT_DIR / "sample_points.gpkg"
TRAINING_TABLE = OUTPUT_DIR / "training_table.parquet"

for d in (DATA_DIR, OUTPUT_DIR, TILE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Processing parameters (Section 3.2.3, 4.1, 4.2)
# ---------------------------------------------------------------------------
TILE_SIZE = 500       # m, lastile tile size
TILE_BUFFER = 10      # m, buffer to avoid edge artifacts
HEIGHT_MIN = 0        # m, lasheight lower cutoff
HEIGHT_MAX = 40       # m, lasheight upper cutoff (erroneous returns removed)

PIXEL_SIZE = 20       # m, matches SAR/Sentinel-2 20 m grid (Section 4.1)
SAMPLE_STRIDE = 3     # every 3rd pixel (Section 4.2 "regular spatial sample")
SAMPLE_JITTER_MAX = 20  # m, random jitter applied to sample locations

# Height bands for density metrics (Table 2)
DENSITY_BANDS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 10), (10, 15), (15, 20), (20, 25),
]
HEIGHT_PERCENTILES = [5, 10, 25, 50, 75, 90, 95, 98]

# ---------------------------------------------------------------------------
# Date range and seasons (Section 4.2)
# ---------------------------------------------------------------------------
DATE_START = "2024-06-01"
DATE_END = "2025-05-31"

# (start_month, start_day, end_month, end_day) — inclusive, Austral seasons
SEASONS = {
    "winter": ((6, 1), (8, 31)),    # dry season
    "spring": ((9, 1), (11, 30)),   # dry season
    "summer": ((12, 1), (2, 29)),   # wet season (wraps year)
    "autumn": ((3, 1), (5, 31)),    # wet season
}

# Late dry season window used for "DrySeas" composites
DRY_SEASON = ((9, 1), (11, 30))

# ---------------------------------------------------------------------------
# Sentinel-2 bands (Table 1) — DEA Collection 3 (ga_s2am/bm_ard_3 / ga_s2_ard_3)
# ---------------------------------------------------------------------------
S2_BANDS = [
    "nbart_blue",
    "nbart_green",
    "nbart_red",
    "nbart_red_edge_1",
    "nbart_red_edge_2",
    "nbart_red_edge_3",
    "nbart_nir_1",
    "nbart_nir_2",
    "nbart_swir_2",
    "nbart_swir_3",
]

S2_CLOUD_PROB_THRESHOLD = 0.5  # mask pixels where max cloud prob within 100 m > 0.5
S2_MAX_CLOUD_COVER = 80        # % scene-level cloud cover cutoff

# DEA STAC / ODC settings
DEA_STAC_URL = "https://explorer.dea.ga.gov.au/stac"
DEA_S2_COLLECTIONS = ["ga_s2am_ard_3", "ga_s2bm_ard_3"]

# ---------------------------------------------------------------------------
# Sentinel-1 SAR via openEO (Section 3.3.3)
# ---------------------------------------------------------------------------
OPENEO_BACKEND_URL = "https://openeo.dataspace.copernicus.eu"
S1_COLLECTION = "SENTINEL1_GRD"
S1_BANDS = ["VV", "VH"]

# Microsoft Planetary Computer
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1_COLLECTION_PC = "sentinel-1-rtc"   # radiometric terrain corrected = gamma0

# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------
TARGET_CRS = "EPSG:3577"  # GDA94 / Australian Albers — matches DEA/GA products
