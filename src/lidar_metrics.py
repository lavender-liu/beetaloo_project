"""
UAV-LiDAR processing pipeline (reproduces Section 3.2.3 and 4.1 of the report).

Original workflow used LAStools (lastile, lasground, lasheight) + R/lidR for
metric computation. This module reproduces the equivalent steps in Python
using laspy + numpy + rasterio, gridding directly onto a 20 m raster that
matches the SAR/Sentinel-2 pixel grid (Figure 6, Section 4.1).

Notes on fidelity:
- Ground classification: if the input .las/.laz is NOT already ground-classified
  (RIEGL/RiPROCESS output normally is), a simple CSF (cloth simulation filter)
  ground classifier from `csf`/`pdal` is recommended. This module assumes the
  point cloud already carries valid ground classification (classification
  code 2) as produced by lasground, OR that a 'HeightAboveGround' field is
  present. If neither is available, points are treated as already normalised.
- Height normalisation and the 0-40 m filtering reproduces `lasheight`
  (Section 3.2.3).
- The 23 metrics in Table 2 are computed per 20 m cell directly from the
  normalised point cloud, matching the lidR-based approach in Section 4.1.
"""

from __future__ import annotations

import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin

from . import config


# ---------------------------------------------------------------------------
# 1. Load + height-normalise
# ---------------------------------------------------------------------------
def load_and_normalise(las_path, height_min=config.HEIGHT_MIN, height_max=config.HEIGHT_MAX):
    """
    Load a .las/.laz file and return arrays of x, y, height-above-ground (z_norm),
    filtered to height_min <= z_norm <= height_max (reproduces lasheight cleanup,
    Section 3.2.3).

    Returns
    -------
    x, y, z_norm, classification : np.ndarray
    crs : pyproj CRS or None
    """
    las = laspy.read(las_path)

    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)
    classification = np.asarray(las.classification)

    # Height above ground: prefer an existing extra dimension if present
    extra_dims = set(las.point_format.extra_dimension_names)
    if "HeightAboveGround" in extra_dims:
        z_norm = np.asarray(las["HeightAboveGround"])
    else:
        # Fall back: normalise against the lowest ground-classified point
        # in each (approximate) local neighbourhood is expensive; for a
        # quick local test file we normalise against the per-tile minimum
        # ground elevation as a coarse approximation. For production data
        # already processed through RiPROCESS/lasground/lasheight this
        # branch is not needed (HeightAboveGround should already exist).
        ground_mask = classification == 2
        if ground_mask.sum() > 0:
            # Simple raster-based ground surface at coarse resolution,
            # then bilinear-ish nearest lookup per point.
            z_norm = _normalise_against_ground(x, y, z, ground_mask)
        else:
            print(
                "WARNING: no 'HeightAboveGround' field and no ground-classified "
                "points (class 2) found. Treating Z as already normalised."
            )
            z_norm = z.copy()

    # Reproduce lasheight cleanup: drop points below ground or > 40 m
    keep = (z_norm >= height_min) & (z_norm <= height_max)

    crs = las.header.parse_crs() if hasattr(las.header, "parse_crs") else None

    return x[keep], y[keep], z_norm[keep], classification[keep], crs


def _normalise_against_ground(x, y, z, ground_mask, cell=5.0):
    """
    Coarse fallback height normalisation: builds a low-resolution ground
    surface from classified ground points (median z per `cell` m grid cell)
    and subtracts the nearest ground cell value from every point's z.

    Only used if the input file lacks a precomputed HeightAboveGround field.
    For a properly processed UAV-LiDAR dataset (as in the report pipeline,
    Section 3.2.2-3.2.3) this branch should not be triggered.
    """
    gx, gy, gz = x[ground_mask], y[ground_mask], z[ground_mask]

    xmin, ymin = gx.min(), gy.min()
    col = ((gx - xmin) // cell).astype(int)
    row = ((gy - ymin) // cell).astype(int)
    ncols, nrows = col.max() + 1, row.max() + 1

    ground_grid = np.full((nrows, ncols), np.nan)
    for r, c, v in zip(row, col, gz):
        if np.isnan(ground_grid[r, c]) or v < ground_grid[r, c]:
            ground_grid[r, c] = v

    # Fill gaps with overall median ground elevation
    ground_grid = np.where(np.isnan(ground_grid), np.nanmedian(gz), ground_grid)

    col_all = np.clip(((x - xmin) // cell).astype(int), 0, ncols - 1)
    row_all = np.clip(((y - ymin) // cell).astype(int), 0, nrows - 1)
    ground_z = ground_grid[row_all, col_all]

    return z - ground_z


# ---------------------------------------------------------------------------
# 2. Per-cell structural metrics (Table 2, 23 metrics)
# ---------------------------------------------------------------------------
def _cell_metrics(z_norm: np.ndarray) -> dict:
    """Compute the 23 structural metrics (Table 2) for one 20 m cell."""

    n_total = z_norm.size
    if n_total == 0:
        return {k: np.nan for k in METRIC_NAMES}

    n_ground = np.sum(z_norm <= 0)
    n_nonground = n_total - n_ground

    out = {}

    # Gap Fraction Probability: proportion of returns from/near the ground
    # (i.e. canopy "gaps" letting laser reach the ground), Table 2.
    out["GFP"] = n_ground / n_total if n_total > 0 else np.nan

    # Plant Area Index: derived from the canopy gap fraction following the
    # Beer-Lambert relation commonly used with lidR (e.g. -ln(GFP)/k, k~0.5).
    # If GFP == 0, PAI is undefined; cap to a large but finite value.
    gfp = out["GFP"]
    if gfp > 0:
        out["PAI"] = -np.log(gfp) / 0.5
    else:
        out["PAI"] = -np.log(1.0 / n_total) / 0.5  # finite-sample approx

    nonground_z = z_norm[z_norm > 0]

    # Standard deviation of height of all non-ground returns
    out["stdev"] = np.std(nonground_z) if nonground_z.size > 0 else np.nan

    # Height percentiles (Table 2)
    for p in config.HEIGHT_PERCENTILES:
        out[f"p{p}"] = (
            np.percentile(nonground_z, p) if nonground_z.size > 0 else np.nan
        )

    # Density by height band: proportion of non-ground returns in each band
    # relative to TOTAL returns (per Table 2 wording "/ total number of returns")
    for lo, hi in config.DENSITY_BANDS:
        key = f"dens_{lo}_{hi}m"
        n_band = np.sum((z_norm > lo) & (z_norm <= hi))
        out[key] = n_band / n_total if n_total > 0 else np.nan

    out["dens_>1m"] = np.sum(z_norm > 1) / n_total if n_total > 0 else np.nan
    out["dens_>2m"] = np.sum(z_norm > 2) / n_total if n_total > 0 else np.nan
    out["dens_total"] = n_nonground / n_total if n_total > 0 else np.nan

    return out


METRIC_NAMES = (
    ["GFP", "PAI", "stdev"]
    + [f"p{p}" for p in config.HEIGHT_PERCENTILES]
    + [f"dens_{lo}_{hi}m" for lo, hi in config.DENSITY_BANDS]
    + ["dens_>1m", "dens_>2m", "dens_total"]
)


# ---------------------------------------------------------------------------
# 3. Grid the point cloud onto the 20 m raster and write GeoTIFF
# ---------------------------------------------------------------------------
def compute_structural_metrics_raster(
    x, y, z_norm, crs, out_path=config.METRICS_RASTER, pixel_size=config.PIXEL_SIZE
):
    """
    Bin points into `pixel_size` m cells and compute the 23 structural metrics
    (Table 2) per cell. Writes a multiband GeoTIFF aligned to a regular grid
    (matching the SAR/Sentinel-2 20 m grid, Section 4.1, Figure 6).

    Returns the output path and the affine transform used.
    """
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # Snap extent to the pixel grid
    xmin = np.floor(xmin / pixel_size) * pixel_size
    ymax = np.ceil(ymax / pixel_size) * pixel_size

    ncols = int(np.ceil((xmax - xmin) / pixel_size))
    nrows = int(np.ceil((ymax - ymin) / pixel_size))

    col = ((x - xmin) // pixel_size).astype(int)
    row = ((ymax - y) // pixel_size).astype(int)

    col = np.clip(col, 0, ncols - 1)
    row = np.clip(row, 0, nrows - 1)

    cell_id = row * ncols + col

    n_metrics = len(METRIC_NAMES)
    data = np.full((n_metrics, nrows, ncols), np.nan, dtype=np.float32)

    order = np.argsort(cell_id)
    cell_id_sorted = cell_id[order]
    z_sorted = z_norm[order]

    # Iterate over unique cells using boundaries in the sorted array
    unique_cells, start_idx = np.unique(cell_id_sorted, return_index=True)
    end_idx = np.append(start_idx[1:], len(cell_id_sorted))

    for cid, s, e in zip(unique_cells, start_idx, end_idx):
        r, c = divmod(int(cid), ncols)
        metrics = _cell_metrics(z_sorted[s:e])
        for i, name in enumerate(METRIC_NAMES):
            data[i, r, c] = metrics[name]

    transform = from_origin(xmin, ymax, pixel_size, pixel_size)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=n_metrics,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        for i, name in enumerate(METRIC_NAMES):
            dst.write(data[i], i + 1)
            dst.set_band_description(i + 1, name)

    return out_path, transform


# ---------------------------------------------------------------------------
# 4. Convenience: bounding box in lon/lat for SAR/Sentinel-2 queries
# ---------------------------------------------------------------------------
def get_bbox_lonlat(x, y, crs, buffer_m=0):
    """
    Return (minlon, minlat, maxlon, maxlat) for the point cloud extent,
    reprojected to EPSG:4326, with optional buffer in metres (applied in
    the source CRS before reprojection).
    """
    from pyproj import Transformer

    xmin, xmax = x.min() - buffer_m, x.max() + buffer_m
    ymin, ymax = y.min() - buffer_m, y.max() + buffer_m

    if crs is None:
        raise ValueError(
            "LAS file has no CRS defined. Set the CRS manually, e.g. "
            "crs='EPSG:32753' (UTM zone 53S, typical for Beetaloo NT)."
        )

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)

    return (
        min(lon_min, lon_max),
        min(lat_min, lat_max),
        max(lon_min, lon_max),
        max(lat_min, lat_max),
    )


def get_bbox_target_crs(x, y, target_crs=config.TARGET_CRS, source_crs=None, buffer_m=0):
    """
    Return (minx, miny, maxx, maxy) of the point cloud extent reprojected to
    `target_crs` (default EPSG:3577, matching DEA products), with optional
    buffer in metres (applied in the source CRS before reprojection).
    """
    from pyproj import Transformer

    xmin, xmax = x.min() - buffer_m, x.max() + buffer_m
    ymin, ymax = y.min() - buffer_m, y.max() + buffer_m

    if source_crs is None:
        raise ValueError("source_crs must be provided (LAS file CRS).")

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    corners_x, corners_y = transformer.transform(
        [xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax]
    )
    return (min(corners_x), min(corners_y), max(corners_x), max(corners_y))
