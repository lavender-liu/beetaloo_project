"""
Spaceborne SAR (Sentinel-1, via openEO) and multi-spectral (Sentinel-2, via
Digital Earth Australia / odc-stac) data access, reproducing Sections 3.3
and 3.4 of the report.

SAR (Section 3.3.3):
    Reproduces the GRD -> Gamma0 -> Lee filter -> multilook -> 20 m grid
    processing chain using openEO's pre-built Sentinel-1 GRD processes
    (Copernicus Data Space Ecosystem backend), which exposes
    `sar_backscatter` (gamma0-terrain) with built-in speckle filtering.

Sentinel-2 (Section 3.4.2):
    Reproduces DEA Collection 3 analysis-ready surface reflectance access
    via odc-stac against the DEA STAC catalogue, with s2cloudless-based
    cloud masking and geometric median compositing.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import pandas as pd
import os
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["AWS_DEFAULT_REGION"] = "ap-southeast-2"

from . import config


# ===========================================================================
# SAR (Sentinel-1) via openEO
# ===========================================================================
def connect_openeo(backend_url=config.OPENEO_BACKEND_URL):
    """
    Authenticate with an openEO backend (default: Copernicus Data Space
    Ecosystem). Opens an interactive OIDC login on first use.
    """
    import openeo

    con = openeo.connect(backend_url)
    con.authenticate_oidc()
    # con.authenticate_oidc_device()
    return con

def fetch_sar_backscatter(
    bbox_lonlat,
    date_start=config.DATE_START,
    date_end=config.DATE_END,
    bands=config.S1_BANDS,
    resolution=config.PIXEL_SIZE,
    target_crs=config.TARGET_CRS,
    chunks=None,
):
    """
    Load Sentinel-1 RTC (radiometric terrain corrected, equivalent to gamma0)
    from Microsoft Planetary Computer STAC — free, no credentials required.
    Uses odc-stac, consistent with the Sentinel-2 loading approach.
    """
    import pystac_client
    import planetary_computer
    import odc.stac

    catalog = pystac_client.Client.open(
        config.PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    minlon, minlat, maxlon, maxlat = bbox_lonlat
    search = catalog.search(
        collections=[config.S1_COLLECTION_PC],
        bbox=[minlon, minlat, maxlon, maxlat],
        datetime=f"{date_start}/{date_end}",
    )
    items = list(search.items())
    if len(items) == 0:
        raise ValueError("No Sentinel-1 RTC items found for the given bbox/date range.")
    print(f"Found {len(items)} Sentinel-1 RTC scenes")

    ds = odc.stac.load(
        items,
        bands=["vv", "vh"],
        crs=target_crs,
        resolution=resolution,
        bbox=[minlon, minlat, maxlon, maxlat],
        chunks=chunks or {},
        groupby="solar_day",
    )
    # Rename to uppercase to match rest of pipeline
    ds = ds.rename({"vv": "VV", "vh": "VH"})
    return ds

# def fetch_sar_backscatter(
#     connection,
#     bbox_lonlat,
#     date_start=config.DATE_START,
#     date_end=config.DATE_END,
#     bands=config.S1_BANDS,
#     resolution=config.PIXEL_SIZE,
#     target_crs=config.TARGET_CRS,
#     out_path=None,
# ):
#     """
#     Request Sentinel-1 GRD gamma0 (terrain-corrected) backscatter via openEO,
#     matching the report's processing chain: Calibration to Gamma Nought ->
#     speckle filter -> resample to 20 m grid (Section 3.3.3).

#     Parameters
#     ----------
#     connection : openeo.Connection
#         From `connect_openeo()`.
#     bbox_lonlat : tuple
#         (minlon, minlat, maxlon, maxlat), e.g. from
#         `lidar_metrics.get_bbox_lonlat`.
#     out_path : str or Path, optional
#         If given, downloads the result as a NetCDF to this path.

#     Returns
#     -------
#     xarray.Dataset with VV, VH bands (linear power; convert to dB with
#     10*log10 if needed) on the requested CRS/resolution grid, daily time
#     steps over the requested date range.
#     """
#     minlon, minlat, maxlon, maxlat = bbox_lonlat
#     spatial_extent = {
#         "west": minlon, "south": minlat, "east": maxlon, "north": maxlat,
#     }
#     temporal_extent = [date_start, date_end]

#     s1 = connection.load_collection(
#         config.S1_COLLECTION,
#         spatial_extent=spatial_extent,
#         temporal_extent=temporal_extent,
#         bands=bands,
#     )

#     # sar_backscatter: gamma0-terrain with built-in multitemporal speckle
#     # filtering, equivalent to the SNAP Calibration + Lee filter steps
#     # described in Section 3.3.3.
#     s1_gamma0 = s1.sar_backscatter(
#         # coefficient="gamma0-terrain",
#         coefficient="sigma0-ellipsoid",
#         elevation_model=None,
#         mask=False,
#         contributing_area=False,
#         local_incidence_angle=False,
#         ellipsoid_incidence_angle=False,
#         noise_removal=True,
#     )

#     # Resample to the 20 m target grid / CRS used throughout the report
#     s1_resampled = s1_gamma0.resample_spatial(
#         resolution=resolution, projection=target_crs, method="bilinear"
#     )

#     if out_path is not None:
#         job = s1_resampled.execute_batch(
#             outputfile=str(out_path), out_format="NetCDF"
#         )
#         return xr.open_dataset(out_path)

#     # Synchronous download for small areas/test files
#     result = s1_resampled.download(format="NetCDF")
#     return xr.open_dataset(result)


def to_db(da: xr.DataArray) -> xr.DataArray:
    """Convert linear backscatter power to decibels (10*log10)."""
    return 10 * np.log10(da.where(da > 0))


# ===========================================================================
# Sentinel-2 surface reflectance via Digital Earth Australia (odc-stac)
# ===========================================================================
def load_sentinel2_dea(
    bbox_lonlat,
    date_start=config.DATE_START,
    date_end=config.DATE_END,
    bands=config.S2_BANDS,
    resolution=config.PIXEL_SIZE,
    target_crs=config.TARGET_CRS,
    max_cloud_cover=config.S2_MAX_CLOUD_COVER,
    collections=config.DEA_S2_COLLECTIONS,
    stac_url=config.DEA_STAC_URL,
    chunks=None,
):
    """
    Load Sentinel-2A/2B analysis-ready surface reflectance (DEA Collection 3,
    `ga_s2am_ard_3` / `ga_s2bm_ard_3`) over `bbox_lonlat` and `[date_start,
    date_end]`, reproducing the data source described in Section 3.4.2.

    Bands are resampled to a common `resolution` (default 20 m) on
    `target_crs` (default EPSG:3577), matching step (iii) of the
    compositing procedure.

    Returns an xarray.Dataset with the requested bands plus `oa_s2cloudless`
    (cloud probability) and `oa_fmask` for masking.

    Parameters
    ----------
    chunks : dict, optional
        Dask chunking, e.g. {"x": 1024, "y": 1024, "time": 1}. If None,
        loads eagerly (fine for small test areas).
    """
    import pystac_client
    import odc.stac

    catalog = pystac_client.Client.open(stac_url)

    minlon, minlat, maxlon, maxlat = bbox_lonlat

    search = catalog.search(
        collections=collections,
        bbox=[minlon, minlat, maxlon, maxlat],
        datetime=f"{date_start}/{date_end}",
        # query={"eo:cloud_cover": {"lt": max_cloud_cover}}, # not supported anymore
        filter={"op": "lt", "args": [{"property": "eo:cloud_cover"}, max_cloud_cover]},
        filter_lang="cql2-json",
    )
    items = list(search.items())
    # Run this in a new cell to inspect available bands
    print(items[0].assets.keys())

    if len(items) == 0:
        raise ValueError(
            "No Sentinel-2 items found for the given bbox/date range. "
            "Check the bounding box and date range."
        )

    # load_bands = list(bands) + ["oa_s2cloudless", "oa_fmask"]
    load_bands = list(bands) + ["oa_s2cloudless_prob", "oa_fmask"]

    ds = odc.stac.load(
        items,
        bands=load_bands,
        crs=target_crs,
        resolution=resolution,
        bbox=[minlon, minlat, maxlon, maxlat],
        chunks=chunks or {},
        groupby="solar_day",
    )

    return ds


def mask_clouds_s2(ds: xr.Dataset, prob_threshold=config.S2_CLOUD_PROB_THRESHOLD,
                    dilation_m=100, resolution=config.PIXEL_SIZE):
    """
    Mask clouds using oa_s2cloudless_prob (0–100) and oa_fmask.
    Dilates the cloud mask by dilation_m metres to catch cloud edges.
    """
    cloud_prob = ds["oa_s2cloudless_prob"]  # 0–100 integer

    window_px = max(1, int(round(dilation_m / resolution)))
    dilated = cloud_prob.rolling(
        x=window_px * 2 + 1, y=window_px * 2 + 1, center=True, min_periods=1
    ).max()

    # Also include fmask cloud/shadow pixels
    fmask = ds["oa_fmask"]
    cloud_mask = (dilated > prob_threshold * 100) | (fmask == 2) | (fmask == 3)

    sr_bands = [b for b in ds.data_vars if b.startswith("nbart_")]
    masked = ds.copy()
    for b in sr_bands:
        masked[b] = ds[b].where(~cloud_mask)
    return masked


# ===========================================================================
# Temporal compositing (Section 4.2): annual / seasonal / monthly composites
# ===========================================================================
def _season_mask(time_index: pd.DatetimeIndex, season: str) -> np.ndarray:
    """Boolean mask selecting timestamps falling in the given Austral season."""
    (sm, sd), (em, ed) = config.SEASONS[season]
    months = time_index.month
    days = time_index.day

    if sm <= em:  # season does not wrap the calendar year
        in_season = (
            ((months > sm) | ((months == sm) & (days >= sd)))
            & ((months < em) | ((months == em) & (days <= ed)))
        )
    else:  # wraps year-end (e.g. summer: Dec-Feb)
        in_season = (
            ((months > sm) | ((months == sm) & (days >= sd)))
            | ((months < em) | ((months == em) & (days <= ed)))
        )
    return in_season


def compute_sar_composite(ds: xr.Dataset, period="annual", reducer="median"):
    """
    Compute a temporal composite of SAR VV/VH (and VV:VH ratio in dB) for one
    aggregation period, reproducing Section 4.2 / 3.3.3:

    - 'annual': median over the full time range
    - 'winter' / 'spring' / 'summer' / 'autumn': median over that Austral
      season across all years
    - 'dryseason': late dry season (Sep-Nov), see config.DRY_SEASON
    - 'month_acq': supply a pre-filtered ds containing only the acquisition
      month's timestamps and call with period='annual'

    Returns a Dataset with VV_db, VH_db (median backscatter in dB) and
    VVVH_ratio_db (VV - VH in dB).
    """
    if period == "annual":
        sub = ds
    elif period == "dryseason":
        time_index = pd.to_datetime(ds.time.values)
        (sm, sd), (em, ed) = config.DRY_SEASON
        months, days = time_index.month, time_index.day
        in_season = (
            ((months > sm) | ((months == sm) & (days >= sd)))
            & ((months < em) | ((months == em) & (days <= ed)))
        )
        sub = ds.isel(time=in_season)
    elif period in config.SEASONS:
        time_index = pd.to_datetime(ds.time.values)
        sub = ds.isel(time=_season_mask(time_index, period))
    else:
        raise ValueError(f"Unknown period: {period}")

    reduce_fn = getattr(sub, reducer)

    vv_db = to_db(sub["VV"]).pipe(lambda da: getattr(da, reducer)(dim="time"))
    vh_db = to_db(sub["VH"]).pipe(lambda da: getattr(da, reducer)(dim="time"))
    ratio_db = vv_db - vh_db

    out = xr.Dataset({"VV_db": vv_db, "VH_db": vh_db, "VVVH_ratio_db": ratio_db})
    out.attrs["period"] = period
    out.attrs["reducer"] = reducer
    return out


def compute_sar_variability(ds: xr.Dataset):
    """
    Inter-seasonal variability (Section 4.2, 'Var'): standard deviation
    across the four seasonal composite values, computed per variable
    (VV_db, VH_db, VVVH_ratio_db).
    """
    seasonal = {s: compute_sar_composite(ds, period=s) for s in config.SEASONS}
    stacked = xr.concat(seasonal.values(), dim="season")
    return stacked.std(dim="season")


def geometric_median(ds: xr.Dataset, bands, max_iter=50, tol=1e-4):
    """
    Compute the geometric median composite across `time` for the given
    `bands`, reproducing the Sentinel-2 compositing approach in Section
    3.4.2 (Roberts et al. 2017/2018 Weiszfeld-algorithm geomedian).

    Note: this is computationally expensive (the report ran it on a 64-core
    / 512 GB HPC node). For a small local test extent it is feasible but
    will be slow for large stacks — consider using
    `odc.algo.geomedian.xr_geomedian` (from `odc-algo`) if installed, which
    is a faster compiled implementation.
    """
    try:
        from odc.algo import xr_geomedian
        stacked = ds[bands].to_array(dim="band")
        gm = xr_geomedian(ds[bands], num_threads=4, eps=tol, maxiters=max_iter)
        return gm
    except ImportError:
        pass

    # Pure-numpy Weiszfeld fallback (slow; intended for small test extents)
    arr = np.stack([ds[b].values for b in bands], axis=0)  # (band, time, y, x)
    n_band, n_time, ny, nx = arr.shape

    flat = arr.reshape(n_band, n_time, ny * nx)
    out = np.full((n_band, ny * nx), np.nan, dtype=np.float32)

    for px in range(ny * nx):
        pts = flat[:, :, px]  # (band, time)
        valid = ~np.isnan(pts).any(axis=0)
        if valid.sum() == 0:
            continue
        pts_valid = pts[:, valid].T  # (n_valid, band)
        y = np.nanmedian(pts_valid, axis=0)
        for _ in range(max_iter):
            d = np.linalg.norm(pts_valid - y, axis=1)
            d[d < 1e-12] = 1e-12
            w = 1.0 / d
            y_new = (pts_valid * w[:, None]).sum(axis=0) / w.sum()
            if np.linalg.norm(y_new - y) < tol:
                y = y_new
                break
            y = y_new
        out[:, px] = y

    out = out.reshape(n_band, ny, nx)
    coords = {"y": ds.y, "x": ds.x}
    return xr.Dataset(
        {b: (("y", "x"), out[i]) for i, b in enumerate(bands)}, coords=coords
    )


def compute_s2_composite(ds: xr.Dataset, period="annual", bands=config.S2_BANDS):
    """
    Sentinel-2 geometric median composite for one aggregation period
    (Section 4.2): 'annual', 'winter'/'spring'/'summer'/'autumn', or
    'dryseason'.
    """
    if period == "annual":
        sub = ds
    elif period == "dryseason":
        time_index = pd.to_datetime(ds.time.values)
        (sm, sd), (em, ed) = config.DRY_SEASON
        months, days = time_index.month, time_index.day
        in_season = (
            ((months > sm) | ((months == sm) & (days >= sd)))
            & ((months < em) | ((months == em) & (days <= ed)))
        )
        sub = ds.isel(time=in_season)
    elif period in config.SEASONS:
        time_index = pd.to_datetime(ds.time.values)
        sub = ds.isel(time=_season_mask(time_index, period))
    else:
        raise ValueError(f"Unknown period: {period}")

    return geometric_median(sub, bands)
