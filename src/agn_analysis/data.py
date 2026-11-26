
import numpy as np
import pandas as pd
from pathlib import Path
from .model import AGNLC


def outlier_mask(t_days, y, dy, window=30.0, threshold=5.0, min_neighbours=5):
    """
    Flag outliers relative to running median.
    """
    n = len(t_days)
    local_median = np.full(n, np.nan)
    for i in range(n):
        in_window = np.abs(t_days - t_days[i]) <= window
        in_window[i] = False                # leave point itself out
        if in_window.sum() >= min_neighbours:
            local_median[i] = np.median(y[in_window])
        
    has_ref = np.isfinite(local_median)
    z = (y - local_median)/dy

    z_ref = z[has_ref]
    centre = np.median(z_ref)
    mad = np.median(np.abs(z_ref - centre))
    z = (z - centre) / (1.4826 * mad)

    keep = np.ones(n, dtype=bool)
    keep[has_ref] = np.abs(z[has_ref]) <= threshold
    return keep

def load_lc(path: Path, time_col: str, mag_col: str, err_col: str, filtercode: str, mask_outliers: bool = True, window: float = 30.0, threshold: float = 5.0, min_neighbours: int = 5):

    if filtercode not in ["zg", "zr", "zi"]:
        raise ValueError("Choose a filtercode between zg, zr or zi.")

    df = pd.read_csv(path)

    correct_filtercode = df["filtercode"] == filtercode

    df = df.loc[correct_filtercode]
    counts = df["oid"].value_counts()
    oid = counts.index[0]
    mask = df["oid"] == oid
    df = df.loc[mask]

    df = df[[time_col, mag_col, err_col]].apply(pd.to_numeric)
    df = df.sort_values(time_col)
    good = np.isfinite(df.to_numpy()).all(axis=1)
    df = df.loc[good]

    if mask_outliers:
        keep = outlier_mask(df[time_col].to_numpy(), df[mag_col].to_numpy(), df[err_col].to_numpy(), window=window, threshold=threshold, min_neighbours=min_neighbours)
        print(f"Outlier clipping removed {(~keep).sum()} of {len(keep)} points.")
        df = df.loc[keep]

    print(f"Loaded {len(df)} valid observations from {path} in band {filtercode}.")

    filter_settings = {
        "oid": int(oid),
        "filtercode": filtercode,
        "outlier_mask": mask_outliers,
        "clipping": {"window": window, "threshold": threshold, "min_neighbours": min_neighbours, "removed_points": int((~keep).sum())} if mask_outliers else None,
    }

    return AGNLC(oid, df[time_col].to_numpy(), df[mag_col].to_numpy(), df[err_col].to_numpy(), filter_settings)