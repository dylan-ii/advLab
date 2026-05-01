from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Scan:
    label: str
    x_mm: np.ndarray
    y: np.ndarray
    y_err: np.ndarray | None
    background: float


def load_laser_scans(xlsx):
    df = pd.read_excel(xlsx, sheet_name="laser", header=None)
    block_cols = {"both": (0, 1), "right": (3, 4), "left": (6, 7)}
    bg_cols = [(9, 10), (12, 13)]

    bg_vals = [
        pd.to_numeric(df.iloc[2:, cy], errors="coerce").dropna().mean()
        for _, cy in bg_cols
    ]
    background = float(np.mean(bg_vals))

    scans = {}
    for name, (cx, cy) in block_cols.items():
        x = pd.to_numeric(df.iloc[2:, cx], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df.iloc[2:, cy], errors="coerce").to_numpy(dtype=float)
        m = ~np.isnan(x) & ~np.isnan(y)
        scans[name] = Scan(f"laser: {name}", x[m], y[m] - background, None, background)
    return scans


def load_bulb_scans(xlsx):
    df = pd.read_excel(xlsx, sheet_name="bulb", header=None)
    blocks = {
        "both":  (5,  [6, 7, 8]),
        "right": (10, [11, 12, 13]),
        "left":  (15, [16, 17, 18]),
        "dark":  (20, [21, 22, 23]),
    }

    scans = {}
    for name, (cx, cys) in blocks.items():
        x = pd.to_numeric(df.iloc[5:, cx], errors="coerce").to_numpy(dtype=float)
        trials = np.array([
            pd.to_numeric(df.iloc[5:, cy], errors="coerce").to_numpy(dtype=float)
            for cy in cys
        ])
        if name in ("left", "right"):
            trials[trials > 3000] = np.nan

        with np.errstate(invalid="ignore"):
            mean = np.nanmean(trials, axis=0)
            n_ok = np.sum(~np.isnan(trials), axis=0)
            std = np.nanstd(trials, axis=0, ddof=1)
            sem = np.where(n_ok > 1, std / np.sqrt(n_ok), np.nan)

        ok = ~np.isnan(x) & ~np.isnan(mean)
        scans[name] = Scan(f"bulb: {name}", x[ok], mean[ok], sem[ok], 0.0)

    dark = scans["dark"]
    for k in ("both", "right", "left"):
        s = scans[k]
        bg = np.interp(s.x_mm, dark.x_mm, dark.y)
        scans[k] = Scan(s.label, s.x_mm, s.y - bg, s.y_err, float(np.nanmean(dark.y)))
    return scans


def load_pmt_optimization(xlsx):
    df = pd.read_excel(xlsx, sheet_name="bulb", header=None)
    hv_raw = pd.to_numeric(df.iloc[4:22, 0], errors="coerce").to_numpy(dtype=float)
    light = pd.to_numeric(df.iloc[4:22, 1], errors="coerce").to_numpy(dtype=float)
    dark = pd.to_numeric(df.iloc[4:22, 2], errors="coerce").to_numpy(dtype=float)

    hv = pd.Series(hv_raw).ffill().to_numpy()
    ok = ~np.isnan(light) & ~np.isnan(dark) & ~np.isnan(hv)
    return {"hv": hv[ok], "light": light[ok], "dark": dark[ok]}
