import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from config import D2_MM, NOMINAL_SLIT_WIDTH_MM


def find_extrema(x, y, min_sep_mm=0.4):
    dx = np.median(np.diff(x))
    distance = max(1, int(round(min_sep_mm / dx)))
    prom = 0.05 * (np.nanmax(y) - np.nanmin(y))
    pk_max, _ = find_peaks(y, distance=distance, prominence=prom)
    pk_min, _ = find_peaks(-y, distance=distance, prominence=prom)
    return x[pk_max], y[pk_max], x[pk_min], y[pk_min]


def fringe_spacing(max_positions):
    diffs = np.diff(np.sort(max_positions))
    return float(np.mean(diffs)), float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))


def lambda_from_spacing(dx_mm, dx_err_mm, d_mm, D2_mm):
    lam_mm = d_mm * dx_mm / D2_mm
    lam_err_mm = d_mm * dx_err_mm / D2_mm
    return lam_mm * 1e6, lam_err_mm * 1e6


def superposition_test(scans, at_x_mm):
    out = {k: float(np.interp(at_x_mm, scans[k].x_mm, scans[k].y))
           for k in ("both", "left", "right")}
    out["sum_singles"] = out["left"] + out["right"]
    out["ratio_4"] = out["both"] / max(out["sum_singles"], 1e-12)
    return out


def detect_duplicate_singles(scans, tol_pct=8.0):
    bad = []
    both = scans["both"]
    for k in ("left", "right"):
        s = scans[k]
        if min(len(both.x_mm), len(s.x_mm)) < 5:
            continue
        ys = np.interp(both.x_mm, s.x_mm, s.y)
        ref = np.maximum(np.abs(both.y), np.percentile(np.abs(both.y), 50))
        rel = np.median(np.abs(both.y - ys) / ref) * 100
        if rel < tol_pct:
            bad.append(k)
    return bad


def two_slit_fraunhofer(x_mm, I0, x0, a_mm, d_mm, lam_nm, D2_mm, bg):
    lam_mm = lam_nm * 1e-6
    theta = (x_mm - x0) / D2_mm
    alpha = np.pi * a_mm * np.sin(theta) / lam_mm
    beta = np.pi * d_mm * np.sin(theta) / lam_mm
    sinc2 = np.where(np.abs(alpha) < 1e-12, 1.0, (np.sin(alpha) / alpha) ** 2)
    return I0 * sinc2 * np.cos(beta) ** 2 + bg


def one_slit_fraunhofer(x_mm, I0, x0, a_mm, lam_nm, D2_mm, bg):
    lam_mm = lam_nm * 1e-6
    theta = (x_mm - x0) / D2_mm
    alpha = np.pi * a_mm * np.sin(theta) / lam_mm
    sinc2 = np.where(np.abs(alpha) < 1e-12, 1.0, (np.sin(alpha) / alpha) ** 2)
    return I0 * sinc2 + bg


def fit_two_slit(scan, lam_guess_nm, d_fixed_mm):
    x, y = scan.x_mm, scan.y
    p0 = [np.max(y), x[np.argmax(y)], NOMINAL_SLIT_WIDTH_MM, lam_guess_nm, 0.0]
    bounds = (
        [0,            x.min(), 0.01, 0.5 * lam_guess_nm, -np.max(y)],
        [10*np.max(y), x.max(), 0.5,  2.0 * lam_guess_nm,  np.max(y)],
    )
    fn = lambda x, I0, x0, a, lam, bg: two_slit_fraunhofer(
        x, I0, x0, a, d_fixed_mm, lam, D2_MM, bg)
    popt, pcov = curve_fit(fn, x, y, p0=p0, bounds=bounds,
                           sigma=scan.y_err, absolute_sigma=False, maxfev=20000)
    err5 = np.sqrt(np.diag(pcov))
    popt6 = np.array([popt[0], popt[1], popt[2], d_fixed_mm, popt[3], popt[4]])
    perr6 = np.array([err5[0], err5[1], err5[2], 0.0, err5[3], err5[4]])
    return popt6, perr6


def fit_one_slit(scan, lam_guess_nm):
    x, y = scan.x_mm, scan.y
    p0 = [np.max(y), x[np.argmax(y)], NOMINAL_SLIT_WIDTH_MM, lam_guess_nm, 0.0]
    bounds = (
        [0,            x.min(), 0.01, 0.5 * lam_guess_nm, -np.max(y)],
        [10*np.max(y), x.max(), 0.5,  2.0 * lam_guess_nm,  np.max(y)],
    )
    fn = lambda x, I0, x0, a, lam, bg: one_slit_fraunhofer(
        x, I0, x0, a, lam, D2_MM, bg)
    popt, pcov = curve_fit(fn, x, y, p0=p0, bounds=bounds,
                           sigma=scan.y_err, absolute_sigma=False, maxfev=20000)
    return popt, np.sqrt(np.diag(pcov))
