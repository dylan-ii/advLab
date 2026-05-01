import warnings

import numpy as np

from config import (XLSX_PATH, OUT_DIR, D2_MM, SLIT_SPACINGS_MM,
                    LASER_LAMBDA_NM, BULB_LAMBDA_NM)
from loaders import load_laser_scans, load_bulb_scans, load_pmt_optimization
from analysis import (fringe_spacing, lambda_from_spacing, superposition_test,
                      detect_duplicate_singles, fit_two_slit)
from plots import plot_raw, plot_extrema, plot_fraunhofer_fit, plot_pmt_plateau


def analyse(scans, lam_guess_nm, label, fname_prefix, d_card_mm=None):
    summary = {"label": label}
    duplicates = detect_duplicate_singles(scans)
    summary["duplicate_singles"] = duplicates

    plot_raw(scans,
             y_label=("Voltage (mV)" if label == "LASER" else "Counts / 1 s"),
             title=f"{label}: scans (background-corrected)",
             fname=OUT_DIR / f"{fname_prefix}_raw.png")

    both = scans["both"]
    xmax, ymax, xmin, ymin = plot_extrema(
        both, title=f"{label}: maxima/minima",
        fname=OUT_DIR / f"{fname_prefix}_extrema.png")

    central_x = float(xmax[np.argmax(ymax)])
    summary["central_max_x_mm"] = central_x
    print(f"[{label}] maxima (mm): {np.round(xmax, 2).tolist()}")
    print(f"[{label}] minima (mm): {np.round(xmin, 2).tolist()}")
    print(f"[{label}] central maximum: x = {central_x:.2f} mm, signal = {np.max(ymax):.1f}")

    dx, dx_err = fringe_spacing(xmax)
    summary["fringe_spacing_mm"] = dx
    summary["fringe_spacing_err_mm"] = dx_err
    print(f"[{label}] fringe spacing Δx = {dx:.3f} ± {dx_err:.3f} mm")

    lam_options = {}
    for tag, d_mm in SLIT_SPACINGS_MM.items():
        lam_nm, lam_err = lambda_from_spacing(dx, dx_err, d_mm, D2_MM)
        lam_options[tag] = (d_mm, lam_nm, lam_err)
        print(f"[{label}] slit #{tag} (d = {d_mm:.3f} mm) → λ = {lam_nm:.0f} ± {lam_err:.0f} nm")
    summary["lambda_options_nm"] = lam_options

    if d_card_mm is None:
        best_tag = min(SLIT_SPACINGS_MM,
                       key=lambda k: abs(
                           lambda_from_spacing(dx, dx_err,
                                               SLIT_SPACINGS_MM[k], D2_MM)[0]
                           - lam_guess_nm))
        d_card_mm = SLIT_SPACINGS_MM[best_tag]
    print(f"[{label}] d held fixed at {d_card_mm:.3f} mm for Fraunhofer fit")

    central_min_x = float(xmin[np.argmin(np.abs(xmin - central_x))])
    res_max = superposition_test(scans, central_x)
    res_min = superposition_test(scans, central_min_x)
    print(f"[{label}] x={central_x:.2f} (max): "
          f"both={res_max['both']:.2f}, L={res_max['left']:.2f}, "
          f"R={res_max['right']:.2f}, L+R={res_max['sum_singles']:.2f}, "
          f"both/(L+R)={res_max['ratio_4']:.2f}")
    print(f"[{label}] x={central_min_x:.2f} (min): "
          f"both={res_min['both']:.2f}, L={res_min['left']:.2f}, "
          f"R={res_min['right']:.2f}, L+R={res_min['sum_singles']:.2f}, "
          f"both/(L+R)={res_min['ratio_4']:.2f}")
    summary["super_at_max"] = res_max
    summary["super_at_min"] = res_min

    popt2, perr2 = fit_two_slit(both, lam_guess_nm, d_card_mm)
    I0, x0, a, d, lam, bg = popt2
    print(f"[{label}] fit I0 = {I0:.1f} ± {perr2[0]:.1f}")
    print(f"[{label}] fit x0 = {x0:.3f} ± {perr2[1]:.3f} mm")
    print(f"[{label}] fit a  = {a:.3f} ± {perr2[2]:.3f} mm")
    print(f"[{label}] fit d  = {d:.3f} mm (fixed)")
    print(f"[{label}] fit λ  = {lam:.1f} ± {perr2[4]:.1f} nm")
    print(f"[{label}] fit bg = {bg:.2f} ± {perr2[5]:.2f}")
    summary["fit"] = {"I0": I0, "x0": x0, "a_mm": a, "d_mm": d,
                      "lambda_nm": lam, "bg": bg, "errors": perr2.tolist()}

    plot_fraunhofer_fit(
        both, scans["left"], scans["right"], popt2,
        title=f"{label}: Fraunhofer two-slit fit (d = {d:.3f} mm)",
        fname=OUT_DIR / f"{fname_prefix}_fraunhofer.png",   
        skip_singles=tuple(duplicates))
    return summary


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    laser_scans = load_laser_scans(XLSX_PATH)
    bulb_scans = load_bulb_scans(XLSX_PATH)
    pmt_data = load_pmt_optimization(XLSX_PATH)

    hv_best, ratio_best, hv_grid, light_mean, dark_mean = plot_pmt_plateau(
        pmt_data, fname=OUT_DIR / "fig_pmt_plateau.png")

    print(f"[BULB] dark background subtracted: {bulb_scans['both'].background:.1f} counts/s")
    bulb_summary = analyse(bulb_scans, BULB_LAMBDA_NM, "BULB", "fig_bulb")

    inferred_d_mm = bulb_summary["fit"]["d_mm"]
    inferred_card = {round(v, 3): k for k, v in SLIT_SPACINGS_MM.items()}.get(
        round(inferred_d_mm, 3))
    print(f"[LASER] zero offset subtracted: {laser_scans['both'].background:.2f} mV")
    print(f"[LASER] d = {inferred_d_mm:.3f} mm (slit card #{inferred_card})")
    laser_summary = analyse(laser_scans, LASER_LAMBDA_NM, "LASER", "fig_laser",
                            d_card_mm=inferred_d_mm)

    rat = laser_summary["fringe_spacing_mm"] / bulb_summary["fringe_spacing_mm"]
    print(f"[CROSS] Δx_laser/Δx_bulb = {rat:.3f}")
    for lam_b in (541, 546, 551):
        print(f"[CROSS] λ_bulb = {lam_b} nm  →  λ_laser = {rat * lam_b:.0f} nm")

    fl = laser_summary["fit"]
    fb = bulb_summary["fit"]
    print(f"{'parameter':<12} {'LASER':>22} {'BULB':>22}")
    print(f"{'λ (nm)':<12} {fl['lambda_nm']:>10.1f} ± {fl['errors'][4]:<8.1f} "
          f"{fb['lambda_nm']:>10.1f} ± {fb['errors'][4]:<8.1f}")
    print(f"{'a (mm)':<12} {fl['a_mm']:>10.3f} ± {fl['errors'][2]:<8.3f} "
          f"{fb['a_mm']:>10.3f} ± {fb['errors'][2]:<8.3f}")
    print(f"{'x0 (mm)':<12} {fl['x0']:>10.3f} ± {fl['errors'][1]:<8.3f} "
          f"{fb['x0']:>10.3f} ± {fb['errors'][1]:<8.3f}")
    print(f"{'Δx (mm)':<12} "
          f"{laser_summary['fringe_spacing_mm']:>10.3f} ± "
          f"{laser_summary['fringe_spacing_err_mm']:<8.3f} "
          f"{bulb_summary['fringe_spacing_mm']:>10.3f} ± "
          f"{bulb_summary['fringe_spacing_err_mm']:<8.3f}")


if __name__ == "__main__":
    main()
