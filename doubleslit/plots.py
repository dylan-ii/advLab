import numpy as np
import matplotlib.pyplot as plt

from config import D2_MM
from analysis import (find_extrema, two_slit_fraunhofer, one_slit_fraunhofer,
                      fit_one_slit)


plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

_COLOURS = {"both": "tab:blue", "left": "tab:green", "right": "tab:orange"}


def plot_raw(scans, y_label, title, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    for k in ("both", "left", "right"):
        s = scans[k]
        if s.y_err is not None:
            ax.errorbar(s.x_mm, s.y, yerr=s.y_err, fmt="o-", ms=3, lw=1,
                        capsize=2, color=_COLOURS[k], label=k.capitalize())
        else:
            ax.plot(s.x_mm, s.y, "o-", ms=3, lw=1,
                    color=_COLOURS[k], label=k.capitalize())
    ax.set_xlabel("Detector-slit position (mm)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def plot_extrema(scan, title, fname):
    xmax, ymax, xmin, ymin = find_extrema(scan.x_mm, scan.y)
    fig, ax = plt.subplots(figsize=(9, 5))
    if scan.y_err is not None:
        ax.errorbar(scan.x_mm, scan.y, yerr=scan.y_err, fmt="o-",
                    ms=3, lw=1, capsize=2, color="tab:blue")
    else:
        ax.plot(scan.x_mm, scan.y, "o-", ms=3, lw=1, color="tab:blue")
    ax.plot(xmax, ymax, "r^", ms=10, label=f"maxima ({len(xmax)})")
    ax.plot(xmin, ymin, "kv", ms=10, label=f"minima ({len(xmin)})")
    ax.set_xlabel("Detector-slit position (mm)")
    ax.set_ylabel("Background-corrected signal")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return xmax, ymax, xmin, ymin


def plot_fraunhofer_fit(scan_both, scan_left, scan_right, popt2, title, fname,
                        skip_singles=()):
    I0, x0, a, d, lam, bg = popt2
    xx = np.linspace(scan_both.x_mm.min(), scan_both.x_mm.max(), 1000)
    yy = two_slit_fraunhofer(xx, I0, x0, a, d, lam, D2_MM, bg)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax = axes[0]
    ax.plot(scan_both.x_mm, scan_both.y, "o", ms=3, color="tab:blue",
            label="Both slits")
    ax.plot(xx, yy, "-", lw=1.5, color="tab:red",
            label=(f"Fraunhofer fit:\n"
                   f"  a = {a:.3f} mm,  d = {d:.3f} mm\n"
                   f"  λ = {lam:.0f} nm,  x₀ = {x0:.2f} mm"))
    ax.set_ylabel("Signal (counts/s)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    ax = axes[1]
    for s, name, c in [(scan_left, "left", "tab:green"),
                       (scan_right, "right", "tab:orange")]:
        suspect = name in skip_singles
        lab = (f"{name.capitalize()} slit (suspect duplicate)" if suspect
               else f"{name.capitalize()} slit ")
        ax.plot(s.x_mm, s.y, "o", ms=3, color=c, label=lab)
        if suspect:
            continue
        try:
            fit, _ = fit_one_slit(s, lam)
            I0s, x0s, asng, lamn, bgs = fit
            yfit = one_slit_fraunhofer(xx, I0s, x0s, asng, lamn, D2_MM, bgs)
            ax.plot(xx, yfit, "-", lw=1.2, color=c,
                    label=f"{name.capitalize()} fit (a={asng:.3f} mm)")
        except Exception:
            pass
    ax.set_xlabel("Detector-slit position (mm)")
    ax.set_ylabel("Signal")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def plot_pmt_plateau(pmt, fname):
    hv_unique = np.array(sorted(set(pmt["hv"].tolist())))
    light_mean = np.array([pmt["light"][pmt["hv"] == v].mean() for v in hv_unique])
    dark_mean = np.array([pmt["dark"][pmt["hv"] == v].mean() for v in hv_unique])
    ratio_mean = light_mean / np.maximum(dark_mean, 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(pmt["hv"], pmt["light"], "o", color="tab:red", alpha=0.4,
                label="Shutter open (replicates)")
    ax.semilogy(pmt["hv"], pmt["dark"], "s", color="tab:blue", alpha=0.4,
                label="Shutter closed (replicates)")
    ax.semilogy(hv_unique, light_mean, "-", color="tab:red", lw=2,
                label="Shutter open (mean)")
    ax.semilogy(hv_unique, dark_mean, "-", color="tab:blue", lw=2,
                label="Shutter closed (mean)")

    i_best = int(np.argmax(ratio_mean))
    ax.axvline(hv_unique[i_best], color="k", ls="--", lw=1)
    ax.text(hv_unique[i_best], dark_mean[i_best],
            f"  best L/D = {ratio_mean[i_best]:.1f} @ HV = {hv_unique[i_best]:.0f} V",
            va="bottom", fontsize=9)
    ax.set_xlabel("PMT bias indicator V")
    ax.set_ylabel("Counts in 1-s gate")
    ax.set_title("PMT HV Scan")
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return hv_unique[i_best], ratio_mean[i_best], hv_unique, light_mean, dark_mean
