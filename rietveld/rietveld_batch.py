#!/usr/bin/env python
"""
rietveld_batch.py  --  automated, staged Rietveld refinement of a folder of
Bruker .raw powder patterns with GSAS-II (GSASIIscriptable).

Designed for (Ni,Fe,Co,Cr,Mn) high-entropy spinel catalysts measured on a Cu
lab diffractometer, but works for any phase set you give it via CIFs.

What it does, per pattern:
    1. load the data (GSAS-II's Bruker reader; falls back to bruker_raw.py)
    2. add the phase(s) from CIF
    3. STAGED refinement, converging between each step (the order matters):
         scale -> background -> sample displacement -> unit cell
               -> size + microstrain -> (optional) preferred orientation
    4. extract: lattice parameter(s), crystallite size, microstrain,
       weight fractions (multiphase), Rwp
    5. save a .gpx project + an obs/calc/diff plot
Then it writes one results.csv for the whole folder and flags any pattern whose
Rwp exceeds a threshold for manual review.

IMPORTANT caveats baked in:
  * No LaB6/Si standard yet -> size & microstrain are RELATIVE (instrument
    broadening is approximate). Columns are labelled accordingly.
  * Cu radiation on Cr/Mn/Fe/Co fluoresces heavily -> large background; the
    Chebyshev background order is set generously. A Kbeta filter / energy-
    discriminating detector would improve size/strain a lot.
  * Weight fractions follow Hill & Howard (scale x cell-mass, normalised).

Usage:
    python rietveld_batch.py            # uses the CONFIG block below
    python rietveld_batch.py  DATA_DIR  CIF1 [CIF2 ...]
"""
import os, sys, glob, csv, traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIG  -- edit these, or pass DATA_DIR + CIFs on the command line
# ----------------------------------------------------------------------------
CONFIG = dict(
    data_dir   = ".",                       # folder of .raw files
    pattern    = "*.raw",                   # which files
    phases     = ["magnetite.cif"],         # one entry per phase; add more for mixtures
    phasenames = None,                       # optional pretty names, else from CIF
    instprm    = "default_Cu_lab.instprm",  # Cu lab instrument parameter file
    out_dir    = "rietveld_out",
    twotheta_limits = (15.0, 90.0),         # fit window (deg); trims low-angle junk
    background_order = 8,                    # Chebyshev terms (high -> absorbs fluorescence)
    refine_size      = True,
    refine_strain    = True,
    refine_prefor    = False,                # March-Dollase; turn on for platy/layered
    prefor_hkl       = (3, 1, 1),            # preferred-orientation axis if used
    max_cycles       = 8,
    rwp_flag         = 12.0,                 # flag patterns with Rwp above this (%)
    size_is_relative = True,                 # True until a LaB6/Si standard exists
)

# ----------------------------------------------------------------------------
# locate GSAS-II (pip install, or a g2full/standalone checkout)
# ----------------------------------------------------------------------------
def _import_g2():
    import importlib
    # 1) already importable? (pip install, or already on PYTHONPATH)
    for mod in ("GSASIIscriptable", "GSASII.GSASIIscriptable"):
        try:
            return importlib.import_module(mod)
        except ImportError:
            pass
    # 2) locate a GSAS-II install dir (the folder that CONTAINS the 'GSASII'
    #    package and a 'backcompat' shim). Covers the gsas2main one-click
    #    installer and common manual layouts.
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    bases = [os.environ.get("GSASII_HOME")] + [
        rf"C:\Users\{user}\AppData\Local\gsas2main\GSAS-II",
        rf"C:\Users\{user}\AppData\Local\gsas2full\GSAS-II",
        r"C:\GSAS-II", r"C:\G2\GSAS-II", r"C:\g2full\GSAS-II",
        os.path.expanduser("~/.gsas2main/GSAS-II"),
        os.path.expanduser("~/g2full/GSAS-II"),
        os.path.expanduser("~/GSAS-II"),
    ]
    for base in bases:
        if not base or not os.path.isdir(base):
            continue
        # add the dir holding the GSASII package, plus its backcompat shim
        for p in (base, os.path.join(base, "backcompat")):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        for mod in ("GSASII.GSASIIscriptable", "GSASIIscriptable"):
            try:
                return importlib.import_module(mod)
            except ImportError:
                continue
    raise ImportError(
        "Could not import GSASIIscriptable. Run this with GSAS-II's bundled Python "
        "and, if needed, set GSASII_HOME to the GSAS-II folder (the one containing "
        "the 'GSASII' package), e.g. "
        r"C:\Users\<you>\AppData\Local\gsas2main\GSAS-II.")


# ----------------------------------------------------------------------------
# data loading: GSAS-II native Bruker reader, with bruker_raw.py fallback
# ----------------------------------------------------------------------------
def load_histogram(gpx, rawfile, instprm, out_dir):
    """Add a powder histogram from a Bruker .raw file; return the hist object."""
    # 1) try GSAS-II's own Bruker RAW importer
    for hint in ("Bruker RAW", None):
        try:
            return gpx.add_powder_histogram(rawfile, instprm, fmthint=hint)
        except Exception:
            continue
    # 2) fallback: parse ourselves -> .xy -> import as 2-column text
    import bruker_raw
    x, y, meta = bruker_raw.read_raw(rawfile)
    xy = os.path.join(out_dir, os.path.splitext(os.path.basename(rawfile))[0] + ".xy")
    bruker_raw.write_xy(xy, x, y, header=os.path.basename(rawfile))
    for hint in ("Topas xye", "GSAS-II text file", "2 column", None):
        try:
            return gpx.add_powder_histogram(xy, instprm, fmthint=hint)
        except Exception:
            continue
    raise RuntimeError(f"Could not import {rawfile} by any route.")


# ----------------------------------------------------------------------------
# staged refinement
# ----------------------------------------------------------------------------
def refine_staged(gpx, hist, phases, cfg):
    """Run the staged refinement in the documented GSAS-II order."""
    lo, hi = cfg["twotheta_limits"]
    hist.set_refinements({"Limits": [lo, hi]})
    gpx.data["Controls"]["data"]["max cyc"] = cfg["max_cycles"]

    # Stage 1: scale only (phase scale factors are on by default) + a flat bkg
    gpx.do_refinements([{}])

    # Stage 2: background (Chebyshev)
    gpx.do_refinements([{"set": {"Background": {"type": "chebyschev-1",
                                                "no. coeffs": cfg["background_order"],
                                                "refine": True}}}])

    # NOTE: sample displacement (Bragg-Brentano height error) is intentionally
    # NOT refined. It is ~96% correlated with the lattice parameter: refining
    # both makes the cell metric tensor unstable ("Invalid metric tensor"), AND
    # it splits each sample's peak-position shift arbitrarily between cell and
    # displacement, which destroys sample-to-sample comparability (all cells
    # collapse to one value). With displacement fixed at 0, every peak-position
    # difference goes into the cell, so RELATIVE lattice parameters across a
    # series are meaningful. Absolute values then carry one constant displacement
    # offset -- calibrate it out later with a LaB6/Si standard.

    # Stage 3: unit cell(s)  (stays refined through the end -> proper esd)
    gpx.set_refinement({"set": {"Cell": True}})
    gpx.do_refinements([{}])

    # Stage 4: size, then microstrain (isotropic), each seeded at a physical value
    # so the least-squares starts near the answer instead of railing to a limit.
    # (Size value is Size[1][0] in microns; Mustrain value is Mustrain[1][0].)
    if cfg["refine_size"]:
        for ph in phases:
            ph.set_HAP_refinements({"Size": {"type": "isotropic", "refine": True}},
                                   histograms=[hist])
            try:
                ph.data["Histograms"][hist.name]["Size"][1][0] = 0.010   # ~10 nm
            except Exception:
                pass
        gpx.do_refinements([{}])
    if cfg["refine_strain"]:
        for ph in phases:
            ph.set_HAP_refinements({"Mustrain": {"type": "isotropic", "refine": True}},
                                   histograms=[hist])
            try:
                ph.data["Histograms"][hist.name]["Mustrain"][1][0] = 1000.0  # ~0.1%
            except Exception:
                pass
        gpx.do_refinements([{}])

    # Stage 6 (optional): preferred orientation, March-Dollase
    if cfg["refine_prefor"]:
        for ph in phases:
            try:
                ph.set_HAP_refinements(
                    {"Pref.Ori.": ["MD", True, list(cfg["prefor_hkl"])]},
                    histograms=[hist])
            except Exception:
                pass
        gpx.do_refinements([{}])


# ----------------------------------------------------------------------------
# result extraction  (paths confirmed/adjusted against live objects on first run)
# ----------------------------------------------------------------------------
def weight_fractions(phases, hist):
    """Hill & Howard: W_p proportional to (phase scale) x (unit-cell mass)."""
    hn = hist.name
    num, out = {}, {}
    for ph in phases:
        try:
            scale = ph.data["Histograms"][hn]["Scale"][0]
            mass  = ph.data["General"]["Mass"]        # mass of unit-cell contents
            num[ph.name] = scale * mass
        except Exception:
            num[ph.name] = np.nan
    tot = np.nansum(list(num.values()))
    for k, v in num.items():
        out[k] = 100.0 * v / tot if tot and np.isfinite(v) else np.nan
    return out


def extract(ph, hist):
    row = {}
    # lattice parameter(s) + esd  -- GSAS-II keys are length_a / volume
    try:
        cell, esd = ph.get_cell_and_esd()
        row["a"], row["a_esd"] = cell["length_a"], esd["length_a"]
        row["vol"] = cell["volume"]
    except Exception:
        try:
            c = ph.get_cell()
            row["a"], row["vol"] = c["length_a"], c["volume"]
        except Exception:
            row["a"] = row["vol"] = np.nan
    hapd = ph.data["Histograms"].get(hist.name, {})
    # crystallite size: GSAS-II isotropic Size value is in microns -> nm.
    # It rails to ~0.001 um (1 nm, the lower limit) if the fit can't resolve it.
    try:
        size_um = hapd["Size"][1][0]
        row["size_nm"] = size_um * 1000.0
        row["size_railed"] = size_um <= 0.00101
    except Exception:
        row["size_nm"] = np.nan
    # isotropic microstrain: GSAS-II value is in units of 1e-6 -> percent
    try:
        row["microstrain_pct"] = hapd["Mustrain"][1][0] / 1e6 * 100.0
    except Exception:
        row["microstrain_pct"] = np.nan
    return row


def export_profile(hist, path):
    """Write the fit profile as CSV columns ready for Origin:
    2theta, Yobs, Ycalc, difference (obs-calc), background, Yobs esd."""
    try:
        x   = np.array(hist.getdata("x"))
        yo  = np.array(hist.getdata("yobs"))
        yc  = np.array(hist.getdata("ycalc"))
        bg  = np.array(hist.getdata("background"))
    except Exception as e:
        print("   (profile export skipped:", e, ")"); return
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["2theta_deg", "Yobs", "Ycalc", "Yobs_minus_Ycalc",
                    "Background", "Yobs_esd"])
        for i in range(len(x)):
            w.writerow([f"{x[i]:.5f}", f"{yo[i]:.3f}", f"{yc[i]:.3f}",
                        f"{yo[i]-yc[i]:.3f}", f"{bg[i]:.3f}",
                        f"{np.sqrt(max(yo[i],0.0)):.3f}"])


def export_reflections(phases, path, lam=1.540598, tth_range=(5.0, 90.0)):
    """Write Bragg tick positions (2theta) per phase, computed from the refined
    cubic cell. One file; a 'phase' column distinguishes phases if multiphase."""
    rows = []
    for ph in phases:
        try:
            a = ph.get_cell()["length_a"]
        except Exception:
            continue
        for h in range(0, 10):
            for k in range(0, h + 1):
                for l in range(0, k + 1):
                    s = h*h + k*k + l*l
                    if s == 0:
                        continue
                    allodd  = (h % 2 and k % 2 and l % 2)
                    alleven = (h % 2 == 0 and k % 2 == 0 and l % 2 == 0)
                    if not (allodd or (alleven and (h + k + l) % 4 == 0)):
                        continue  # Fd-3m spinel reflection conditions
                    d = a / np.sqrt(s)
                    val = lam / (2 * d)
                    if val <= 1:
                        tt = np.degrees(2 * np.arcsin(val))
                        if tth_range[0] <= tt <= tth_range[1]:
                            rows.append((ph.name, tt, round(d, 4), f"{h}{k}{l}"))
    rows.sort(key=lambda r: (r[0], r[1]))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase", "2theta_deg", "d_Angstrom", "hkl"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", r[2], r[3]])


def _spinel_ticks(a, lam, lo, hi):
    """Bragg 2theta positions for a cubic Fd-3m spinel of cell a, within [lo,hi]."""
    tt = []
    for h in range(0, 10):
        for k in range(0, h + 1):
            for l in range(0, k + 1):
                s = h*h + k*k + l*l
                if s == 0:
                    continue
                allodd  = (h % 2 and k % 2 and l % 2)
                alleven = (h % 2 == 0 and k % 2 == 0 and l % 2 == 0)
                if not (allodd or (alleven and (h + k + l) % 4 == 0)):
                    continue
                val = lam / (2 * (a / np.sqrt(s)))
                if val <= 1:
                    t = np.degrees(2 * np.arcsin(val))
                    if lo <= t <= hi:
                        tt.append(t)
    return sorted(tt)


def plot_fit(hist, phases, path, title, tth_range=(15.0, 90.0), lam=1.540598):
    """obs/calc/difference/background plot, trimmed to the fit window, with a
    row of Bragg tick marks per phase."""
    try:
        x   = np.array(hist.getdata("x"))
        yo  = np.array(hist.getdata("yobs"))
        yc  = np.array(hist.getdata("ycalc"))
        bkg = np.array(hist.getdata("background"))
    except Exception as e:
        print("   (plot skipped:", e, ")"); return
    lo, hi = tth_range
    m = (x >= lo) & (x <= hi)
    x, yo, yc, bkg = x[m], yo[m], yc[m], bkg[m]
    if len(x) == 0:
        return
    rng = max(yo.max() - yo.min(), 1.0)
    off = yo.min() - 0.20 * rng
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, yo, ".", ms=2, color="#1b3a6b", label="obs")
    ax.plot(x, yc, "-", lw=1.0, color="#c0392b", label="calc")
    ax.plot(x, bkg, "-", lw=0.8, color="#7f8c8d", label="background")
    ax.plot(x, (yo - yc) + off, "-", lw=0.6, color="#27ae60", label="difference")
    ax.axhline(off, color="k", lw=0.4)
    tickcols = ["#2c3e50", "#8e44ad", "#1f618d"]
    for pi, ph in enumerate(phases):
        try:
            c = ph.get_cell()
            a = c["length_a"]
            cubic = (abs(c["length_a"] - c["length_b"]) < 1e-3 and
                     abs(c["length_a"] - c["length_c"]) < 1e-3 and
                     abs(c["angle_alpha"] - 90) < 1e-2 and abs(c["angle_beta"] - 90) < 1e-2)
        except Exception:
            continue
        # cubic-spinel tick positions are computed analytically; for other
        # symmetries we skip ticks (would need the full reflection list)
        if not cubic:
            continue
        ticks = _spinel_ticks(a, lam, lo, hi)
        yrow = off - 0.09 * rng - pi * 0.06 * rng
        ax.plot(ticks, [yrow] * len(ticks), "|", ms=7, mew=1.0,
                color=tickcols[pi % len(tickcols)],
                label=(f"{ph.name} Bragg" if len(phases) > 1 else "Bragg"))
    ax.set_xlim(lo, hi); ax.set_xlabel("2theta (deg)"); ax.set_ylabel("counts")
    ax.set_title(title); ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    try:
        plt.tight_layout(); plt.savefig(path, dpi=120)
    except Exception as e:
        # non-fatal: e.g. the PNG is open/locked in an image viewer
        print("   (plot save failed:", e, "- is it open in a viewer?)")
    finally:
        plt.close(fig)


# ----------------------------------------------------------------------------
# main batch
# ----------------------------------------------------------------------------
def run(cfg):
    G2sc = _import_g2()
    os.makedirs(cfg["out_dir"], exist_ok=True)
    files = sorted(glob.glob(os.path.join(cfg["data_dir"], cfg["pattern"])))
    if not files:
        print("No files match", os.path.join(cfg["data_dir"], cfg["pattern"])); return
    print(f"Found {len(files)} pattern(s). Phases: {cfg['phases']}\n")

    results, flagged = [], []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        print(f"=== {name} ===")
        try:
            gpx = G2sc.G2Project(newgpx=os.path.join(cfg["out_dir"], name + ".gpx"))
            hist = load_histogram(gpx, f, cfg["instprm"], cfg["out_dir"])
            phases = []
            for i, cif in enumerate(cfg["phases"]):
                pn = (cfg["phasenames"] or [])[i] if cfg["phasenames"] else None
                phases.append(gpx.add_phase(cif, phasename=pn, histograms=[hist], fmthint="CIF"))
            refine_staged(gpx, hist, phases, cfg)
            gpx.save()

            rwp = hist.get_wR()
            wts = weight_fractions(phases, hist) if len(phases) > 1 else {phases[0].name: 100.0}
            for ph in phases:
                row = dict(file=name, phase=ph.name, Rwp=round(rwp, 3),
                           wt_pct=round(wts.get(ph.name, np.nan), 2))
                row.update(extract(ph, hist))
                results.append(row)
            plot_fit(hist, phases, os.path.join(cfg["out_dir"], name + "_fit.png"),
                     f"{name}   Rwp={rwp:.2f}%", tth_range=cfg["twotheta_limits"])
            # Origin-ready exports: fit profile columns + Bragg tick positions
            export_profile(hist, os.path.join(cfg["out_dir"], name + "_profile.csv"))
            export_reflections(phases, os.path.join(cfg["out_dir"], name + "_reflections.csv"),
                               tth_range=cfg["twotheta_limits"])
            print(f"   Rwp={rwp:.2f}%  " +
                  "  ".join(f"{p.name}:a={extract(p,hist).get('a',float('nan')):.4f}" for p in phases))
            if rwp > cfg["rwp_flag"]:
                flagged.append((name, rwp))
        except Exception:
            print("   FAILED:\n" + "      ".join(traceback.format_exc().splitlines(True)))
            results.append(dict(file=name, phase="ERROR", Rwp=np.nan, wt_pct=np.nan))
            flagged.append((name, "error"))

    # write CSV
    cols = ["file", "phase", "Rwp", "wt_pct", "a", "a_esd", "vol",
            "size_nm", "microstrain_pct"]
    csv_path = os.path.join(cfg["out_dir"], "results.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    note = " (size/strain are RELATIVE: no instrument standard)" if cfg["size_is_relative"] else ""
    print(f"\nWrote {csv_path}{note}")
    if flagged:
        print("FLAGGED for manual review (high Rwp / error):")
        for n, r in flagged:
            print(f"   {n}: {r}")


if __name__ == "__main__":
    cfg = dict(CONFIG)
    if len(sys.argv) >= 2:
        cfg["data_dir"] = sys.argv[1]
    if len(sys.argv) >= 3:
        cfg["phases"] = sys.argv[2:]
    try:
        run(cfg)
    except ImportError as e:
        print("\n" + str(e))
        sys.exit(1)
