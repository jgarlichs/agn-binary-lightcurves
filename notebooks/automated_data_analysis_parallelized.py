# data analysis loop for given binary candidates
# lightcurve data needs to be downloaded in advance (see import_data.ipynb)

import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from agn_analysis.data import load_lc
from agn_analysis.fitting import fit_model, get_result_path
from agn_analysis.plotting import plot_corner, plot_fitted_model

SCRIPT_DIR = Path(__file__).resolve().parent
LIGHTCURVE_PATH = SCRIPT_DIR.parent / "data" / "lightcurves"
RESULTS_PATH = SCRIPT_DIR.parent / "data" / "results"

OBJECTS = None #["UM211"] # list of objects to analyse (None if all)

MODELS = ["red", "red_qpo_sine"] # list of models used

NLIVE = 500
BATCHES = 5
SEED = 42

OVERWRITE = True
BAND = "zg"
N_WOKRERS = 10
MAKE_PLOTS = True

def run_fit(path: Path, model: str) -> dict:
    """ Load one lightcurve, fit one model, save results and plots.
    Runs inside a worker process and returns a small summary dict."""

    start = time.perf_counter()
    summary = {"target": path.stem, "model": model}
    try:
        lc = load_lc(path, "mjd", "mag", "magerr", BAND)
        summary.update(oid=int(lc.oid), n=lc.n)

        result_file = get_result_path(lc, SEED) / f"{model}_results.json"
        if not OVERWRITE and result_file.exists():
            summary["status"] = "skipped"
        else:
            result = fit_model(lc, model, NLIVE, BATCHES, SEED, print_progress=False)
            if MAKE_PLOTS:
                plot_fitted_model(lc, result, show=False)
                plot_corner(lc, model, SEED, show=False)
            summary.update(status="ok", lnZ=result["lnZ"], lnZ_err=result["lnZ_err"])
    except Exception:
        summary.update(status="failed", error=traceback.format_exc(limit=3))

    summary["minutes"] = round((time.perf_counter() - start) / 60, 2)
    return summary

def light_curve_size(path: Path) -> int:
    """Number of points after cleaning, used to start the longest fits first."""
    try:
        return load_lc(path, "mjd", "mag", "magerr", BAND).n
    except Exception:
        return 0

def main():
    if OBJECTS is None:
        files = sorted(LIGHTCURVE_PATH.glob("*.csv"))
    else:
        files = [p for name in OBJECTS for p in LIGHTCURVE_PATH.glob(f"*{name}*.csv")]
    if not files:
        raise SystemExit(f"No lightcurves found in {LIGHTCURVE_PATH} for {OBJECTS}.")

    # largest lc first
    sizes = {f: light_curve_size(f) for f in files}
    jobs = sorted(((f, m) for f in files for m in MODELS),
                  key=lambda job: sizes[job[0]], reverse=True)

    print(f"Running {len(jobs)} fits on {N_WOKRERS} workers.")

    summaries = []
    with ProcessPoolExecutor(max_workers=N_WOKRERS) as pool:
        futures = [pool.submit(run_fit, f, m) for f, m in jobs]
        for k, fut in enumerate(as_completed(futures), start=1):
            s = fut.result()
            summaries.append(s)
            info = f"lnZ = {s['lnZ']:.2f} ± {s['lnZ_err']:.2f}" if s["status"] == "ok" else s["status"]
            print(f"[{k}/{len(jobs)}] {s['target']} / {s['model']}: {info} ({s['minutes']} min)")
            if s["status"] == "failed":
                print(s["error"])

    summary = pd.DataFrame(summaries).sort_values(["target", "model"])
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_PATH / f"summary_{BAND}_{SEED}.csv", index=False)
    print(f"Done. Summary written to {RESULTS_PATH / f'summary_{BAND}_{SEED}.csv'}")

if __name__ == "__main__":
    main()