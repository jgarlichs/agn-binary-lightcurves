# data analysis loop for given binary candidates
# lightcurve data needs to be downloaded in advance (se import_data.ipynb)

from pathlib import Path
import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt
SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_PATH = SCRIPT_DIR.parent / "src" / "light.mplstyle"
plt.style.use(STYLE_PATH)
from agn_analysis.utils import AGNLC, load_lc, fit_model, plot_fitted_model, plot_corner

OBJECTS = ["UM211"] # list of objects to analyse (None if all)
LIGHTCURVE_PATH = SCRIPT_DIR.parent / "data" / "lightcurves"
RESULTS_PATH = SCRIPT_DIR.parent / "data" / "results"

MODELS = ["red", "red_qpo_sine"] # list of models used

NLIVE = 100
BATCHES = 3
SEED = 42

OVERWRITE = True
BAND = "zg"

if OBJECTS is None:
    files = [f for f in LIGHTCURVE_PATH.iterdir() if f.is_file()]
else:
    files = [p for oid in OBJECTS for p in LIGHTCURVE_PATH.glob(f"*{oid}*")]

for lcdata in files:
    lc = load_lc(SCRIPT_DIR.parent / "data" / lcdata, "mjd", "mag", "magerr", BAND)
    for model in MODELS:
        result_file = RESULTS_PATH / str(lc.oid) / f"{model}_results.json"
        if not OVERWRITE and result_file.exists():
            print(f"Skipping {lc.oid} / {model}: already fitted.")
            continue

        print(f"Fitting {model}-model to LC data:")
        result = fit_model(lc, model, NLIVE, BATCHES, SEED, outdir=RESULTS_PATH)
        plot_fitted_model(lc, result, show=False)
        plot_corner(lc, model, show=False)
