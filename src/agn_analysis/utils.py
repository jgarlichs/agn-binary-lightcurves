# AGNLC class and functions
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from pathlib import Path
import pandas as pd
import dynesty
from dynesty.utils import resample_equal
import corner
import json
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_PATH = SCRIPT_DIR.parent / "light.mplstyle"
plt.style.use(STYLE_PATH)

# different hypothesis models
MODELS = ("red", "red_sine", "red_qpo", "drw", "drw_sine", "red_qpo_sine")

PRIOR_RANGES = {
    "nu": (0.1, 2.0),
    "log_sigmahat2": (-6.0, 0.0),
    "log_tau0": (-4.0, 4.0),
    "gamma": (0.05, 2.0),
    "A": (0.0, 0.5),
    "phi": (0.0, 2 * np.pi),
    "T0": (0.3, 10.0),
    "Tq": (0.3, 10.0)
}

def theta_names(model: str):
    if model not in MODELS:
        raise ValueError(f"Unknown model.")
    names = ["m", "nu", "log_sigmahat2", "log_tau0"]
    if not model.startswith("drw"):
        names.append("gamma")
    if "sine" in model:
        names.extend(["A", "phi", "T0"])
    if "qpo" in model:
        names.append("Tq")
    return names

class AGNLC:
    def __init__(self, oid: int, t_mjd: np.ndarray, mag: np.ndarray, magerr: np.ndarray):
        t_mjd = np.asarray(t_mjd, dtype=float)
        self.y = np.asarray(mag, dtype=float)
        self.dy = np.asarray(magerr, dtype=float)
        self.oid = oid

        # assure correct data points
        if not (t_mjd.shape == self.y.shape == self.dy.shape):
            raise ValueError("time, mag and magerr must be same shape.")
        if t_mjd.ndim != 1 or len(t_mjd) < 5:
            raise ValueError("Need at least five 1D data points.")
        if not (np.all(np.isfinite(t_mjd)) and np.all(np.isfinite(self.y)) and np.all(np.isfinite(self.dy))):
            raise ValueError("All lightcurve values must be finite.")

        self.t = (t_mjd - min(t_mjd)) / 365.25
        self.tau = np.abs(self.t[:, None] - self.t[None, :])
        self.diag = np.diag_indices(len(self.t))
        self.n = len(self.t)

        # prior on mean magnitude m 
        self.prior_m = (float(np.median(self.y) - 1), float(np.median(self.y) + 1))

    def prior_transform(self, u: np.ndarray, model: str):
        names = theta_names(model)
        ranges = {**PRIOR_RANGES, "m": self.prior_m}
        return np.array([ranges[k][0] + (ranges[k][1] - ranges[k][0]) * v for k, v in zip(names, u)], dtype=float)

    def loglikelihood(self, theta: np.ndarray, model: str):
        p = dict(zip(theta_names(model), theta))
        tau0 = np.exp(p["log_tau0"])
        sigmahat2 = np.exp(p["log_sigmahat2"])
        gamma = 1.0 if model.startswith("drw") else p["gamma"]

        # red noise covariance matrix
        C = 0.5 * sigmahat2 * tau0 * np.exp(-(self.tau/tau0)**gamma)

        # QPO multiplied with covariance matrix
        if "qpo" in model:
            C *= np.cos(2 * np.pi * self.tau / p["Tq"])

        # white noise of measurement
        C[self.diag] += (p["nu"] * self.dy)**2

        # vectorize mean
        mean = np.full(self.n, p["m"])
        # add sine signal
        if "sine" in model:
            mean += p["A"] * np.sin(2 * np.pi * self.t / p["T0"] + p["phi"])

        residual = self.y - mean

        # calculate the Cholesky decomposition to evaluate the inverse and ln|C|
        chol = cho_factor(C, lower=True, check_finite=False)
        alpha = cho_solve(chol, residual, check_finite=False)

        logdet = 2 * np.log(np.diag(chol[0])).sum()
        return float(-0.5 * (residual @ alpha + logdet + self.n * np.log(2 * np.pi)))

def load_lc(path: Path, time_col: str, mag_col: str, err_col: str):
    df = pd.read_csv(path)

    oid = (df["oid"][0])

    df = df[[time_col, mag_col, err_col]].apply(pd.to_numeric)

    good = np.isfinite(df.to_numpy()).all(axis=1)
    df = df.loc[good].sort_values(time_col)

    print(f"Loaded {len(df)} valid observatrions from {path}.")
    return AGNLC(oid, df[time_col].to_numpy(), df[mag_col].to_numpy(), df[err_col].to_numpy())

def fit_model(lc: AGNLC, model: str, nlive: int, maxbatch: int, seed: int, outdir: Path | None = None):
    names = theta_names(model)

    def transform(u):
        return lc.prior_transform(u, model)
    def logl(theta):
        return lc.loglikelihood(theta, model)

    sampler = dynesty.DynamicNestedSampler(logl, transform, ndim=len(names), nlive=nlive, bound="multi", sample="rwalk", rstate=np.random.default_rng(seed))

    sampler.run_nested(nlive_init=nlive, nlive_batch=max(50, nlive // 2), dlogz_init=0.2, maxbatch=maxbatch, print_progress=True)

    result = sampler.results

    weights = np.exp(result.logwt - result.logz[-1])
    weights /= weights.sum()

    posterior = resample_equal(result.samples, weights, rstate=np.random.default_rng(seed + 1))
    if outdir is not None:
        outdir = Path(outdir / f"{lc.oid}")
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(posterior, columns=names).to_csv(outdir / f"{model}_posterior.csv", index=False)

    summaries = {}
    posterior_medians = {}
    for j, name in enumerate(names):
        q16, q50, q84 = [float(v) for v in np.percentile(posterior[:, j], [16, 50, 84])]
        summaries[name] = [q16, q50, q84]
        posterior_medians[name] = q50


    # Best likelihood value among nested-sampling points
    best_index = int(np.argmax(result.logl))
    maximum_likelihood_sample = {
        "parameters": {
            name: float(result.samples[best_index, j])
            for j, name in enumerate(names)
        },
        "lnL": float(result.logl[best_index])
    }

    model_result = {
        "model": model,
        "lnZ": float(result.logz[-1]),
        "lnZ_err": float(result.logzerr[-1]),
        "parameters": names,
        "posterior_medians": posterior_medians,
        "quantiles_16_50_84": summaries,
        "maximum_likelihood_sample": maximum_likelihood_sample,
        "n_likelihood_evaluations": int(result.ncall.sum()),
        "n_posterior_samples": int(len(posterior)),
        "sampler_settings": {
            "nlive": int(nlive), "maxbatch": int(maxbatch), "seed": int(seed)
        },
        "prior_ranges": {
            name: [float(v) for v in (lc.prior_m if name == "m" else PRIOR_RANGES[name])]
            for name in names
        },
    }

    if outdir is not None:
        (outdir / f"{model}_results.json").write_text(json.dumps(model_result, indent=2), encoding="utf-8")

        np.savez_compressed(
            outdir / f"{model}_nested_samples.npz",
            samples=result.samples,
            logl = result.logl,
            logwt = result.logwt,
            logz = result.logz,
            logzerr = result.logzerr,
            weights = weights,
            ncall = result.ncall,
            parameter_names = np.array(names)
        )
    return model_result


def plot_fitted_model(lc: AGNLC, result, n_grid=500):
    model = result["model"]
    p = result["maximum_likelihood_sample"]["parameters"]

    t_plot = np.linspace(lc.t.min(), lc.t.max(), n_grid)

    def kernel(t1, t2):
        dt = np.abs(t1[:, None] - t2[None, :])

        tau0 = np.exp(p["log_tau0"])
        sigmahat2 = np.exp(p["log_sigmahat2"])
        gamma = 1.0 if model.startswith("drw") else p["gamma"]

        K = 0.5 * sigmahat2 * tau0 * np.exp(-(dt/tau0)**gamma)

        if "qpo" in model:
            K *= np.cos(2 * np.pi * dt / p["Tq"])

        return K

    def mean_function(t):
        mean = np.full(len(t), p["m"])

        if "sine" in model:
            mean += p["A"] * np.sin(2 * np.pi * t / p["T0"] + p["phi"])

        return mean


    C = kernel(lc.t, lc.t)
    C[lc.diag] += (p["nu"] * lc.dy)**2

    K_star = kernel(t_plot, lc.t)

    residual = lc.y - mean_function(lc.t)
    chol = cho_factor(C, lower=True)
    alpha = cho_solve(chol, residual)

    y_fit = mean_function(t_plot) + K_star @ alpha

    V = solve_triangular(chol[0], K_star.T, lower=True)

    prior_var = 0.5 * np.exp(p["log_sigmahat2"]) * np.exp(p["log_tau0"])

    y_std = np.sqrt(np.maximum(prior_var - np.sum(V**2, axis=0), 0))

    fig, ax = plt.subplots(figsize=(12,5))

    ax.errorbar(lc.t, lc.y, yerr=lc.dy, fmt=".", color="gray", alpha=0.5, label="Observed data")

    ax.plot(t_plot, y_fit, color="C0", lw=2, label=f"GP fit: {model}")

    ax.fill_between(t_plot, y_fit - y_std, y_fit + y_std, color="C0", alpha=0.2, label="Conditional 68% interval")

    ax.invert_yaxis()
    ax.set_xlabel("Years since first observation")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"AGN {lc.oid} - {model}")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return fig, ax


LATEX_LABELS = {
    "m": r"$m$",
    "nu": r"$\nu$",
    "log_sigmahat2": r"$\log \hat{\sigma}^2$",
    "log_tau0": r"$\log \tau_0$",
    "gamma": r"$\gamma$",
    "A": r"$A$",
    "phi": r"$\phi$",
    "T0": r"$T_0$",
    "Tq": r"$T_{\mathrm{q}}$",
}



def plot_corner(lc: AGNLC, model: str):
    posterior = pd.read_csv(f"../data/results/{lc.oid}/{model}_posterior.csv")
    params = [*posterior]
    labels = [
        LATEX_LABELS.get(param, param)
        for param in params
    ]

    fig = corner.corner(
        posterior[params].to_numpy(),
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".2f"
    )

    plt.show()

