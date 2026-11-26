import numpy as np
from scipy.linalg import cho_factor, cho_solve
from pathlib import Path
import pandas as pd
import corner
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STYLE_PATH = PROJECT_ROOT.parent / "src" / "light.mplstyle"
plt.style.use(STYLE_PATH)
from .model import AGNLC

def plot_fitted_model(lc: AGNLC, result, n_grid=500, show=True):
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
    plt.savefig(PROJECT_ROOT.parent / "data" / "results" / str(lc.oid) / f"{model}_fitted_lc.png")
    if show:
        plt.show()
    else:
        plt.close(fig)

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

def plot_corner(lc: AGNLC, model: str, show=True):
    result_folder = PROJECT_ROOT.parent / "data" / "results" / str(lc.oid)
    posterior = pd.read_csv(result_folder / f"{model}_posterior.csv")
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
    plt.savefig(result_folder / f"{model}_corner.png")
    if show:
        plt.show()
    else:
        plt.close(fig)