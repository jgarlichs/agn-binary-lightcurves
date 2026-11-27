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
from .model import AGNLC, red_noise_kernel, mean_function
from .fitting import get_result_path

def plot_fitted_model(lc: AGNLC, result, n_grid=500, show=True):
    model = result["model"]
    p = result["maximum_likelihood_sample"]["parameters"]
    seed = result["sampler_settings"]["seed"]
    outdir = get_result_path(lc, seed)

    t_plot = np.linspace(lc.t.min(), lc.t.max(), n_grid)

    tau_plot = np.abs(t_plot[:, None] - lc.t[None, :])

    C = lc.covariance(p, model)

    K_star = red_noise_kernel(tau_plot, p, model)

    residual = lc.y - mean_function(lc.t, p, model)
    chol = cho_factor(C, lower=True)
    alpha = cho_solve(chol, residual)

    y_fit = mean_function(t_plot, p, model) + K_star @ alpha

    V = solve_triangular(chol[0], K_star.T, lower=True)

    prior_var = red_noise_kernel(0.0, p, model)

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
    plt.savefig(outdir / f"{model}_fitted_lc.png")
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

def plot_corner(lc: AGNLC, model: str, seed: int, show=True):
    result_folder = get_result_path(lc, seed)
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