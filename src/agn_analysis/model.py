# AGNLC class and functions
import numpy as np
from scipy.linalg import cho_factor, cho_solve

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

def red_noise_kernel(tau: np.ndarray, p: dict, model: str):
    """ Red-noise (x QPO) covariance for time lags tau"""
    tau0 = np.exp(p["log_tau0"])
    sigmahat2 = np.exp(p["log_sigmahat2"])
    gamma = 1.0 if model.startswith("drw") else p["gamma"]

    K = 0.5 * sigmahat2 * tau0 * np.exp(-(tau / tau0)**gamma)

    if "qpo" in model:
        K *= np.cos(2 * np.pi * tau / p["Tq"])
    return K

def mean_function(t: np.ndarray, p: dict, model: str):
    mean = np.full(len(t), p["m"], dtype=float)
    if "sine" in model:
        mean += p["A"] * np.sin(2 * np.pi * t / p["T0"] + p["phi"])

    return mean

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
    def __init__(self, oid: int, t_mjd: np.ndarray, mag: np.ndarray, magerr: np.ndarray, filter_settings: dict | None = None):
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

        # store filter settings
        self.filter_settings = filter_settings

    def prior_transform(self, u: np.ndarray, model: str):
        names = theta_names(model)
        ranges = {**PRIOR_RANGES, "m": self.prior_m}
        return np.array([ranges[k][0] + (ranges[k][1] - ranges[k][0]) * v for k, v in zip(names, u)], dtype=float)

    def covariance(self, p: dict, model: str):
        C = red_noise_kernel(self.tau, p, model)
        # white noise of measurement
        C[self.diag] += (p["nu"] * self.dy)**2
        return C
    def loglikelihood(self, theta: np.ndarray, model: str):

        p = dict(zip(theta_names(model), theta))

        C = self.covariance(p, model)
        residual = self.y - mean_function(self.t, p, model)
    
        # calculate the Cholesky decomposition to evaluate the inverse and ln|C|
        chol = cho_factor(C, lower=True, check_finite=False)
        alpha = cho_solve(chol, residual, check_finite=False)
    
        logdet = 2 * np.log(np.diag(chol[0])).sum()
        return float(-0.5 * (residual @ alpha + logdet + self.n * np.log(2 * np.pi)))