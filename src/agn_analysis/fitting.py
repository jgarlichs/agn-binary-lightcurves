import numpy as np
from pathlib import Path
import pandas as pd
import json
import dynesty
from dynesty.utils import resample_equal
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from .model import AGNLC, PRIOR_RANGES, theta_names


def fit_model(lc: AGNLC, model: str, nlive: int, maxbatch: int, seed: int, print_progress: bool = True):
    names = theta_names(model)

    def transform(u):
        return lc.prior_transform(u, model)
    def logl(theta):
        return lc.loglikelihood(theta, model)

    sampler = dynesty.DynamicNestedSampler(logl, transform, ndim=len(names), nlive=nlive, bound="multi", sample="rwalk", rstate=np.random.default_rng(seed))

    sampler.run_nested(nlive_init=nlive, nlive_batch=max(50, nlive // 2), dlogz_init=0.2, maxbatch=maxbatch, print_progress=print_progress)

    result = sampler.results

    weights = np.exp(result.logwt - result.logz[-1])
    weights /= weights.sum()

    posterior = resample_equal(result.samples, weights, rstate=np.random.default_rng(seed + 1))

    outdir = get_result_path(lc, seed)
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
        "filter_settings": lc.filter_settings,
    }

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

def read_result_json(lc: AGNLC, model: str, seed: int):
    RESULTS_PATH = get_result_path(lc, seed) / f"{model}_results.json"
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return result

def get_result_path(lc: AGNLC, seed: int) -> Path:
    return Path(PROJECT_ROOT.parent / "data" / "results" / str(lc.oid) / f"{lc.filter_settings["filtercode"]}" / f"seed{seed}")