"""Generate a synthetic restaurant-ratings dataset for the Dirichlet
regression example in `chapters/mcmc.qmd`.

For each of N restaurants we observe:

- Three covariates measured on a [0, 1] scale (e.g. from a separate audit):
    * food     — food quality
    * service  — service quality
    * location — location / ambience quality
- A composition y = (y_poor, y_avg, y_good, y_excellent) giving the share of
  customer reviews falling into each of K = 4 rating buckets.

The data are generated from the Dirichlet regression model

    mu(x)  = softmax(x @ beta)              (mean composition, last column = 0)
    phi(x) = exp(x @ gamma)                 (total concentration)
    y | x  ~ Dirichlet(mu(x) * phi(x))

with x = (1, food, service, location). The "true" parameters are chosen so
that better restaurants have ratings that are both shifted toward
"excellent" and more concentrated (higher consensus among reviewers).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist
import pandas as pd

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
N = 80  # number of restaurants
K = 4  # rating buckets: poor, average, good, excellent
RATING_LABELS = ["poor", "average", "good", "excellent"]
SEED = 0

# True regression coefficients.
# Design matrix columns are (intercept, food, service, location).
# beta has shape (d, K) with the last column held at zero for identifiability.
# Each non-baseline column says: "as this covariate increases, push mass
# toward bucket k relative to 'excellent'." Negative entries in the
# 'poor'/'average' columns mean better covariates suppress those buckets.
TRUE_BETA = jnp.array(
    [
        # poor   average  good   excellent (= baseline, fixed at 0)
        [0.5, 1.2, 1.0, 0.0],  # intercept
        [-3.0, -2.0, -0.5, 0.0],  # food
        [-2.0, -1.5, -0.3, 0.0],  # service
        [-1.0, -0.8, -0.2, 0.0],  # location
    ]
)

# gamma controls the log-concentration. Larger phi => tighter Dirichlet
# (more reviewer consensus). Better restaurants have higher consensus.
TRUE_GAMMA = jnp.array([2.0, 1.5, 1.0, 0.5])  # (intercept, food, service, location)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def dirichlet_regression_params(X: jnp.ndarray, beta: jnp.ndarray, gamma: jnp.ndarray):
    """Compute (mu, phi, alpha) for each row of X."""
    logits = X @ beta  # (N, K)
    mu = jax.nn.softmax(logits, axis=-1)  # (N, K)
    phi = jnp.exp(X @ gamma)  # (N,)
    alpha = mu * phi[:, None]  # (N, K)
    return mu, phi, alpha


def main() -> None:
    key = random.PRNGKey(SEED)
    key_x, key_y = random.split(key)

    # Covariates: independent uniform on [0, 1].
    covariates = random.uniform(key_x, shape=(N, 3))
    food, service, location = covariates.T

    # Design matrix with intercept.
    X = jnp.concatenate([jnp.ones((N, 1)), covariates], axis=1)

    # Compositional outcomes.
    _, _, alpha = dirichlet_regression_params(X, TRUE_BETA, TRUE_GAMMA)
    y = dist.Dirichlet(alpha).sample(key_y)  # (N, K)

    # Assemble dataframe.
    df = pd.DataFrame(
        {
            "food": food,
            "service": service,
            "location": location,
            **{f"share_{label}": y[:, k] for k, label in enumerate(RATING_LABELS)},
        }
    )

    out_path = HERE / "restaurant-ratings.csv"
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Wrote {len(df)} rows to {out_path}")

    # Save the ground-truth parameters alongside for reference.
    truth = {
        "beta": TRUE_BETA.tolist(),
        "gamma": TRUE_GAMMA.tolist(),
        "rating_labels": RATING_LABELS,
        "covariate_names": ["intercept", "food", "service", "location"],
        "n": N,
        "seed": SEED,
    }
    import json

    with open(HERE / "true-params.json", "w") as f:
        json.dump(truth, f, indent=2)
    print(f"Wrote ground-truth parameters to {HERE / 'true-params.json'}")


if __name__ == "__main__":
    main()
