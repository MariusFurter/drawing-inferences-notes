"""Generate a 1D MCMC illustration: target density + random walk trace."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ── Target density: mixture of two Gaussians ──
def target_pdf(x):
    return 0.6 * norm.pdf(x, loc=2, scale=0.8) + 0.4 * norm.pdf(x, loc=5, scale=1.2)


def target_log_pdf(x):
    return np.log(target_pdf(x))


# ── Metropolis-Hastings sampler ──
np.random.seed(42)
n_steps = 40
proposal_sd = 0.8
x = 0.5  # start

chain = [x]
accepted = [True]
proposals = [x]

for _ in range(n_steps - 1):
    x_prop = x + np.random.normal(0, proposal_sd)
    log_alpha = target_log_pdf(x_prop) - target_log_pdf(x)
    accept = np.log(np.random.uniform()) < log_alpha
    proposals.append(x_prop)
    accepted.append(accept)
    if accept:
        x = x_prop
    chain.append(x)

chain = np.array(chain)
proposals = np.array(proposals)
accepted = np.array(accepted)

# ── Plot ──
fig, ax = plt.subplots(figsize=(8, 3.5))

# Density curve
xs = np.linspace(-2, 9, 400)
ys = target_pdf(xs)
ax.fill_between(xs, ys, alpha=0.15, color="#5891d1")
ax.plot(xs, ys, color="#5891d1", lw=2.5, label="Target density")

# Place samples on the density
y_chain = target_pdf(chain)

# Draw the walk as thin lines between successive points
for i in range(1, len(chain)):
    color = "#87d6c5" if accepted[i] else "#f66164"
    alpha = 0.6
    # Line from previous position to new position (on the density)
    ax.plot(
        [chain[i - 1], chain[i]],
        [target_pdf(chain[i - 1]), target_pdf(chain[i])],
        color=color,
        lw=1.0,
        alpha=alpha,
        zorder=2,
    )
    # If rejected, show the proposal as a small x
    if not accepted[i]:
        ax.scatter(
            proposals[i],
            target_pdf(proposals[i]),
            marker="x",
            color="#f66164",
            s=30,
            alpha=0.5,
            zorder=3,
        )

# Draw the sample points
ax.scatter(
    chain, y_chain, color="#2c3e50", s=25, zorder=4, edgecolors="white", linewidths=0.5
)

# Start and end markers
ax.scatter(
    chain[0],
    y_chain[0],
    color="#f0a500",
    s=80,
    zorder=5,
    edgecolors="white",
    linewidths=1.5,
    label="Start",
)
ax.scatter(
    chain[-1],
    y_chain[-1],
    color="#f66164",
    s=80,
    zorder=5,
    edgecolors="white",
    linewidths=1.5,
    marker="D",
    label="Current",
)

ax.set_xlabel("$\\theta$", fontsize=14)
ax.set_ylabel("$p(\\theta \\mid y)$", fontsize=14)
ax.set_yticks([])
ax.set_xlim(-2, 9)
ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("slides/images/mcmc-illustration.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved slides/images/mcmc-illustration.png")
