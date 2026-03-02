"""Generate datasets for Exercise 3: Hidden Markov Models."""

import random
import csv
import math

random.seed(2026)

# ============================================================
# Part 1: CpG island dataset
# ============================================================
nucs = ["A", "C", "G", "T"]

# Emission probabilities
emit = {
    "non-island": [0.30, 0.20, 0.20, 0.30],  # A, C, G, T
    "island": [0.15, 0.35, 0.35, 0.15],  # enriched in C and G
}

# Transition probabilities [from][to]
# non-island -> island with p=0.01  (expected run ~100)
# island -> non-island with p=0.02  (expected run ~50)
trans_cpg = [[0.99, 0.01], [0.02, 0.98]]


def generate_cpg(length, init_state=0):
    states, obs = [], []
    state = init_state
    for _ in range(length):
        states.append(state)
        label = "island" if state == 1 else "non-island"
        # Emit
        r = random.random()
        cum = 0
        for j, p in enumerate(emit[label]):
            cum += p
            if r < cum:
                obs.append(nucs[j])
                break
        # Transition
        r = random.random()
        state = 0 if r < trans_cpg[state][0] else 1
    return states, obs


# Training: 2000 positions
train_states, train_obs = generate_cpg(2000)
# Test: 500 positions
test_states, test_obs = generate_cpg(500, init_state=0)

with open("cpg_train.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["position", "nucleotide", "state"])
    for i, (s, o) in enumerate(zip(train_states, train_obs)):
        w.writerow([i, o, "island" if s == 1 else "non-island"])

with open("cpg_test.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["position", "nucleotide"])
    for i, o in enumerate(test_obs):
        w.writerow([i, o])

with open("cpg_test_labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["position", "state"])
    for i, s in enumerate(test_states):
        w.writerow([i, "island" if s == 1 else "non-island"])

n_island_train = sum(train_states)
n_island_test = sum(test_states)
print(
    f"CpG training: {len(train_states)} pos, {n_island_train} island ({100*n_island_train/len(train_states):.1f}%)"
)
print(
    f"CpG test:     {len(test_states)} pos, {n_island_test} island ({100*n_island_test/len(test_states):.1f}%)"
)

# ============================================================
# Part 2: Stock market volatility regime dataset
# ============================================================
# Hidden states: 0 = low-vol, 1 = high-vol
# Observations: daily return discretized into 5 bins
#   0: large drop  (r < -1.5%)
#   1: small drop  (-1.5% <= r < -0.5%)
#   2: flat        (-0.5% <= r < 0.5%)
#   3: small gain  (0.5% <= r < 1.5%)
#   4: large gain  (r >= 1.5%)
return_labels = ["large_drop", "small_drop", "flat", "small_gain", "large_gain"]
bin_edges = [-float("inf"), -1.5, -0.5, 0.5, 1.5, float("inf")]

# Emission: return distribution depends on regime
# Low-vol: concentrated around flat, small moves
emit_low = [0.02, 0.13, 0.55, 0.23, 0.07]  # slightly positive skew
emit_high = [0.15, 0.20, 0.20, 0.20, 0.25]  # fat tails, wider

# Transition
# low-vol -> high-vol with p=0.02  (expected run ~50 days)
# high-vol -> low-vol with p=0.05  (expected run ~20 days)
trans_vol = [[0.98, 0.02], [0.05, 0.95]]


def generate_vol(length, init_state=0):
    states, obs = [], []
    state = init_state
    for _ in range(length):
        states.append(state)
        e = emit_low if state == 0 else emit_high
        r = random.random()
        cum = 0
        for j, p in enumerate(e):
            cum += p
            if r < cum:
                obs.append(j)
                break
        r = random.random()
        state = 0 if r < trans_vol[state][0] else 1
    return states, obs


# Training: 1000 days (~4 years)
vol_train_states, vol_train_obs = generate_vol(1000)
# Test: 500 days (~2 years)
vol_test_states, vol_test_obs = generate_vol(500, init_state=0)

with open("vol_train.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["day", "return_bin", "regime"])
    for i, (s, o) in enumerate(zip(vol_train_states, vol_train_obs)):
        w.writerow([i, return_labels[o], "high_vol" if s == 1 else "low_vol"])

with open("vol_test.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["day", "return_bin"])
    for i, o in enumerate(vol_test_obs):
        w.writerow([i, return_labels[o]])

with open("vol_test_labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["day", "regime"])
    for i, s in enumerate(vol_test_states):
        w.writerow([i, "high_vol" if s == 1 else "low_vol"])

n_hv_train = sum(vol_train_states)
n_hv_test = sum(vol_test_states)
print(
    f"Vol training: {len(vol_train_states)} days, {n_hv_train} high-vol ({100*n_hv_train/len(vol_train_states):.1f}%)"
)
print(
    f"Vol test:     {len(vol_test_states)} days, {n_hv_test} high-vol ({100*n_hv_test/len(vol_test_states):.1f}%)"
)

print("\nDone. Files written to data/")
