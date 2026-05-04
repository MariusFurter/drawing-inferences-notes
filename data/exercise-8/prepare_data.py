"""Preprocess the raw sleepstudy.csv for Exercise 8.

Reads `sleepstudy.csv` (columns: Reaction, Days, Subject) and writes
`sleepstudy_processed.csv` with:

- a contiguous subject index ``subject_idx`` in [0, J-1],
- shorter, lowercased column names ``reaction``, ``day``, ``subject``,
- an artificially **imbalanced** number of observations per subject so
  that partial pooling produces visible shrinkage. Subjects are assigned
  one of three "regimes": full (all 10 days), medium (5 days kept), and
  sparse (2 days kept). Within each subject the kept days are sampled
  without replacement using a fixed RNG seed so the output is reproducible.

Run with:

    python data/exercise-8/prepare_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SRC = HERE / "sleepstudy.csv"
DST = HERE / "sleepstudy_processed.csv"
DST_HELDOUT = HERE / "sleepstudy_heldout.csv"

# Subjects with full data, medium data, and sparse data. Hand-picked from the
# sorted list of subject ids so that the pattern is reproducible and spans the
# full range of slopes seen in the raw data.
N_DAYS_BY_REGIME = {"full": 10, "medium": 5, "sparse": 2}
REGIME_BY_SUBJECT = {
    308: "full",
    309: "sparse",
    310: "medium",
    330: "full",
    331: "medium",
    332: "full",
    333: "medium",
    334: "sparse",
    335: "full",
    337: "full",
    349: "medium",
    350: "sparse",
    351: "full",
    352: "medium",
    369: "sparse",
    370: "medium",
    371: "full",
    372: "sparse",
}

RNG_SEED = 0


def main() -> None:
    df = pd.read_csv(SRC)

    rng = np.random.default_rng(RNG_SEED)
    keep_rows = []
    held_rows = []
    for sid, group in df.groupby("Subject", sort=True):
        regime = REGIME_BY_SUBJECT[int(sid)]
        n_keep = N_DAYS_BY_REGIME[regime]
        # Keep day 0 if possible (baseline anchor) and sample the rest.
        days_available = group["Days"].to_numpy()
        if 0 in days_available and n_keep >= 1:
            others = rng.choice(
                days_available[days_available != 0],
                size=n_keep - 1,
                replace=False,
            )
            kept = np.concatenate([[0], others])
        else:
            kept = rng.choice(days_available, size=n_keep, replace=False)
        kept.sort()
        keep_rows.append(group[group["Days"].isin(kept)])
        held_rows.append(group[~group["Days"].isin(kept)])

    out = pd.concat(keep_rows, ignore_index=True)
    held = pd.concat(held_rows, ignore_index=True)

    # Build a contiguous subject index over the subjects that survive (all 18
    # do, by construction). Rename to lowercase for ergonomic loading code.
    subject_ids = sorted(out["Subject"].unique().tolist())
    subject_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
    out = out.rename(
        columns={"Reaction": "reaction", "Days": "day", "Subject": "subject"}
    )
    out["subject_idx"] = out["subject"].map(subject_to_idx).astype(int)
    out = out[["subject", "subject_idx", "day", "reaction"]]
    out = out.sort_values(["subject_idx", "day"]).reset_index(drop=True)

    out.to_csv(DST, index=False)

    held = held.rename(
        columns={"Reaction": "reaction", "Days": "day", "Subject": "subject"}
    )
    held["subject_idx"] = held["subject"].map(subject_to_idx).astype(int)
    held = held[["subject", "subject_idx", "day", "reaction"]]
    held = held.sort_values(["subject_idx", "day"]).reset_index(drop=True)
    held.to_csv(DST_HELDOUT, index=False)

    counts = out.groupby("subject").size().sort_index()
    print(
        f"Wrote {DST} with {len(out)} rows over {out['subject_idx'].nunique()} subjects."
    )
    print(
        f"Wrote {DST_HELDOUT} with {len(held)} held-out rows over "
        f"{held['subject_idx'].nunique()} subjects."
    )
    print("Observations per subject (training):")
    print(counts.to_string())


if __name__ == "__main__":
    main()
