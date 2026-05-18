"""Prepare the dataset for Exercise 9 (Stroke prediction 2.0).

Reads the train/test splits used in Exercise 3
    `data/exercise-3/stroke_train.csv`
    `data/exercise-3/stroke_test.csv`
and writes the prepared CSVs for Exercise 9 with the following columns:

  * ``age``           -- binned into {0: young (<35), 1: middle [35, 65),
                                       2: senior (>=65)} (same bins as Ex. 3).
  * ``hyp``           -- 0/1, copied from the raw ``hypertension`` column.
  * ``log_glucose``   -- natural log of ``avg_glucose_level`` (mg/dL).
  * ``heart_disease`` -- 0/1, copied from the raw column.

The continuous variable is log-transformed because the raw ``avg_glucose_level``
distribution is right-skewed; the log-transformed values are roughly Gaussian,
which makes a Normal likelihood reasonable for the regression box in Task 3.

Run with:

    python data/exercise-9/prepare_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SRC_DIR = HERE.parent / "exercise-3"


def prepare(src: Path, dst: Path) -> None:
    df = pd.read_csv(src)

    age_bin = pd.cut(
        df["age"],
        bins=[0, 35, 65, float("inf")],
        labels=[0, 1, 2],
        right=False,
    ).astype(int)

    prepared = pd.DataFrame(
        {
            "age": age_bin,
            "hyp": df["hypertension"].astype(int),
            "log_glucose": np.log(df["avg_glucose_level"]).astype(float),
            "heart_disease": df["heart_disease"].astype(int),
        }
    )

    prepared.to_csv(dst, index=False)
    print(f"wrote {dst}  ({len(prepared)} rows)")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    prepare(SRC_DIR / "stroke_train.csv", HERE / "stroke_train_prepared.csv")
    prepare(SRC_DIR / "stroke_test.csv", HERE / "stroke_test_prepared.csv")


if __name__ == "__main__":
    main()
