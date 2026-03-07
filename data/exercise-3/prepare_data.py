import pandas as pd

for name in ["train", "test"]:
    df = pd.read_csv(f"stroke_{name}.csv")

    # Bin age: [0,35)=0, [35,65)=1, [65,inf)=2
    df["age_cat"] = pd.cut(
        df["age"], bins=[0, 35, 65, float("inf")], labels=[0, 1, 2], right=False
    ).astype(int)

    # Smoking: never smoked=0, formerly smoked/smokes=1, Unknown=2
    smoking_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 1, "Unknown": 2}
    df["smoking_cat"] = df["smoking_status"].map(smoking_map).astype(int)

    # Select and rename
    out = df[
        ["age_cat", "smoking_cat", "hypertension", "heart_disease", "stroke"]
    ].copy()
    out.columns = ["age", "smoking", "hypertension", "heart_disease", "stroke"]

    path = f"stroke_{name}_prepared.csv"
    out.to_csv(path, index=False)
    print(f"{path}: {len(out)} rows")
    print(out.head())
    print()
    for col in out.columns:
        print(f"  {col}: {dict(out[col].value_counts().sort_index())}")
    print()
