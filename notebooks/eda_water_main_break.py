from pathlib import Path
import json
import math
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "pipe_break_snapshots.csv"
OUT = ROOT / "reports" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = pd.Timestamp("2015-01-01")


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close()


def safe_num(frame, col):
    return pd.to_numeric(frame[col], errors="coerce") if col in frame else pd.Series(dtype=float)


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Snapshots introuvables : {DATA}")
    df = pd.read_csv(DATA)
    if "snapshot_date" in df:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    for col in ["diameter_mm", "age_years", "prior_break_count", "install_year"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    missing = pd.DataFrame({"missing_count": df.isna().sum(), "missing_rate": df.isna().mean()}).sort_values("missing_rate", ascending=False)
    missing.to_csv(OUT / "missing_values.csv")
    missing["missing_rate"].plot.bar(figsize=(11, 4), title="Taux de valeurs manquantes")
    plt.ylabel("Proportion")
    savefig("missing_values.png")

    duplicate = {"rows": len(df), "exact_duplicates": int(df.duplicated().sum())}
    pd.DataFrame([duplicate]).to_csv(OUT / "duplicate_summary.csv", index=False)

    numeric = [c for c in ["diameter_mm", "age_years", "prior_break_count"] if c in df]
    if numeric:
        fig, axes = plt.subplots(1, len(numeric), figsize=(5 * len(numeric), 4))
        for ax, col in zip([axes] if len(numeric) == 1 else axes, numeric):
            df[col].dropna().hist(ax=ax, bins=30)
            ax.set_title(col)
        savefig("numeric_distributions.png")

    if "material" in df:
        df["material"].fillna("MISSING").value_counts().plot.bar(figsize=(10, 4), title="Distribution du matériau")
        plt.ylabel("Nombre de snapshots")
        savefig("material_distribution.png")

    target = "break_within_horizon"
    if target in df:
        rates = df.groupby("horizon_years")[target].agg(["count", "sum", "mean"]).rename(columns={"sum": "breaks", "mean": "break_rate"}) if "horizon_years" in df else pd.DataFrame({"count": [len(df)], "breaks": [df[target].sum()], "break_rate": [df[target].mean()]})
        rates.to_csv(OUT / "break_rate_by_horizon.csv")
        rates["break_rate"].plot.bar(figsize=(7, 4), title="Taux de rupture par horizon")
        plt.ylabel("Taux de rupture")
        savefig("break_rate_by_horizon.png")

    if "years_until_break" in df:
        y = safe_num(df, "years_until_break").dropna()
        y.describe().to_csv(OUT / "regression_target_summary.csv")
        if not y.empty:
            y.hist(bins=30, figsize=(7, 4))
            plt.title("Distribution de years_until_break (non censurée)")
            plt.xlabel("Années avant rupture")
            savefig("years_until_break_distribution.png")

    outliers = []
    for col in numeric:
        x = df[col].dropna(); q1, q3 = x.quantile(.25), x.quantile(.75); iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers.append({"column": col, "q1": q1, "q3": q3, "lower_iqr": low, "upper_iqr": high, "outlier_count": int(((x < low) | (x > high)).sum())})
    pd.DataFrame(outliers).to_csv(OUT / "outlier_iqr_summary.csv", index=False)

    if "snapshot_date" in df:
        df["partition"] = df["snapshot_date"].ge(SPLIT).map({True: "test", False: "train"})
        cols = [c for c in numeric + ([target] if target in df else []) if c in df]
        comparison = df.groupby("partition")[cols].agg(["count", "mean", "median"])
        comparison.to_csv(OUT / "temporal_train_test_comparison.csv")
        df["partition"].value_counts().plot.bar(title="Taille des partitions temporelles")
        savefig("train_test_comparison.png")

    summary = ["# EDA — Water Main Break Prediction", f"- Snapshots : {len(df)}", f"- Doublons exacts : {duplicate['exact_duplicates']}", f"- Split temporel : train < {SPLIT.date()}, test >= {SPLIT.date()}", "", "Les tableaux CSV et graphiques PNG sont dans reports/eda/."]
    (OUT / "EDA_SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"EDA terminé : {OUT}")


if __name__ == "__main__":
    main()
