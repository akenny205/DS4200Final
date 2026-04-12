"""
Feature Analysis
----------------
Finds which audio features best predict popularity using:
  1. Pearson correlation + p-values
  2. Linear regression coefficients (statsmodels)
  3. Random Forest feature importance

Outputs a combined ranked summary table and saves it to 'feature_importance.csv'.

Usage:
    python feature_analysis.py
    — or —
    from feature_analysis import run_analysis
    results_df = run_analysis(df)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Features to test against popularity
FEATURES = [
    "danceability", "energy", "acousticness", "valence",
    "tempo", "speechiness", "instrumentalness", "loudness",
    "liveness", "duration_ms"
]

TARGET = "popularity"


# ── 1. Pearson Correlation ───────────────────────────────────────────────────

def pearson_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in FEATURES:
        if feat not in df.columns:
            continue
        r, p = stats.pearsonr(df[feat], df[TARGET])
        rows.append({
            "feature": feat,
            "pearson_r": round(r, 4),
            "pearson_p": round(p, 4),
            "abs_r": abs(r)
        })
    return (
        pd.DataFrame(rows)
        .sort_values("abs_r", ascending=False)
        .drop(columns="abs_r")
        .reset_index(drop=True)
    )


# ── 2. Linear Regression (OLS) ───────────────────────────────────────────────

def linear_regression(df: pd.DataFrame) -> pd.DataFrame:
    avail = [f for f in FEATURES if f in df.columns]
    X = df[avail].copy()

    # Standardize so coefficients are comparable
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=avail
    )
    X_scaled = sm.add_constant(X_scaled)
    y = df[TARGET].reset_index(drop=True)

    model = sm.OLS(y, X_scaled).fit()

    coef_df = pd.DataFrame({
        "feature": model.params.index,
        "ols_coef": model.params.values.round(4),
        "ols_p":    model.pvalues.values.round(4)
    })
    # drop the intercept row
    coef_df = coef_df[coef_df["feature"] != "const"].reset_index(drop=True)

    print(f"\nLinear Regression R²: {model.rsquared:.4f}")
    return coef_df


# ── 3. Random Forest Feature Importance ─────────────────────────────────────

def random_forest_importance(df: pd.DataFrame) -> pd.DataFrame:
    avail = [f for f in FEATURES if f in df.columns]
    X = df[avail]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    test_r2 = r2_score(y_test, rf.predict(X_test))
    print(f"Random Forest Test R²: {test_r2:.4f}")

    imp_df = pd.DataFrame({
        "feature":    avail,
        "rf_importance": rf.feature_importances_.round(4)
    }).sort_values("rf_importance", ascending=False).reset_index(drop=True)

    return imp_df


# ── Combined Summary ─────────────────────────────────────────────────────────

def run_analysis(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*55)
    print("  FEATURE ANALYSIS vs. POPULARITY")
    print("="*55)

    # drop rows with missing values in relevant columns
    cols = FEATURES + [TARGET]
    avail_cols = [c for c in cols if c in df.columns]
    df_clean = df[avail_cols].dropna()
    print(f"Rows used: {len(df_clean):,}")

    corr   = pearson_correlations(df_clean)
    ols    = linear_regression(df_clean)
    rf_imp = random_forest_importance(df_clean)

    # Merge all three on feature
    results = (
        corr
        .merge(ols,    on="feature", how="outer")
        .merge(rf_imp, on="feature", how="outer")
    )

    # Rank by RF importance (most predictive first)
    results = results.sort_values("rf_importance", ascending=False).reset_index(drop=True)
    results.index += 1  # 1-based rank

    print("\n── Final Ranked Summary (by RF importance) ──")
    print(results.to_string())
    print()

    results.to_csv("feature_importance.csv", index_label="rank")
    print("Saved → feature_importance.csv\n")

    return results


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("dataset.csv").dropna()
    results = run_analysis(df)