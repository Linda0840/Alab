import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_fraud_rate_numerical(df: pd.DataFrame, feature: str, target: str):
    """
    Visualize fraud rate difference between missing and non-missing values of a feature,
    and fraud rate across 5 monotonic (equal-frequency) bins with rounded labels.
    Y-axis auto-scales to the max fraud rate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # === 1️⃣ Fraud rate: NaN vs Non-NaN ===
    df_nan = df[df[feature].isna()]
    df_nonan = df[df[feature].notna()]
    
    fraud_nan = df_nan[target].mean() if not df_nan.empty else np.nan
    fraud_nonan = df_nonan[target].mean() if not df_nonan.empty else np.nan

    # === 2️⃣ Fraud rate across 5 bins ===
    df_binned = df.copy()
    df_binned['bin'] = pd.qcut(df_binned[feature], q=5, duplicates='drop')
    fraud_by_bin = df_binned.groupby('bin')[target].mean()

    # collect all rates to set y-limits nicely
    all_rates = [fraud_nan, fraud_nonan] + list(fraud_by_bin.values)
    max_rate = np.nanmax(all_rates)  # e.g. 0.0501
    # give it some headroom; floor at 5%
    y_max = max(0.05, max_rate * 1.3)
    y_max = min(1.0, y_max)

    # --- left plot ---
    axes[0].bar(['NaN', 'Non-NaN'], [fraud_nan, fraud_nonan], color=['orange', 'skyblue'])
    axes[0].set_title(f'Fraud Rate: NaN vs Non-NaN\n({feature})')
    axes[0].set_ylabel('Fraud Rate')
    axes[0].set_ylim(0, y_max)
    for i, val in enumerate([fraud_nan, fraud_nonan]):
        if not np.isnan(val):
            axes[0].text(i, val + y_max*0.02, f'{val:.2%}', ha='center', fontsize=10)

    # round bin labels to 3 digits for display
    bin_labels = [
        f"[{round(b.left, 3)}, {round(b.right, 3)})"
        for b in fraud_by_bin.index
    ]

    # --- right plot ---
    axes[1].plot(
        range(len(fraud_by_bin)),
        fraud_by_bin.values,
        marker='o', linestyle='-', color='purple'
    )
    axes[1].set_xticks(range(len(fraud_by_bin)))
    axes[1].set_xticklabels(bin_labels, rotation=45, ha='right')
    axes[1].set_title(f'Fraud Rate by {feature} (5 Quantile Bins)')
    axes[1].set_ylabel('Fraud Rate')
    axes[1].set_ylim(0, y_max)
    for i, val in enumerate(fraud_by_bin.values):
        axes[1].text(i, val + y_max*0.02, f'{val:.2%}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()


###########################################################################################################################################################################################################################################################################3


def visualize_fraud_rate_categorical(df: pd.DataFrame, feature: str, target: str):
    """
    Plot fraud rate for each category of a categorical feature (including NaN as its own bar).

    Parameters
    ----------
    df : pd.DataFrame
    feature : str   # categorical column, can have NaN
    target : str    # binary target, 1 = fraud
    """
    tmp = df[[feature, target]].copy()

    # make NaN explicit so we can groupby it
    tmp[feature] = tmp[feature].astype("object")
    tmp[feature] = tmp[feature].where(tmp[feature].notna(), "NaN")

    # fraud rate per category
    grp = tmp.groupby(feature)[target].agg(["mean", "count"]).reset_index()
    grp.rename(columns={"mean": "fraud_rate", "count": "n"}, inplace=True)

    # sort (optional): highest fraud rate first
    grp = grp.sort_values("fraud_rate", ascending=False)

    # y-limit with headroom
    max_rate = grp["fraud_rate"].max()
    y_max = max(0.05, max_rate * 1.25)
    y_max = min(1.0, y_max)

    plt.figure(figsize=(max(6, 0.7 * len(grp)), 4))
    plt.bar(grp[feature], grp["fraud_rate"])
    plt.ylim(0, y_max)
    plt.ylabel("Fraud Rate")
    plt.title(f"Fraud Rate by {feature}")

    # put % on top of bars
    for i, (cat, rate, n) in enumerate(zip(grp[feature], grp["fraud_rate"], grp["n"])):
        plt.text(i, rate + y_max*0.015, f"{rate:.2%}\n(n={n})", ha="center", va="bottom", fontsize=8)

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()
