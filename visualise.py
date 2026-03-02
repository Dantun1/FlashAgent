"""
Visualisation pipeline for APC (Agentic Plan Caching) benchmark results.

Compares two telemetry CSVs — a no-cache baseline and a cached run — across
150 queries on latency (ms) and token cost.

Usage:
    python visualise.py --baseline data/baseline.csv --cached data/cache_telemetry.csv
    python visualise.py --baseline data/baseline.csv --cached data/cache_telemetry.csv --out plots/
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


BASELINE_COLOR = "#e05c5c"   # red
CACHED_COLOR   = "#4a90d9"   # blue
FILL_ALPHA     = 0.12
FIG_DPI        = 150


def _load(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["total_tokens"] = df["vertex_input_tokens"] + df["vertex_output_tokens"]
    df = df.sort_values("query_index").reset_index(drop=True)
    df["label"] = label
    return df


def _savings_text(baseline_vals, cached_vals, unit: str) -> str:
    b_total = baseline_vals.sum()
    c_total = cached_vals.sum()
    if b_total == 0:
        return ""
    pct = (b_total - c_total) / b_total * 100
    return f"Cache saves {pct:.1f}% {unit} over {len(baseline_vals)} queries"


def plot_per_query_latency(df_base, df_cache, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(df_base["query_index"], df_base["latency_ms"],
            color=BASELINE_COLOR, linewidth=1, label="Baseline (no cache)", alpha=0.85)
    ax.plot(df_cache["query_index"], df_cache["latency_ms"],
            color=CACHED_COLOR, linewidth=1, label="APC (cached)", alpha=0.85)

    ax.fill_between(df_base["query_index"], df_base["latency_ms"], df_cache["latency_ms"],
                    where=(df_base["latency_ms"].values > df_cache["latency_ms"].values),
                    interpolate=True, color=CACHED_COLOR, alpha=FILL_ALPHA, label="Savings region")

    note = _savings_text(df_base["latency_ms"], df_cache["latency_ms"], "latency")
    if note:
        ax.text(0.99, 0.97, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="grey")

    ax.set_xlabel("Query index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-query latency: Baseline vs APC")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_latency_per_query.png"), dpi=FIG_DPI)
    plt.close(fig)
    print("Saved: 01_latency_per_query.png")


def plot_per_query_tokens(df_base, df_cache, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(df_base["query_index"], df_base["total_tokens"],
            color=BASELINE_COLOR, linewidth=1, label="Baseline (no cache)", alpha=0.85)
    ax.plot(df_cache["query_index"], df_cache["total_tokens"],
            color=CACHED_COLOR, linewidth=1, label="APC (cached)", alpha=0.85)

    ax.fill_between(df_base["query_index"], df_base["total_tokens"], df_cache["total_tokens"],
                    where=(df_base["total_tokens"].values > df_cache["total_tokens"].values),
                    interpolate=True, color=CACHED_COLOR, alpha=FILL_ALPHA, label="Savings region")

    note = _savings_text(df_base["total_tokens"], df_cache["total_tokens"], "tokens")
    if note:
        ax.text(0.99, 0.97, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="grey")

    ax.set_xlabel("Query index")
    ax.set_ylabel("Total tokens (input + output)")
    ax.set_title("Per-query token cost: Baseline vs APC")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_tokens_per_query.png"), dpi=FIG_DPI)
    plt.close(fig)
    print("Saved: 02_tokens_per_query.png")


def plot_cumulative_latency(df_base, df_cache, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))

    cum_base  = df_base["latency_ms"].cumsum()
    cum_cache = df_cache["latency_ms"].cumsum()

    ax.plot(df_base["query_index"], cum_base,
            color=BASELINE_COLOR, linewidth=1.5, label="Baseline (no cache)")
    ax.plot(df_cache["query_index"], cum_cache,
            color=CACHED_COLOR, linewidth=1.5, label="APC (cached)")
    ax.fill_between(df_base["query_index"], cum_base, cum_cache,
                    alpha=FILL_ALPHA * 2, color=CACHED_COLOR)

    final_saving_ms = cum_base.iloc[-1] - cum_cache.iloc[-1]
    ax.annotate(f"Total saving: {final_saving_ms:,.0f} ms",
                xy=(df_base["query_index"].iloc[-1], cum_cache.iloc[-1]),
                xytext=(-80, 20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="grey"), fontsize=8, color="grey")

    ax.set_xlabel("Query index")
    ax.set_ylabel("Cumulative latency (ms)")
    ax.set_title("Cumulative latency: Baseline vs APC")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_cumulative_latency.png"), dpi=FIG_DPI)
    plt.close(fig)
    print("Saved: 03_cumulative_latency.png")


def plot_cumulative_tokens(df_base, df_cache, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))

    cum_base  = df_base["total_tokens"].cumsum()
    cum_cache = df_cache["total_tokens"].cumsum()

    ax.plot(df_base["query_index"], cum_base,
            color=BASELINE_COLOR, linewidth=1.5, label="Baseline (no cache)")
    ax.plot(df_cache["query_index"], cum_cache,
            color=CACHED_COLOR, linewidth=1.5, label="APC (cached)")
    ax.fill_between(df_base["query_index"], cum_base, cum_cache,
                    alpha=FILL_ALPHA * 2, color=CACHED_COLOR)

    final_saving_tok = cum_base.iloc[-1] - cum_cache.iloc[-1]
    ax.annotate(f"Total saving: {final_saving_tok:,.0f} tokens",
                xy=(df_base["query_index"].iloc[-1], cum_cache.iloc[-1]),
                xytext=(-80, 20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="grey"), fontsize=8, color="grey")

    ax.set_xlabel("Query index")
    ax.set_ylabel("Cumulative tokens (input + output)")
    ax.set_title("Cumulative token cost: Baseline vs APC")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_cumulative_tokens.png"), dpi=FIG_DPI)
    plt.close(fig)
    print("Saved: 04_cumulative_tokens.png")


def plot_summary(df_base, df_cache, out_dir: str) -> None:
    metrics = {
        "Mean latency (ms)": (df_base["latency_ms"].mean(), df_cache["latency_ms"].mean()),
        "Median latency (ms)": (df_base["latency_ms"].median(), df_cache["latency_ms"].median()),
        "Total tokens": (df_base["total_tokens"].sum(), df_cache["total_tokens"].sum()),
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))

    for ax, (title, (b_val, c_val)) in zip(axes, metrics.items()):
        bars = ax.bar(["Baseline", "APC"], [b_val, c_val],
                      color=[BASELINE_COLOR, CACHED_COLOR], width=0.5)
        pct = (b_val - c_val) / b_val * 100 if b_val else 0
        ax.set_title(f"{title}\n({pct:.1f}% reduction)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        for bar, val in zip(bars, [b_val, c_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:,.0f}", ha="center", va="bottom", fontsize=8)

    # Cache hit rate from cached run
    if "cache_status" in df_cache.columns:
        hit_rate = (df_cache["cache_status"] == "HIT").mean() * 100
        fig.suptitle(f"APC Summary — Cache hit rate: {hit_rate:.1f}%", fontsize=11, y=1.02)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_summary.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 05_summary.png")


def main():
    parser = argparse.ArgumentParser(description="Visualise APC benchmark results.")
    parser.add_argument("--baseline", required=True, help="CSV from the no-cache baseline run.")
    parser.add_argument("--cached",   required=True, help="CSV from the APC cached run.")
    parser.add_argument("--out",      default="./plots", help="Output directory for PNG files.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df_base  = _load(args.baseline, "Baseline")
    df_cache = _load(args.cached,   "APC")

    if len(df_base) != len(df_cache):
        print(f"Warning: query counts differ — baseline={len(df_base)}, cached={len(df_cache)}. "
              "Plots will use the shorter run's length.")
        n = min(len(df_base), len(df_cache))
        df_base  = df_base.iloc[:n]
        df_cache = df_cache.iloc[:n]

    plot_per_query_latency(df_base, df_cache, args.out)
    plot_per_query_tokens(df_base, df_cache, args.out)
    plot_cumulative_latency(df_base, df_cache, args.out)
    plot_cumulative_tokens(df_base, df_cache, args.out)
    plot_summary(df_base, df_cache, args.out)

    print(f"\nAll plots saved to: {args.out}/")


if __name__ == "__main__":
    main()
