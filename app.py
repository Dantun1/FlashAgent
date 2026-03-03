"""
app.py — Streamlit dashboard for APC (Agentic Plan Caching) benchmark results.

Sections:
  1. Cache Hit Overview   — KPI tiles, donut chart, query-level status timeline
  2. Per-Hit Latency      — Box plot (Baseline vs APC) + stats table
  3. Per-Hit Token & Cost — Token/cost bar charts + cosine similarity
  4. Query Explorer       — Filterable data table
  5. Cumulative Impact    — Running latency & cost saved over all queries

Run with:
    streamlit run app.py
"""

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASELINE_COLOR = "#e05c5c"
CACHED_COLOR   = "#4a90d9"

st.set_page_config(page_title="APC Dashboard", page_icon="⚡", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data")
    baseline_path = st.text_input("Baseline CSV", "data/baseline.csv")
    cached_path   = st.text_input("APC CSV",      "data/cache_telemetry.csv")
    if st.button("↻ Refresh now", use_container_width=True):
        st.cache_data.clear()

    st.divider()
    st.header("Live update")
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_secs = st.slider(
        "Interval (s)", min_value=1, max_value=10, value=2,
        disabled=not auto_refresh,
    )

    st.divider()
    st.header("Pricing — Gemini 2.5 Pro")
    input_price  = st.number_input("Input tokens $/M",  value=1.25,  step=0.25, format="%.2f")
    output_price = st.number_input("Output tokens $/M", value=10.00, step=0.50, format="%.2f")


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=1)
def read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if df.empty:
        return df
    df["total_tokens"] = df["vertex_input_tokens"] + df["vertex_output_tokens"]
    return df.sort_values("query_index").reset_index(drop=True)


df_base  = read_csv(baseline_path)
df_cache = read_csv(cached_path)

if df_base.empty:
    st.error(f"Cannot read baseline CSV: **{baseline_path}**")
    st.stop()

n_total = len(df_base)
n_done  = len(df_cache) if not df_cache.empty else 0


# ── Derived data ──────────────────────────────────────────────────────────────
def _cost(df: pd.DataFrame) -> pd.Series:
    return (
        df["vertex_input_tokens"]  * input_price  / 1_000_000 +
        df["vertex_output_tokens"] * output_price / 1_000_000
    )


if not df_cache.empty:
    df_b = df_base[df_base["query_index"].isin(df_cache["query_index"])].copy()
    df_c = df_cache.copy()

    df_b["cost"] = _cost(df_b)
    df_c["cost"] = _cost(df_c)

    # Rename columns to avoid merge conflicts
    base_metrics = df_b[["query_index", "latency_ms", "total_tokens", "cost"]].rename(
        columns={"latency_ms": "lat_base", "total_tokens": "tok_base", "cost": "cost_base"}
    )
    cache_view = df_c[[
        "query_index", "latency_ms", "total_tokens", "cost",
        "cache_status", "cosine_similarity", "matched_blueprint_tag",
        "query_classification", "original_query",
    ]].rename(columns={
        "latency_ms":            "lat_apc",
        "total_tokens":          "tok_apc",
        "cost":                  "cost_apc",
        "cache_status":          "status",
        "cosine_similarity":     "similarity",
        "matched_blueprint_tag": "blueprint",
        "query_classification":  "q_type",
        "original_query":        "query",
    })

    df_cmp = base_metrics.merge(cache_view, on="query_index")

    n_hits   = int((df_cmp["status"] == "HIT").sum())
    n_miss   = n_done - n_hits
    hit_rate = n_hits / n_done * 100 if n_done else 0.0

    total_lat_saved  = float((df_cmp["lat_base"]  - df_cmp["lat_apc"]).sum())
    total_tok_saved  = float((df_cmp["tok_base"]  - df_cmp["tok_apc"]).sum())
    total_cost_saved = float((df_cmp["cost_base"] - df_cmp["cost_apc"]).sum())

    # HIT-only subset
    df_hits = df_cmp[df_cmp["status"] == "HIT"].copy()
    df_hits["lat_saved"]  = df_hits["lat_base"]  - df_hits["lat_apc"]
    df_hits["tok_saved"]  = df_hits["tok_base"]  - df_hits["tok_apc"]
    df_hits["cost_saved"] = df_hits["cost_base"] - df_hits["cost_apc"]

    # Cumulative series
    df_cum = df_cmp[["query_index", "lat_base", "lat_apc", "cost_base", "cost_apc"]].copy()
    df_cum["cum_lat_base"]  = df_cum["lat_base"].cumsum()
    df_cum["cum_lat_apc"]   = df_cum["lat_apc"].cumsum()
    df_cum["cum_cost_base"] = df_cum["cost_base"].cumsum()
    df_cum["cum_cost_apc"]  = df_cum["cost_apc"].cumsum()

else:
    n_hits = n_miss = 0
    hit_rate = total_lat_saved = total_tok_saved = total_cost_saved = 0.0
    df_cmp = df_hits = df_cum = pd.DataFrame()


# ── Title ─────────────────────────────────────────────────────────────────────
st.title("⚡ APC — Agentic Plan Caching Dashboard")
st.caption(
    "Real-time impact of the APC system on the **FinBench custom dataset**. "
    f"Baseline: **{n_total} queries** | "
    f"APC run: **{n_done} / {n_total}** queries processed."
)
st.progress(
    min(1.0, n_done / n_total) if n_total else 0.0,
    text=f"APC run progress: **{n_done} / {n_total}** queries",
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Cache Hit Overview
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. Cache Hit Overview")

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Cache Hit Rate",
    f"{hit_rate:.1f}%" if n_done else "—",
    f"{n_hits} hits / {n_done} processed" if n_done else "No data yet",
)
k2.metric(
    "Latency Saved",
    f"{total_lat_saved / 1000:.1f} s" if n_done else "—",
    f"{total_lat_saved:,.0f} ms total" if n_done else "",
)
k3.metric(
    "Tokens Saved",
    f"{total_tok_saved:,.0f}" if n_done else "—",
)
k4.metric(
    "Est. Cost Saved",
    f"${total_cost_saved:.4f}" if n_done else "—",
    "Gemini 2.5 Pro pricing",
)

st.write("")
col_a, col_b = st.columns([1, 2])

with col_a:
    fig_donut = go.Figure(go.Pie(
        labels=["HIT", "MISS"],
        values=[n_hits, n_miss],
        hole=0.55,
        marker_colors=[CACHED_COLOR, BASELINE_COLOR],
        textinfo="label+percent+value",
        hovertemplate="%{label}: %{value} queries (%{percent})<extra></extra>",
    ))
    fig_donut.update_layout(
        title="HIT vs MISS distribution",
        showlegend=False,
        margin=dict(t=40, b=0, l=0, r=0),
        height=300,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_b:
    if not df_cmp.empty:
        fig_tl = go.Figure()
        for status, color in [("HIT", CACHED_COLOR), ("MISS", BASELINE_COLOR)]:
            sub = df_cmp[df_cmp["status"] == status]
            fig_tl.add_trace(go.Scatter(
                x=sub["query_index"],
                y=[status] * len(sub),
                mode="markers",
                name=status,
                marker=dict(color=color, size=10, opacity=0.8),
            ))
        fig_tl.update_layout(
            title="Cache status per query",
            xaxis_title="Query index",
            yaxis=dict(categoryorder="array", categoryarray=["MISS", "HIT"]),
            height=300,
            legend_title_text="",
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.info("Waiting for APC run data…")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Per-Hit Latency: Baseline vs APC
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("2. Per-Hit Latency: Baseline vs APC")

if not df_hits.empty:
    speedup = df_hits["lat_base"].mean() / df_hits["lat_apc"].mean()
    st.markdown(
        f"Each cache hit replaced a Gemini blueprint-generation call "
        f"(**~{df_hits['lat_base'].mean():,.0f} ms** on average) with a vector-index lookup "
        f"(**~{df_hits['lat_apc'].mean():,.0f} ms**) — a **{speedup:.0f}× speedup**."
    )

    col_box1, col_box2 = st.columns(2)
    for col_st, label, data_col, color in [
        (col_box1, "Baseline",  "lat_base", BASELINE_COLOR),
        (col_box2, "APC (HIT)", "lat_apc",  CACHED_COLOR),
    ]:
        fig_box = go.Figure(go.Box(
            y=df_hits[data_col],
            name=label,
            marker_color=color,
            boxmean=True,
            line_width=2,
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
            marker=dict(size=5, opacity=0.5),
        ))
        fig_box.update_layout(
            title=f"{label} latency ({n_hits} queries)",
            yaxis_title="Latency (ms)",
            yaxis_tickformat=",",
            height=420,
            showlegend=False,
            margin=dict(t=40, b=0, l=0, r=0),
        )
        col_st.plotly_chart(fig_box, use_container_width=True)

    # Compact stats table
    stats_df = pd.DataFrame({
        "Metric":     ["Min (ms)", "Median (ms)", "Mean (ms)", "Max (ms)"],
        "Baseline":   [
            f"{df_hits['lat_base'].min():,.0f}",
            f"{df_hits['lat_base'].median():,.0f}",
            f"{df_hits['lat_base'].mean():,.0f}",
            f"{df_hits['lat_base'].max():,.0f}",
        ],
        "APC (HIT)":  [
            f"{df_hits['lat_apc'].min():,.0f}",
            f"{df_hits['lat_apc'].median():,.0f}",
            f"{df_hits['lat_apc'].mean():,.0f}",
            f"{df_hits['lat_apc'].max():,.0f}",
        ],
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

else:
    st.info("No cache hits yet.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Per-Hit Token & Cost Savings
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("3. Per-Hit Token & Cost Savings")

if not df_hits.empty:
    avg_tok  = df_hits["tok_saved"].mean()
    avg_cost = df_hits["cost_saved"].mean()
    st.markdown(
        f"Cache hits consume **0 tokens** (blueprint retrieved from vector index). "
        f"Per hit: **{avg_tok:,.0f} tokens** ≈ **${avg_cost:.5f}** saved. "
        f"Across all {n_hits} hits: **${total_cost_saved:.4f}** total."
    )

    df_h = df_hits.sort_values("query_index")

    col_e, col_f = st.columns(2)
    with col_e:
        fig_tok = go.Figure(go.Bar(
            x=df_h["query_index"],
            y=df_h["tok_saved"],
            marker_color=CACHED_COLOR,
            hovertemplate="Query %{x}<br>Tokens saved: %{y:,}<extra></extra>",
        ))
        fig_tok.update_layout(
            title="Tokens saved per cache hit",
            xaxis_title="Query index",
            yaxis_title="Tokens saved",
            height=320,
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_tok, use_container_width=True)

    with col_f:
        fig_cost = go.Figure(go.Bar(
            x=df_h["query_index"],
            y=df_h["cost_saved"],
            marker_color=CACHED_COLOR,
            hovertemplate="Query %{x}<br>Cost saved: $%{y:.5f}<extra></extra>",
        ))
        fig_cost.update_layout(
            title="Estimated cost saved per cache hit ($)",
            xaxis_title="Query index",
            yaxis_title="Cost saved ($)",
            yaxis_tickformat=".5f",
            height=320,
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    # Cosine similarity for each hit
    st.subheader("Blueprint match quality (cosine similarity)")
    st.caption(
        "How confidently the APC engine matched each query to a cached blueprint. "
        "Threshold: **0.90**."
    )
    sim_floor = max(0.86, float(df_h["similarity"].min()) - 0.02)
    fig_sim = go.Figure(go.Bar(
        x=df_h["query_index"],
        y=df_h["similarity"],
        marker_color=CACHED_COLOR,
        hovertemplate="Query %{x}<br>Similarity: %{y:.4f}<extra></extra>",
    ))
    fig_sim.add_hline(
        y=0.90, line_dash="dash", line_color="grey",
        annotation_text="Threshold (0.90)",
        annotation_position="top right",
    )
    fig_sim.update_layout(
        title="Cosine similarity score for each cache hit",
        xaxis_title="Query index",
        yaxis_title="Cosine similarity",
        yaxis_range=[sim_floor, 1.01],
        height=280,
        margin=dict(t=40, b=0, l=0, r=0),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

else:
    st.info("No cache hits yet.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Query Explorer
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("4. Query Explorer")

if not df_cmp.empty:
    status_filter = st.multiselect(
        "Filter by cache status",
        options=["HIT", "MISS"],
        default=["HIT", "MISS"],
    )
    df_disp = df_cmp[df_cmp["status"].isin(status_filter)].copy()
    df_disp["blueprint"] = df_disp["blueprint"].str[:70]
    df_disp = df_disp[[
        "query_index", "query", "q_type", "status", "similarity",
        "lat_apc", "tok_apc", "blueprint",
    ]].rename(columns={
        "query_index": "#",
        "query":       "Query",
        "q_type":      "Type",
        "status":      "Status",
        "similarity":  "Similarity",
        "lat_apc":     "Latency (ms)",
        "tok_apc":     "Tokens",
        "blueprint":   "Matched blueprint",
    })
    st.dataframe(df_disp, use_container_width=True, hide_index=True, height=380)
else:
    st.info("Waiting for APC run data…")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Cumulative Impact
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("5. Cumulative Impact")

if not df_cum.empty:
    col_g, col_h = st.columns(2)

    with col_g:
        final_lat_saved = float(df_cum["cum_lat_base"].iloc[-1] - df_cum["cum_lat_apc"].iloc[-1])
        fig_cl = go.Figure()
        fig_cl.add_trace(go.Scatter(
            x=df_cum["query_index"], y=df_cum["cum_lat_base"],
            name="Baseline", line=dict(color=BASELINE_COLOR, width=2),
        ))
        fig_cl.add_trace(go.Scatter(
            x=df_cum["query_index"], y=df_cum["cum_lat_apc"],
            name="APC", line=dict(color=CACHED_COLOR, width=2),
            fill="tonexty", fillcolor="rgba(74,144,217,0.10)",
        ))
        fig_cl.add_annotation(
            x=df_cum["query_index"].iloc[-1],
            y=float(df_cum["cum_lat_apc"].iloc[-1]),
            text=f"Saved {final_lat_saved / 1000:.1f}s total",
            showarrow=True, arrowhead=2, ax=-80, ay=-30,
            font=dict(color="grey", size=11),
        )
        fig_cl.update_layout(
            title="Cumulative latency: Baseline vs APC",
            xaxis_title="Query index",
            yaxis_title="Cumulative latency (ms)",
            yaxis_tickformat=",",
            height=360,
            margin=dict(t=40, b=0, l=0, r=0),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_cl, use_container_width=True)

    with col_h:
        fig_cc = go.Figure()
        fig_cc.add_trace(go.Scatter(
            x=df_cum["query_index"], y=df_cum["cum_cost_base"],
            name="Baseline", line=dict(color=BASELINE_COLOR, width=2),
        ))
        fig_cc.add_trace(go.Scatter(
            x=df_cum["query_index"], y=df_cum["cum_cost_apc"],
            name="APC", line=dict(color=CACHED_COLOR, width=2),
            fill="tonexty", fillcolor="rgba(74,144,217,0.10)",
        ))
        fig_cc.update_layout(
            title="Cumulative estimated cost: Baseline vs APC",
            xaxis_title="Query index",
            yaxis_title="Cumulative cost ($)",
            yaxis_tickformat=".4f",
            height=360,
            margin=dict(t=40, b=0, l=0, r=0),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_cc, use_container_width=True)

else:
    st.info("Waiting for APC run data…")


# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()
