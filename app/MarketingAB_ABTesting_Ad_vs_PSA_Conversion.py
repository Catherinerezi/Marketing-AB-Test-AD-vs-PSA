# -*- coding: utf-8 -*-
"""
MarketingAB_ABTesting_Ad_vs_PSA_streamlit.py

Streamlit app for:
- A/B testing conversion rate: 'ad' vs 'psa'
- EDA (conversion by day/hour, total ads distribution)
- Statistical tests (two-proportion z-test + chi-square + Welch t-test)
- Effect size (Cohen's h) + confidence intervals
- Power / sample size estimation (NormalIndPower)
- Randomization / permutation test simulation
- Altair-only visualizations (fully Altair)
"""

import os
import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from scipy import stats
from statsmodels.stats.power import NormalIndPower
from pathlib import Path

# Streamlit config
st.set_page_config(
    page_title="A/B Testing Conversion — Ad vs PSA",
    layout="wide",
)

st.title("📣 A/B Testing: Conversion Rate — 'ad' vs 'psa'")
st.caption("Dataset: marketing_AB.csv | Fokus: perbedaan conversion rate (converted) antara grup 'ad' dan 'psa'.")

# Altair settings (safe for aggregated data)
alt.data_transformers.disable_max_rows()

# Helpers
def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    p1 = np.clip(p1, 1e-12, 1 - 1e-12)
    p2 = np.clip(p2, 1e-12, 1 - 1e-12)
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

def wald_ci_diff(p1, n1, p2, n2, alpha=0.05):
    """Wald CI for difference in proportions (p1 - p2)."""
    diff = p1 - p2
    se = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
    z = stats.norm.ppf(1 - alpha / 2)
    return diff - z * se, diff + z * se

def two_prop_ztest(x1, n1, x2, n2, alternative="two-sided"):
    """
    Two-proportion z-test (manual).
    alternative: "two-sided", "larger" (p1>p2), "smaller" (p1<p2)
    """
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan

    z = (p1 - p2) / se

    if alternative == "two-sided":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "larger":
        p = 1 - stats.norm.cdf(z)
    elif alternative == "smaller":
        p = stats.norm.cdf(z)
    else:
        raise ValueError("alternative must be: two-sided | larger | smaller")
    return z, p

def safe_read_csv(path_or_buf):
    return pd.read_csv(path_or_buf)

@st.cache_data(show_spinner=True)
def load_repo_csv():
    """
    Cari CSV di beberapa kandidat path:
    - root repo: marketing_AB.csv
    - raw_data/marketing_AB.csv
    - folder yang sama dengan file app
    - parent folder dari app (repo root jika struktur: repo/app/app.py)
    """
    script_path = Path(__file__).resolve()
    # Jika file app ada di folder "app", maka repo root = parent dari folder app
    repo_root = script_path.parents[1] if script_path.parent.name == "app" else script_path.parent

    candidates = [
        repo_root / "marketing_AB.csv",
        repo_root / "raw_data" / "marketing_AB.csv",
        script_path.parent / "marketing_AB.csv",
        script_path.parent / "raw_data" / "marketing_AB.csv",
    ]

    for p in candidates:
        if p.exists():
            return pd.read_csv(p), str(p)

    return None, None


@st.cache_data(show_spinner=True)
def load_from_url(url: str):
    return pd.read_csv(url)

def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # keep original columns, but normalize whitespace for robustness
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def ensure_converted_numeric(df: pd.DataFrame, col="converted") -> pd.DataFrame:
    df = df.copy()
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)
    else:
        # if already 0/1 or True/False strings
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().map({"true": 1, "false": 0})
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# Sidebar: data loading
with st.sidebar:
    st.header("⚙️ Pengaturan")
    st.subheader("Data source")

    mode = st.radio(
        "Pilih sumber data:",
        ["Upload CSV", "Baca file repo (marketing_AB.csv)", "URL CSV (raw)"],
        index=1,
    )

    uploaded = None
    url_in = None

    if mode == "Upload CSV":
        uploaded = st.file_uploader("Upload marketing_AB.csv", type=["csv"])
    elif mode == "URL CSV (raw)":
        url_in = st.text_input(
            "Tempel URL CSV (contoh: raw GitHub / public link)",
            value="",
            placeholder="https://.../marketing_AB.csv",
        )

    st.divider()
    st.subheader("Visual sampling")
    max_points = st.slider(
        "Max baris untuk chart yang butuh data mentah (sampling)",
        min_value=5000,
        max_value=150000,
        value=50000,
        step=5000,
    )

    st.divider()
    st.subheader("Hipotesis / test")
    alt_choice = st.selectbox(
        "Alternative hypothesis untuk z-test:",
        ["two-sided (beda)", "larger (ad > psa)", "smaller (ad < psa)"],
        index=0,
    )
    alpha = st.slider("Alpha (significance level)", 0.01, 0.10, 0.05, 0.01)

    st.divider()
    st.subheader("Power & sample size")
    target_power = st.slider("Target power", 0.70, 0.95, 0.80, 0.01)
    h_input = st.slider("Effect size (Cohen's h)", 0.05, 0.80, 0.20, 0.05)

    st.divider()
    st.subheader("Permutation test")
    n_perm = st.slider("Jumlah permutasi (simulasi)", 200, 3000, 800, 100)

# Load data
df_raw = None

if mode == "Upload CSV":
    if uploaded is None:
        st.info("Upload file CSV untuk memulai.")
        st.stop()
    df_raw = safe_read_csv(uploaded)

elif mode == "Baca file repo (marketing_AB.csv)":
    df_raw, found_path = load_repo_csv()
    if df_raw is None:
        st.error(
            "File 'marketing_AB.csv' tidak ditemukan di root repo.\n\n"
            "✅ Solusi cepat:\n"
            "- Letakkan marketing_AB.csv di root repo, atau\n"
            "- Gunakan 'Upload CSV', atau\n"
            "- Gunakan 'URL CSV (raw)'."
        )
        st.stop()
    st.sidebar.success(f"Loaded from: {found_path}")

else:
    if not url_in:
        st.info("Masukkan URL CSV (raw) untuk memulai.")
        st.stop()
    try:
        df_raw = load_from_url(url_in)
    except Exception as e:
        st.error(f"Gagal membaca URL. Error: {e}")
        st.stop()

df = standardize_cols(df_raw)

# Column mapping
COL_GROUP = "test group"
COL_CONV = "converted"
COL_TOTAL_ADS = "total ads"
COL_DAY = "most ads day"
COL_HOUR = "most ads hour"
COL_USER = "user id"

needed = [COL_GROUP, COL_CONV]
missing = [c for c in needed if c not in df.columns]
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {missing}\n\nKolom tersedia: {list(df.columns)}")
    st.stop()

df = ensure_converted_numeric(df, COL_CONV)

# Basic cleaning for common issues
df[COL_GROUP] = df[COL_GROUP].astype(str).str.strip().str.lower()

valid_groups = set(df[COL_GROUP].dropna().unique().tolist())
if not {"ad", "psa"}.issubset(valid_groups):
    st.error(f"Grup harus mengandung 'ad' dan 'psa'. Ditemukan: {sorted(list(valid_groups))}")
    st.stop()

# Layout: Overview
with st.expander("🎯 Objective & Hypothesis", expanded=True):
    st.markdown(
        f"""
**Objective**  
Menguji apakah terdapat perbedaan signifikan pada **conversion rate** (`{COL_CONV}`) antara:
- **Treatment**: `ad`
- **Control**: `psa`

**Hipotesis**
- **H0**: Tidak ada perbedaan conversion rate antara `ad` dan `psa`
- **H1**: Ada perbedaan conversion rate antara `ad` dan `psa` (atau sesuai pilihan alternative di sidebar)
"""
    )

tab_overview, tab_eda, tab_stats, tab_perm = st.tabs(
    ["📌 Overview Data", "🔎 EDA (Altair)", "🧪 Statistik & Power", "🎲 Permutation Test"]
)

# Overview tab
with tab_overview:
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        st.subheader("Shape & Head")
        st.write("Shape:", df.shape)
        st.dataframe(df.head(10), use_container_width=True)
    with c2:
        st.subheader("Dtypes")
        st.json({col: str(tp) for col, tp in df.dtypes.items()})
    with c3:
        st.subheader("Missing (%)")
        miss = (df.isna().mean() * 100).sort_values(ascending=False).round(3).reset_index()
        miss.columns = ["column", "missing_pct"]
        st.dataframe(miss, use_container_width=True)

    st.subheader("Komposisi grup")
    grp_counts = df[COL_GROUP].value_counts(dropna=False).rename_axis("group").reset_index(name="n")
    chart_grp = (
        alt.Chart(grp_counts)
        .mark_bar()
        .encode(
            x=alt.X("n:Q", title="Jumlah user (rows)"),
            y=alt.Y("group:N", sort="-x", title="Group"),
            tooltip=["group", "n"],
        )
        .properties(height=180)
        .interactive()
    )
    st.altair_chart(chart_grp, use_container_width=True)

# EDA tab
with tab_eda:
    st.subheader("Ringkasan conversion rate per grup")

    summary = (
        df.groupby(COL_GROUP)[COL_CONV]
        .agg(total_users="count", total_converted="sum")
        .reset_index()
    )
    summary["conversion_rate"] = summary["total_converted"] / summary["total_users"]
    summary["conversion_rate_pct"] = (summary["conversion_rate"] * 100).round(3)

    cA, cB = st.columns([1, 2])
    with cA:
        st.dataframe(
            summary.rename(columns={COL_GROUP: "group"})[
                ["group", "total_users", "total_converted", "conversion_rate_pct"]
            ],
            use_container_width=True,
        )

        # KPI cards
        p_ad = float(summary.loc[summary[COL_GROUP] == "ad", "conversion_rate"].iat[0])
        p_psa = float(summary.loc[summary[COL_GROUP] == "psa", "conversion_rate"].iat[0])
        lift_abs = (p_ad - p_psa)
        lift_rel = (p_ad / p_psa - 1) if p_psa > 0 else np.nan

        st.metric("CR (ad)", f"{p_ad*100:.3f}%")
        st.metric("CR (psa)", f"{p_psa*100:.3f}%")
        st.metric("Lift (absolute)", f"{lift_abs*100:.3f} pp")
        st.metric("Lift (relative)", f"{lift_rel*100:.2f}%" if np.isfinite(lift_rel) else "NA")

    with cB:
        chart_cr = (
            alt.Chart(summary)
            .mark_bar()
            .encode(
                x=alt.X("conversion_rate:Q", title="Conversion Rate"),
                y=alt.Y(f"{COL_GROUP}:N", sort="-x", title="Group"),
                tooltip=[
                    alt.Tooltip(COL_GROUP, title="Group"),
                    alt.Tooltip("total_users:Q", title="Total users"),
                    alt.Tooltip("total_converted:Q", title="Converted"),
                    alt.Tooltip("conversion_rate:Q", title="CR", format=".4f"),
                ],
            )
            .properties(height=220, title="Conversion Rate by Group")
            .interactive()
        )
        st.altair_chart(chart_cr, use_container_width=True)

    st.divider()

    # Optional dimensions exist
    has_day = COL_DAY in df.columns
    has_hour = COL_HOUR in df.columns
    has_total_ads = COL_TOTAL_ADS in df.columns

    c4, c5 = st.columns(2)

    # Conversion by day
    with c4:
        st.subheader("Conversion rate by day (most ads day)")
        if has_day:
            day_agg = (
                df.groupby([COL_GROUP, COL_DAY])[COL_CONV]
                .agg(n="count", conv="sum")
                .reset_index()
            )
            day_agg["cr"] = day_agg["conv"] / day_agg["n"]

            day_chart = (
                alt.Chart(day_agg)
                .mark_line(point=True)
                .encode(
                    x=alt.X(f"{COL_DAY}:N", sort=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], title="Day"),
                    y=alt.Y("cr:Q", title="Conversion rate"),
                    color=alt.Color(f"{COL_GROUP}:N", title="Group"),
                    tooltip=[
                        COL_GROUP,
                        alt.Tooltip(COL_DAY, title="Day"),
                        alt.Tooltip("n:Q", title="N"),
                        alt.Tooltip("cr:Q", title="CR", format=".4f"),
                    ],
                )
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(day_chart, use_container_width=True)
        else:
            st.info(f"Kolom '{COL_DAY}' tidak ada di dataset.")

    # Conversion by hour
    with c5:
        st.subheader("Conversion rate by hour (most ads hour)")
        if has_hour:
            hour_agg = (
                df.groupby([COL_GROUP, COL_HOUR])[COL_CONV]
                .agg(n="count", conv="sum")
                .reset_index()
            )
            hour_agg["cr"] = hour_agg["conv"] / hour_agg["n"]

            hour_chart = (
                alt.Chart(hour_agg)
                .mark_line(point=True)
                .encode(
                    x=alt.X(f"{COL_HOUR}:Q", title="Hour (0–23)"),
                    y=alt.Y("cr:Q", title="Conversion rate"),
                    color=alt.Color(f"{COL_GROUP}:N", title="Group"),
                    tooltip=[
                        COL_GROUP,
                        alt.Tooltip(COL_HOUR, title="Hour"),
                        alt.Tooltip("n:Q", title="N"),
                        alt.Tooltip("cr:Q", title="CR", format=".4f"),
                    ],
                )
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(hour_chart, use_container_width=True)
        else:
            st.info(f"Kolom '{COL_HOUR}' tidak ada di dataset.")

    st.divider()

    # Total ads distribution and relationship
    if has_total_ads:
        st.subheader("Total ads distribution & conversion relationship (sampled)")

        # Sample for raw plots
        df_sample = df.sample(
            n=min(max_points, len(df)),
            random_state=42
        ).copy()

        c6, c7 = st.columns(2)

        with c6:
            hist_ads = (
                alt.Chart(df_sample)
                .mark_bar()
                .encode(
                    x=alt.X(f"{COL_TOTAL_ADS}:Q", bin=alt.Bin(maxbins=40), title="Total ads (binned)"),
                    y=alt.Y("count():Q", title="Count"),
                    tooltip=[alt.Tooltip("count():Q", title="Count")],
                    color=alt.Color(f"{COL_GROUP}:N", title="Group"),
                )
                .properties(height=300, title="Distribution of total ads (sampled)")
                .interactive()
            )
            st.altair_chart(hist_ads, use_container_width=True)

        with c7:
            # conversion rate by total ads bins (aggregated)
            bins = pd.cut(df[COL_TOTAL_ADS], bins=30)  # full aggregation
            ads_bin_agg = (
                df.assign(total_ads_bin=bins)
                .groupby([COL_GROUP, "total_ads_bin"])[COL_CONV]
                .agg(n="count", conv="sum")
                .reset_index()
            )
            ads_bin_agg["cr"] = ads_bin_agg["conv"] / ads_bin_agg["n"]
            ads_bin_agg["bin_label"] = ads_bin_agg["total_ads_bin"].astype(str)

            cr_by_ads = (
                alt.Chart(ads_bin_agg)
                .mark_line(point=False)
                .encode(
                    x=alt.X("bin_label:N", sort=None, title="Total ads bin"),
                    y=alt.Y("cr:Q", title="Conversion rate"),
                    color=alt.Color(f"{COL_GROUP}:N", title="Group"),
                    tooltip=[
                        COL_GROUP,
                        alt.Tooltip("n:Q", title="N"),
                        alt.Tooltip("cr:Q", title="CR", format=".4f"),
                        alt.Tooltip("bin_label:N", title="Bin"),
                    ],
                )
                .properties(height=300, title="Conversion rate vs total ads (binned)")
                .interactive()
            )
            st.altair_chart(cr_by_ads, use_container_width=True)
    else:
        st.info(f"Kolom '{COL_TOTAL_ADS}' tidak ada di dataset, jadi chart 'total ads' dilewati.")

# Stats tab
with tab_stats:
    st.subheader("Uji statistik: perbedaan conversion rate (ad vs psa)")

    # Counts
    ad = df[df[COL_GROUP] == "ad"][COL_CONV]
    psa = df[df[COL_GROUP] == "psa"][COL_CONV]

    n1 = int(ad.shape[0]); x1 = int(ad.sum()); p1 = x1 / n1
    n2 = int(psa.shape[0]); x2 = int(psa.sum()); p2 = x2 / n2

    # Z-test
    alt_map = {
        "two-sided (beda)": "two-sided",
        "larger (ad > psa)": "larger",
        "smaller (ad < psa)": "smaller",
    }
    alternative = alt_map[alt_choice]
    z, pz = two_prop_ztest(x1, n1, x2, n2, alternative=alternative)

    # Chi-square test
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    chi2, pchi, dof, exp = stats.chi2_contingency(table, correction=False)

    # Welch t-test (mirroring notebook)
    t_stat, pt = stats.ttest_ind(ad, psa, equal_var=False)

    # CI + effect size
    ci_lo, ci_hi = wald_ci_diff(p1, n1, p2, n2, alpha=alpha)
    h = cohen_h(p1, p2)

    cS1, cS2, cS3, cS4 = st.columns(4)
    cS1.metric("CR (ad)", f"{p1*100:.3f}%")
    cS2.metric("CR (psa)", f"{p2*100:.3f}%")
    cS3.metric("Diff (ad - psa)", f"{(p1-p2)*100:.3f} pp")
    cS4.metric("Cohen's h", f"{h:.4f}")

    st.markdown("#### Ringkasan hasil test")
    res_tbl = pd.DataFrame(
        [
            {"Test": "Two-proportion z-test", "Statistic": z, "P-value": pz},
            {"Test": "Chi-square independence", "Statistic": chi2, "P-value": pchi},
            {"Test": "Welch t-test (binary as 0/1)", "Statistic": t_stat, "P-value": pt},
        ]
    )
    st.dataframe(res_tbl, use_container_width=True)

    st.markdown("#### Confidence interval (difference in proportions)")
    st.write(
        f"CI {(1-alpha)*100:.0f}% untuk (CR_ad - CR_psa): "
        f"**[{ci_lo*100:.3f}, {ci_hi*100:.3f}] pp**"
    )

    # Visualize diff + CI
    ci_df = pd.DataFrame(
        {
            "metric": ["Diff (ad-psa)"],
            "diff": [p1 - p2],
            "ci_lo": [ci_lo],
            "ci_hi": [ci_hi],
        }
    )
    ci_chart = (
        alt.Chart(ci_df)
        .mark_point(filled=True, size=120)
        .encode(
            x=alt.X("diff:Q", title="Difference in conversion rate (ad - psa)"),
            y=alt.Y("metric:N", title=""),
            tooltip=[
                alt.Tooltip("diff:Q", format=".5f", title="Diff"),
                alt.Tooltip("ci_lo:Q", format=".5f", title="CI low"),
                alt.Tooltip("ci_hi:Q", format=".5f", title="CI high"),
            ],
        )
        .properties(height=120, title="Difference & CI")
    )

    rule = (
        alt.Chart(ci_df)
        .mark_rule()
        .encode(x="ci_lo:Q", x2="ci_hi:Q", y="metric:N")
    )

    st.altair_chart(rule + ci_chart, use_container_width=True)

    # Interpretation block
    st.markdown("#### Interpretasi")
    if np.isfinite(pz) and pz < alpha:
        st.success(
            f"P-value z-test = {pz:.6f} < alpha ({alpha}) → **Tolak H0**.\n\n"
            "Ada perbedaan conversion rate yang signifikan (sesuai alternative yang dipilih)."
        )
    else:
        st.warning(
            f"P-value z-test = {pz:.6f} ≥ alpha ({alpha}) → **Gagal menolak H0**.\n\n"
            "Belum ada bukti cukup bahwa conversion rate berbeda (sesuai alternative yang dipilih)."
        )

    st.divider()
    st.subheader("Power / sample size (NormalIndPower, Cohen's h)")

    analysis = NormalIndPower()
    try:
        n_per_group = analysis.solve_power(
            effect_size=h_input,
            alpha=alpha,
            power=target_power,
            ratio=1.0,
            alternative="two-sided" if alternative == "two-sided" else "larger",
        )
        st.write(
            f"Perkiraan minimum **sample size per grup** untuk effect size (Cohen's h) = **{h_input:.2f}**, "
            f"alpha = **{alpha}**, power = **{target_power:.2f}**:"
        )
        st.metric("n per group (approx)", f"{int(math.ceil(n_per_group))}")
    except Exception as e:
        st.error(f"Gagal menghitung sample size. Error: {e}")

    st.caption(
        "Catatan: Cohen's h = 0.2 (small), 0.5 (medium), 0.8 (large). "
        "Semakin kecil effect size, semakin besar sample yang dibutuhkan."
    )

# Permutation test tab
with tab_perm:
    st.subheader("Permutation / randomization test (simulasi under H0)")

    # Observed diff
    obs_diff = p1 - p2

    conv = df[COL_CONV].to_numpy()
    grp = df[COL_GROUP].to_numpy()

    # indices for ad/psa sizes
    n_ad = int((grp == "ad").sum())
    n_psa = int((grp == "psa").sum())

    rng = np.random.default_rng(42)

    diffs = np.empty(n_perm, dtype=float)

    for i in range(n_perm):
        perm = rng.permutation(conv)
        p_ad_perm = perm[:n_ad].mean()
        p_psa_perm = perm[n_ad:n_ad + n_psa].mean()
        diffs[i] = p_ad_perm - p_psa_perm

    # two-sided permutation p-value around 0
    if alternative == "two-sided":
        p_perm = (np.abs(diffs) >= abs(obs_diff)).mean()
    elif alternative == "larger":
        p_perm = (diffs >= obs_diff).mean()
    else:
        p_perm = (diffs <= obs_diff).mean()

    st.metric("Observed diff (ad - psa)", f"{obs_diff*100:.3f} pp")
    st.metric("Permutation p-value", f"{p_perm:.6f}")

    # Plot distribution
    perm_df = pd.DataFrame({"diff": diffs})
    vline_df = pd.DataFrame({"x": [obs_diff]})

    hist = (
        alt.Chart(perm_df)
        .mark_bar()
        .encode(
            x=alt.X("diff:Q", bin=alt.Bin(maxbins=60), title="Simulated diff (ad - psa)"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="Count")],
        )
        .properties(height=320, title="Permutation distribution under H0")
        .interactive()
    )

    vline = (
        alt.Chart(vline_df)
        .mark_rule(strokeWidth=3)
        .encode(x="x:Q")
    )

    st.altair_chart(hist + vline, use_container_width=True)

    st.caption(
        "Histogram menunjukkan distribusi selisih conversion (ad-psa) ketika assignment dianggap acak (H0). "
        "Garis vertikal = observed diff dari data asli."
    )

# Footer recommendation
st.divider()
st.subheader("✅ Rekomendasi singkat")

# Simple decision using z-test by default
if np.isfinite(pz) and pz < alpha and (p1 > p2):
    st.success(
        f"Grup **'ad'** unggul: CR_ad={p1*100:.3f}% vs CR_psa={p2*100:.3f}% "
        f"(diff={obs_diff*100:.3f} pp). Secara statistik signifikan pada alpha={alpha}."
    )
elif np.isfinite(pz) and pz < alpha and (p1 < p2):
    st.success(
        f"Grup **'psa'** unggul: CR_psa={p2*100:.3f}% vs CR_ad={p1*100:.3f}% "
        f"(diff={obs_diff*100:.3f} pp). Secara statistik signifikan pada alpha={alpha}."
    )
else:
    st.info(
        f"Belum ada bukti signifikan (z-test p={pz:.6f} pada alpha={alpha}). "
        "Pertimbangkan: perpanjang durasi eksperimen, cek segmentasi (day/hour/total ads), "
        "atau desain ulang target effect size & power."
    )

st.caption("Tip deploy: pastikan `streamlit`, `pandas`, `numpy`, `scipy`, `statsmodels`, `altair` ada di requirements.txt.")
