
import re
import calendar
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

# ----------------
# ----------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Google Play Store Analytics",
    layout="wide"
)

# Premium dashboard styling
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(69, 92, 153, .16), transparent 28%),
            linear-gradient(135deg, #0b1020 0%, #101827 55%, #0c1322 100%);
        color: #edf2ff;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111a2d 0%, #0b1220 100%);
        border-right: 1px solid rgba(148, 163, 184, .16);
    }
    [data-testid="stHeader"] { background: rgba(11, 16, 32, .75); }
    .hero {
        padding: 1.35rem 1.6rem;
        border: 1px solid rgba(129, 140, 248, .28);
        border-radius: 18px;
        background: linear-gradient(110deg, rgba(49, 67, 120, .55), rgba(18, 29, 50, .72));
        box-shadow: 0 12px 35px rgba(0,0,0,.20);
        margin-bottom: 1.2rem;
    }
    .hero-kicker {
        color: #a5b4fc; text-transform: uppercase; letter-spacing: .14em;
        font-size: .72rem; font-weight: 700;
    }
    .hero h1 { margin: .25rem 0 .35rem 0; font-size: 2.15rem; }
    .hero p { color: #cbd5e1; margin: 0; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(30, 41, 67, .88), rgba(17, 27, 47, .9));
        border: 1px solid rgba(148, 163, 184, .18);
        border-radius: 14px;
        padding: 15px 17px;
        box-shadow: 0 8px 22px rgba(0,0,0,.15);
    }
    div[data-testid="stMetricLabel"] { color: #aebbd0; }
    div[data-testid="stMetricValue"] { color: #f8fafc; }
    div[data-testid="stPlotlyChart"] {
        border: 1px solid rgba(148,163,184,.13);
        border-radius: 14px;
        padding: .35rem;
        background: rgba(15, 23, 42, .30);
    }
    .section-note {
        padding: .8rem 1rem; border-left: 3px solid #818cf8;
        background: rgba(99,102,241,.10); border-radius: 7px;
        color: #cbd5e1; margin: .6rem 0 1rem 0;
    }
    .stButton button, .stDownloadButton button {
        border-radius: 10px; border: 1px solid rgba(129,140,248,.45);
    }
    footer {visibility: hidden;}
</style>
<div class="hero">
  <div class="hero-kicker">Interactive Business Intelligence</div>
  <h1>Google Play Store Analytics</h1>
  <p>Explore app performance, ratings, installs and category trends through six interactive visualizations.</p>
</div>
""", unsafe_allow_html=True)

WINDOWS = {
    "1 · Clustered Heatmap": (15, 17),
    "2 · Hierarchical Sunburst": (18, 20),
    "3 · Free vs Paid Radar": (13, 14),
    "4 · Monthly Calendar Heatmap": (18, 21),
    "5 · Hexbin Density": (17, 19),
    "6 · Monthly Streamgraph": (16, 18),
}


# --------------------------------------------------
# 2. DATA CLEANING FUNCTIONS
# --------------------------------------------------

def number(value):

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text.lower() == "free":
        return 0.0

    text = (
        text.replace(",", "")
        .replace("+", "")
        .replace("$", "")
    )

    if text.lower() in ("", "nan", "varies with device"):
        return np.nan

    try:
        return float(text)

    except ValueError:
        return np.nan


def size_mb(value):

    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()

    match = re.fullmatch(r"([\d.]+)\s*([MK])", text)

    if not match:
        return np.nan

    value = float(match.group(1))

    if match.group(2) == "K":
        return value / 1024

    return value


def android_version(value):

    match = re.match(r"\s*(\d+(?:\.\d+)?)", str(value))

    return float(match.group(1)) if match else np.nan


# --------------------------------------------------
# 3. LOAD AND CLEAN DATASET
# --------------------------------------------------

@st.cache_data
def clean_apps(file):
    """
    Load and standardize the Google Play Store dataset.

    Keep task-specific fields nullable here. Each visualization applies
    its own required-field filters so a missing Size or Last Updated value
    does not unnecessarily remove an otherwise usable row from Task 5/6.
    """
    df = pd.read_csv(file)

    required = ["App", "Category"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_columns = ["Rating", "Reviews", "Installs", "Price"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = df[column].map(number)
        else:
            df[column] = np.nan

    if "Size" in df.columns:
        df["Size_MB"] = df["Size"].map(size_mb)
    else:
        df["Size_MB"] = np.nan

    if "Android Ver" in df.columns:
        df["Android_Version"] = df["Android Ver"].map(android_version)
    else:
        df["Android_Version"] = np.nan

    if "Last Updated" in df.columns:
        df["Updated_Date"] = pd.to_datetime(
            df["Last Updated"], errors="coerce"
        )
    else:
        df["Updated_Date"] = pd.NaT

    df["Updated_Month"] = df["Updated_Date"].dt.month

    # Standardize text fields without turning missing values into the
    # literal strings "nan" or "None".
    for column in ["App", "Category", "Type", "Content Rating"]:
        if column in df.columns:
            df[column] = df[column].where(df[column].notna(), np.nan)
            df[column] = df[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

    if "Type" in df.columns:
        df["Type"] = df["Type"].astype("string").str.strip().str.title()
        df.loc[df["Type"].str.lower().isin(["nan", "none", ""]), "Type"] = pd.NA
    else:
        df["Type"] = pd.NA

    # Keep revenue explicitly missing when a paid app has no usable price
    # or install count. Free-app estimated revenue is zero.
    is_paid = df["Type"].eq("Paid")
    is_free = df["Type"].eq("Free")
    df["Revenue"] = np.nan
    df.loc[is_free, "Revenue"] = 0.0
    valid_paid_revenue = (
        is_paid
        & df["Installs"].notna()
        & df["Price"].notna()
        & df["Installs"].ge(0)
        & df["Price"].ge(0)
    )
    df.loc[valid_paid_revenue, "Revenue"] = (
        df.loc[valid_paid_revenue, "Installs"]
        * df.loc[valid_paid_revenue, "Price"]
    )

    df["Engagement_Rate"] = np.where(
        df["Installs"].gt(0) & df["Reviews"].notna(),
        df["Reviews"] / df["Installs"] * 100,
        np.nan
    )

    # Remove only rows that cannot be identified as an app/category record.
    df = df.dropna(subset=["App", "Category"]).copy()
    df = df[
        df["App"].astype(str).str.strip().ne("")
        & df["Category"].astype(str).str.strip().ne("")
    ].copy()

    # Validate available values, but retain missing task-specific values.
    # Individual tasks will exclude rows missing fields they actually need.
    df = df[
        (df["Rating"].isna() | df["Rating"].between(0, 5))
        & (df["Reviews"].isna() | df["Reviews"].ge(0))
        & (df["Installs"].isna() | df["Installs"].ge(0))
        & (df["Price"].isna() | df["Price"].ge(0))
        & (df["Size_MB"].isna() | df["Size_MB"].gt(0))
    ].copy()

    # A title can appear in more than one category or app type. Deduplicate
    # within App + Category + Type, keeping the most recent listed update.
    # For equal/missing dates, prefer the record with more reviews, then installs.
    df["_source_order"] = np.arange(len(df))
    sort_columns = ["Updated_Date", "Reviews", "Installs", "_source_order"]
    df = df.sort_values(
        sort_columns,
        ascending=True,
        na_position="first",
        kind="mergesort"
    )
    df = df.drop_duplicates(
        subset=["App", "Category", "Type"],
        keep="last"
    ).copy()
    df = df.drop(columns=["_source_order"])

    return df


# --------------------------------------------------
# 4. TASK 1 FILTERING
# --------------------------------------------------

def filter_task1(df):

    filtered = df[
        (df["Rating"] >= 4.0)
        & (df["Size_MB"] > 10)
        & (df["Installs"] >= 10000)
        & (df["Reviews"] > 1000)
        & (df["Updated_Month"] == 1)
        & (~df["App"].astype(str).str.contains(
            r"\d",
            regex=True,
            na=False
        ))
    ].copy()

    return filtered


# --------------------------------------------------
# 5. CATEGORY-LEVEL METRICS
# --------------------------------------------------

def calculate_metrics(df):

    if df.empty:
        return pd.DataFrame()

    records = []

    for category, group in df.groupby("Category"):

        total_reviews = group["Reviews"].sum()

        total_installs = group["Installs"].sum()

        # Weighted average rating
        weighted_rating = (
            (group["Rating"] * group["Reviews"]).sum()
            / total_reviews
        )

        # Total reviews
        reviews = total_reviews

        # Total installs
        installs = total_installs

        # Average application size
        avg_size = group["Size_MB"].mean()

        # Engagement rate
        engagement = (
            total_reviews / total_installs * 100
            if total_installs > 0
            else 0
        )

        # Update frequency proxy:
        # Distinct update dates in the eligible January records
        update_frequency = group["Updated_Date"].nunique()

        records.append({

            "Category": category,

            "Weighted Rating": weighted_rating,

            "Total Reviews": reviews,

            "Total Installs": installs,

            "Average Size (MB)": avg_size,

            "Engagement Rate (%)": engagement,

            "Update Frequency (Proxy)": update_frequency

        })

    return pd.DataFrame(records).set_index("Category")


# --------------------------------------------------
# 6. Z-SCORE NORMALIZATION
# --------------------------------------------------

def normalize_metrics(metrics):

    normalized = metrics.copy()

    for column in metrics.columns:

        mean = metrics[column].mean()

        std = metrics[column].std(ddof=0)

        if std == 0 or pd.isna(std):

            normalized[column] = 0.0

        else:

            normalized[column] = (
                metrics[column] - mean
            ) / std

    return normalized


# --------------------------------------------------
# 7. COMPOSITE PERFORMANCE SCORE
# --------------------------------------------------

def calculate_composite(normalized, weights):

    total_weight = sum(weights.values())

    if total_weight == 0:
        return pd.Series(
            0.0,
            index=normalized.index
        )

    score = pd.Series(
        0.0,
        index=normalized.index
    )

    for metric, weight in weights.items():

        score += (
            normalized[metric]
            * weight
            / total_weight
        )

    return score


# --------------------------------------------------
# 8. HIERARCHICAL CLUSTERING
# --------------------------------------------------

def cluster_categories(normalized):

    if len(normalized) < 2:

        return list(normalized.index)

    distances = pdist(
        normalized.values,
        metric="euclidean"
    )

    hierarchy = linkage(
        distances,
        method="ward"
    )

    order = leaves_list(hierarchy)

    return normalized.index[order].tolist()


# --------------------------------------------------
# 9. INTERACTIVE CLUSTERED HEATMAP
# --------------------------------------------------

def show_clustered_heatmap(df):

    st.header("Task 1: Interactive Clustered Heatmap")

    st.write(
        "Compare the top 10 Google Play Store categories "
        "using six performance metrics."
    )

    filtered = filter_task1(df)

    st.subheader("Filtering Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Eligible Apps",
        f"{len(filtered):,}"
    )

    col2.metric(
        "Eligible Categories",
        filtered["Category"].nunique()
    )

    col3.metric(
        "January Updated Apps",
        f"{len(filtered):,}"
    )

    if filtered.empty:

        st.warning(
            "No applications satisfy all Task 1 filters. "
            "Check the dataset and filtering conditions."
        )

        return

    metrics = calculate_metrics(filtered)

    if metrics.empty or len(metrics) < 2:

        st.warning(
            "At least two eligible categories are needed "
            "for hierarchical clustering."
        )

        return

    # Normalize all six metrics
    normalized = normalize_metrics(metrics)

    st.subheader("Composite Score Weights")

    st.caption(
        "Adjust the weights to dynamically change category rankings. "
        "All metrics are treated as higher-is-better for this score."
    )

    metric_names = list(metrics.columns)

    weights = {}

    weight_cols = st.columns(3)

    for i, metric in enumerate(metric_names):

        with weight_cols[i % 3]:

            weights[metric] = st.slider(
                metric,
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.5,
                key=f"weight_{i}"
            )

    # Calculate composite score
    scores = calculate_composite(
        normalized,
        weights
    )

    metrics["Composite Score"] = scores

    # Rank all eligible categories
    metrics["Rank"] = (
        metrics["Composite Score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Select top 10 dynamically
    top10 = metrics.nlargest(
        10,
        "Composite Score"
    )

    # Get normalized values for top 10
    top10_normalized = normalized.loc[top10.index]

    # Apply hierarchical clustering
    category_order = cluster_categories(
        top10_normalized
    )

    # Select display mode
    st.subheader("Heatmap Display")

    display_mode = st.radio(
        "Choose metric values",
        ["Normalized (Z-score)", "Raw Values"],
        horizontal=True
    )

    if display_mode == "Normalized (Z-score)":

        display_data = top10_normalized.loc[category_order]

        colorbar_title = "Z-score"

        text_values = display_data.map(
            lambda x: f"{x:.2f}"
        )

    else:

        display_data = metrics.loc[
            category_order,
            metric_names
        ]

        colorbar_title = "Raw Value"

        def format_raw(value):

            if abs(value) >= 1000000:
                return f"{value / 1000000:.2f}M"

            if abs(value) >= 1000:
                return f"{value / 1000:.1f}K"

            return f"{value:.2f}"

        text_values = display_data.map(
            format_raw
        )

    # Highlight top 3 and bottom 3
    top3 = set(
        metrics.nlargest(
            3,
            "Composite Score"
        ).index
    )

    bottom3 = set(
        metrics.nsmallest(
            3,
            "Composite Score"
        ).index
    )

    y_labels = []

    for category in category_order:

        if category in top3:

            label = f"🏆 {category}"

        elif category in bottom3:

            label = f"▼ {category}"

        else:

            label = category

        y_labels.append(label)

    # Build interactive Plotly heatmap
    fig = go.Figure(

        data=go.Heatmap(

            z=display_data.values,

            x=display_data.columns,

            y=y_labels,

            text=text_values.values,

            texttemplate="%{text}",

            colorscale="RdYlGn",

            colorbar=dict(
                title=colorbar_title
            ),

            hovertemplate=(
                "Category: %{y}<br>"
                "Metric: %{x}<br>"
                "Value: %{text}"
                "<extra></extra>"
            )

        )

    )

    fig.update_layout(

        title="Top 10 App Categories — Clustered Performance Heatmap",

        xaxis_title="Performance Metrics",

        yaxis_title="App Categories",

        height=650,

        template="plotly_dark",

        margin=dict(
            l=150,
            r=50,
            t=80,
            b=120
        )

    )

    fig.update_xaxes(
        tickangle=-25
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------
    # 10. TOP AND BOTTOM CATEGORY ANNOTATIONS
    # --------------------------------------------------

    st.subheader("Category Performance Highlights")

    best = metrics.nlargest(
        3,
        "Composite Score"
    )

    worst = metrics.nsmallest(
        3,
        "Composite Score"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🏆 Top 3 Categories")

        st.dataframe(
            best[
                [
                    "Composite Score",
                    "Weighted Rating",
                    "Rank"
                ]
            ].style.format({
                "Composite Score": "{:.3f}",
                "Weighted Rating": "{:.3f}"
            }),
            use_container_width=True
        )

    with col2:

        st.markdown("### ▼ Bottom 3 Categories")

        st.dataframe(
            worst[
                [
                    "Composite Score",
                    "Weighted Rating",
                    "Rank"
                ]
            ].style.format({
                "Composite Score": "{:.3f}",
                "Weighted Rating": "{:.3f}"
            }),
            use_container_width=True
        )

    # --------------------------------------------------
    # 11. COMPLETE CATEGORY RANKING
    # --------------------------------------------------

    st.subheader("Complete Category Ranking")

    ranking = metrics.sort_values(
        "Composite Score",
        ascending=False
    )

    st.dataframe(
        ranking.style.format({
            "Weighted Rating": "{:.3f}",
            "Total Reviews": "{:,.0f}",
            "Total Installs": "{:,.0f}",
            "Average Size (MB)": "{:.2f}",
            "Engagement Rate (%)": "{:.4f}",
            "Update Frequency (Proxy)": "{:,.0f}",
            "Composite Score": "{:.3f}"
        }),
        use_container_width=True
    )

    st.download_button(
        "Download Category Analysis CSV",
        data=ranking.to_csv().encode("utf-8"),
        file_name="clustered_heatmap_analysis.csv",
        mime="text/csv"
    )

    st.info(
        "Methodology: Weighted rating uses review counts as weights. "
        "Engagement = total reviews / total installs × 100. "
        "Update frequency is represented by distinct update dates "
        "because historical update events are unavailable. "
        "Composite scores use the adjustable weighted mean "
        "of the six Z-score metrics."
    )


# --------------------------------------------------
# TASK 2: HIERARCHICAL SUNBURST CHART
# --------------------------------------------------

from pathlib import Path


def translate_category(category):

    translations = {
        "BUSINESS": "ব্যবসা",  # Tamil translation set below
        "TRAVEL & LOCAL": "Voyage et local",
        "PRODUCTIVITY": "Productividad"
    }

    if category.upper() == "BUSINESS":
        return "வணிகம்"

    return translations.get(category.upper(), category)


def load_task2_data():

    file_path = (
        Path(__file__).parent
        / "data"
        / "task2_validated_country_app_master.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            "Task 2 master dataset not found. "
            "Place task2_validated_country_app_master.csv "
            "inside the data folder."
        )

    df = pd.read_csv(file_path)

    required = [
        "app_id",
        "app_name",
        "country_name",
        "category",
        "app_type",
        "rating",
        "review_count",
        "installs",
        "size_mb",
        "eligible_task2"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Task 2 dataset is missing columns: {missing}"
        )

    for col in [
        "rating",
        "review_count",
        "installs",
        "size_mb"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["eligible_task2"] = (
        df["eligible_task2"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    df = df[
        df["eligible_task2"]
        & df["country_name"].notna()
        & df["category"].notna()
        & df["app_id"].notna()
        & df["rating"].ge(4.0)
        & df["review_count"].gt(1000)
        & df["installs"].gt(10000)
        & df["size_mb"].between(15, 80, inclusive="both")
        & df["app_name"].astype(str).str.contains(
            r"\d", regex=True, na=False
        ).eq(False)
        & ~df["category"].astype(str).str.upper().str.startswith(
            ("A", "C", "G", "S")
        )
    ].copy()

    # Keep the latest available country-app observation
    if "scrape_date" in df.columns:
        df["scrape_date"] = pd.to_datetime(
            df["scrape_date"], errors="coerce"
        )

        df = (
            df.sort_values("scrape_date")
            .drop_duplicates(
                subset=["app_id", "country_name"],
                keep="last"
            )
        )
    else:
        df = df.drop_duplicates(
            subset=["app_id", "country_name"],
            keep="last"
        )

    return df


def assign_rating_band(rating):

    if 4.0 <= rating < 4.2:
        return "4.0–4.2"

    elif 4.2 <= rating < 4.5:
        return "4.2–4.5"

    elif 4.5 <= rating < 4.7:
        return "4.5–4.7"

    elif 4.7 <= rating <= 5.0:
        return "4.7–5.0"

    return None


def build_sunburst_nodes(df):

    data = df.copy()

    data["rating_band"] = data["rating"].apply(
        assign_rating_band
    )

    data = data.dropna(subset=["rating_band"])

    # Calculate top 5 categories using global installs
    top_categories = (
        data.groupby("category")["installs"]
        .sum()
        .nlargest(5)
        .index.tolist()
    )

    data = data[
        data["category"].isin(top_categories)
    ].copy()

    data["category_display"] = data["category"].apply(
        translate_category
    )

    # Build hierarchical paths
    levels = [
        ("country_name", "Country"),
        ("category_display", "Category"),
        ("app_type", "App Type"),
        ("rating_band", "Rating Band")
    ]

    total_installs = data["installs"].sum()

    total_reviews = data["review_count"].sum()

    root = {
        "id": "root",
        "label": "Global",
        "parent": "",
        "value": total_installs,
        "rating": (
            (data["rating"] * data["review_count"]).sum()
            / total_reviews
            if total_reviews > 0 else 0
        ),
        "reviews": total_reviews,
        "apps": data["app_id"].nunique(),
        "share": 100.0
    }

    nodes = [root]

    parent_values = {"root": total_installs}

    path_columns = []

    for column, level_name in levels:

        path_columns.append(column)

        grouped = data.groupby(
            path_columns,
            observed=True
        ).agg(
            installs=("installs", "sum"),
            reviews=("review_count", "sum"),
            weighted_rating=(
                "rating",
                lambda x: np.nan
            ),
            apps=("app_id", "nunique")
        ).reset_index()

        # Calculate review-weighted rating for every group
        weighted = (
            data.assign(
                rating_weighted=data["rating"]
                * data["review_count"]
            )
            .groupby(
                path_columns,
                observed=True
            )
            .agg(
                rating_weighted=("rating_weighted", "sum")
            )
            .reset_index()
        )

        grouped = grouped.drop(
            columns=["weighted_rating"]
        ).merge(
            weighted,
            on=path_columns
        )

        grouped["weighted_rating"] = np.where(
            grouped["reviews"] > 0,
            grouped["rating_weighted"] / grouped["reviews"],
            np.nan
        )

        for _, row in grouped.iterrows():

            current_path = [
                str(row[col]) for col in path_columns
            ]

            parent_path = current_path[:-1]

            node_id = "/".join(current_path)

            parent_id = (
                "/".join(parent_path)
                if parent_path else "root"
            )

            value = float(row["installs"])

            parent_value = parent_values.get(
                parent_id, 0
            )

            contribution = (
                value / parent_value * 100
                if parent_value > 0 else 0
            )

            nodes.append({

                "id": node_id,

                "label": current_path[-1],

                "parent": parent_id,

                "value": value,

                "rating": float(
                    row["weighted_rating"]
                ) if pd.notna(row["weighted_rating"]) else 0,

                "reviews": float(row["reviews"]),

                "apps": int(row["apps"]),

                "share": contribution

            })

            parent_values[node_id] = value

    return pd.DataFrame(nodes), top_categories


def show_sunburst():

    st.header(
        "Task 2: Interactive Hierarchical Sunburst"
    )

    st.write(
        "Explore country-wise Google Play Store app performance "
        "through four hierarchical levels."
    )

    try:
        data = load_task2_data()

    except Exception as exc:

        st.error(str(exc))
        return

    if data.empty:

        st.warning(
            "No records satisfy all Task 2 conditions."
        )
        return

    nodes, top_categories = build_sunburst_nodes(data)

    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Eligible Records",
        f"{len(data):,}"
    )

    c2.metric(
        "Countries",
        data["country_name"].nunique()
    )

    c3.metric(
        "Selected Categories",
        len(top_categories)
    )

    st.caption(
        "Top five categories by global installs: "
        + ", ".join(top_categories)
    )

    # Prepare visual properties
    border_colors = [
        "#FFD700" if value > 1_000_000 else "#444444"
        for value in nodes["value"]
    ]

    border_widths = [
        3 if value > 1_000_000 else 1
        for value in nodes["value"]
    ]

    customdata = np.column_stack([
        nodes["value"],
        nodes["rating"],
        nodes["reviews"],
        nodes["share"],
        nodes["apps"]
    ])

    fig = go.Figure(
        go.Sunburst(

            ids=nodes["id"],

            labels=nodes["label"],

            parents=nodes["parent"],

            values=nodes["value"],

            branchvalues="total",

            sort=False,

            maxdepth=5,

            customdata=customdata,

            marker=dict(

                colors=nodes["rating"],

                colorscale="RdYlGn",

                cmin=4.0,

                cmax=5.0,

                colorbar=dict(
                    title="Weighted Rating",
                    thickness=18
                ),

                line=dict(
                    color=border_colors,
                    width=border_widths
                )

            ),

            insidetextorientation="radial",

            hovertemplate=(
                "<b>%{label}</b><br>"
                "Installs: %{customdata[0]:,.0f}<br>"
                "Weighted Rating: %{customdata[1]:.3f}<br>"
                "Reviews: %{customdata[2]:,.0f}<br>"
                "Parent Contribution: %{customdata[3]:.2f}%<br>"
                "Unique Apps: %{customdata[4]:,.0f}"
                "<extra></extra>"
            )

        )
    )

    fig.update_layout(

        title="Country → Category → App Type → Rating Band",

        height=800,

        template="plotly_dark",

        margin=dict(
            t=70,
            l=20,
            r=20,
            b=20
        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True
        }
    )

    st.info(
        "Gold borders identify segments exceeding one million installs. "
        "Color represents review-weighted average rating. "
        "Click a segment to drill down and use the center to move back."
    )

    st.subheader("Top Five Categories by Global Installs")

    category_summary = (
        data.groupby("category")
        .agg(
            Total_Installs=("installs", "sum"),
            Weighted_Rating=(
                "rating",
                lambda x: np.nan
            ),
            Total_Reviews=("review_count", "sum")
        )
    )

    weighted_rating = (
        data.assign(
            weighted=data["rating"] * data["review_count"]
        )
        .groupby("category")["weighted"]
        .sum()
        / data.groupby("category")["review_count"].sum()
    )

    category_summary["Weighted_Rating"] = weighted_rating

    category_summary = (
        category_summary
        .loc[top_categories]
        .sort_values(
            "Total_Installs",
            ascending=False
        )
    )

    st.dataframe(
        category_summary.style.format({
            "Total_Installs": "{:,.0f}",
            "Weighted_Rating": "{:.3f}",
            "Total_Reviews": "{:,.0f}"
        }),
        use_container_width=True
    )

    st.download_button(
        "Download Sunburst Category Summary",
        data=category_summary.to_csv().encode("utf-8"),
        file_name="sunburst_category_summary.csv",
        mime="text/csv"
    )



# --------------------------------------------------
# TASK 3: FREE VS PAID RADAR CHART
# --------------------------------------------------

def show_free_paid_radar(apps):
    st.header("Task 3: Free vs Paid App Performance")
    st.caption(
        "Comparative analysis using six percentile-normalized "
        "performance metrics."
    )

    df = apps.copy()
    required = [
        "App", "Category", "Rating", "Reviews", "Installs",
        "Size_MB", "Price", "Android_Version",
        "Content Rating", "Type"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    numeric_columns = [
        "Rating", "Reviews", "Installs",
        "Size_MB", "Price", "Android_Version"
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Type"] = df["Type"].astype(str).str.strip().str.title()
    df["Category"] = df["Category"].astype(str).str.strip()
    df["App"] = df["App"].astype(str).str.strip()
    df["Content Rating"] = df["Content Rating"].astype(str).str.strip()

    df["Revenue"] = np.where(
        df["Type"].eq("Paid"),
        df["Installs"] * df["Price"],
        0.0
    )
    df["Engagement_Rate"] = np.where(
        df["Installs"] > 0,
        df["Reviews"] / df["Installs"] * 100,
        np.nan
    )

    df = df[
        df["Installs"].ge(10000)
        & df["Android_Version"].gt(4.0)
        & df["Size_MB"].gt(15)
        & df["Content Rating"].str.lower().eq("everyone")
        & df["App"].str.len().le(30)
        & df["Rating"].between(0, 5)
        & df["Reviews"].ge(0)
        & df["Type"].isin(["Free", "Paid"])
    ].copy()

    df = df[
        df["Type"].eq("Free")
        | (df["Type"].eq("Paid") & df["Revenue"].gt(10000))
    ].copy()

    if df.empty:
        st.warning("No apps satisfy all Task 3 filters.")
        return

    top_categories = (
        df.groupby("Category")["Installs"]
        .sum()
        .nlargest(5)
        .index.tolist()
    )
    df = df[df["Category"].isin(top_categories)].copy()

    if df.empty:
        st.warning("No eligible apps remain in the selected categories.")
        return

    percentile_sources = {
        "Installs": "Installs_Percentile",
        "Rating": "Rating_Percentile",
        "Reviews": "Reviews_Percentile",
        "Size_MB": "Size_Percentile",
        "Revenue": "Revenue_Percentile",
        "Engagement_Rate": "Engagement_Percentile",
    }
    for source_col, target_col in percentile_sources.items():
        values = pd.to_numeric(df[source_col], errors="coerce")
        df[target_col] = values.rank(method="average", pct=True) * 100

    def calculate_metrics(group, category_name):
        total_installs = group["Installs"].sum()
        total_reviews = group["Reviews"].sum()

        weighted_rating = (
            (group["Rating"] * group["Reviews"]).sum() / total_reviews
            if total_reviews > 0 else np.nan
        )
        engagement = (
            total_reviews / total_installs * 100
            if total_installs > 0 else np.nan
        )
        rating_percentile = (
            np.average(
                group["Rating_Percentile"],
                weights=group["Reviews"].clip(lower=0)
            )
            if group["Reviews"].sum() > 0
            else group["Rating_Percentile"].mean()
        )

        return {
            "Category": category_name,
            "Type": group["Type"].iloc[0],
            "Average_Installs": group["Installs"].mean(),
            "Weighted_Rating": weighted_rating,
            "Total_Reviews": total_reviews,
            "Average_Size": group["Size_MB"].mean(),
            "Revenue": group["Revenue"].sum(),
            "Engagement_Rate": engagement,
            "Average_Installs_Percentile": group["Installs_Percentile"].mean(),
            "Weighted_Rating_Percentile": rating_percentile,
            "Total_Reviews_Percentile": group["Reviews_Percentile"].mean(),
            "Average_Size_Percentile": group["Size_Percentile"].mean(),
            "Revenue_Percentile": group["Revenue_Percentile"].mean(),
            "Engagement_Rate_Percentile": group["Engagement_Percentile"].mean(),
        }

    records = []
    for category in top_categories:
        category_data = df[df["Category"].eq(category)]
        for app_type in ["Free", "Paid"]:
            group = category_data[category_data["Type"].eq(app_type)]
            if not group.empty:
                records.append(calculate_metrics(group, category))

    for app_type in ["Free", "Paid"]:
        group = df[df["Type"].eq(app_type)]
        if not group.empty:
            records.append(calculate_metrics(group, "Overall"))

    results = pd.DataFrame(records)
    if results.empty:
        st.warning("Not enough data for comparison.")
        return

    normalized_columns = [
        "Average_Installs_Percentile",
        "Weighted_Rating_Percentile",
        "Total_Reviews_Percentile",
        "Average_Size_Percentile",
        "Revenue_Percentile",
        "Engagement_Rate_Percentile",
    ]
    results["Composite_Score"] = results[normalized_columns].mean(axis=1)

    selected_category = st.selectbox(
        "Select category for comparison",
        ["Overall"] + top_categories,
        key="task3_category_selector"
    )
    comparison = results[results["Category"].eq(selected_category)].copy()

    if comparison.empty:
        st.warning("No Free/Paid comparison is available for this selection.")
        return

    comparison = comparison.set_index("Type")
    if not {"Free", "Paid"}.issubset(comparison.index):
        st.warning(
            "Both Free and Paid apps are required for a complete comparison. "
            "One app type has no eligible records in this selection."
        )
        return

    free_score = float(comparison.loc["Free", "Composite_Score"])
    paid_score = float(comparison.loc["Paid", "Composite_Score"])

    if free_score > paid_score:
        better_type = "Free"
    elif paid_score > free_score:
        better_type = "Paid"
    else:
        better_type = "Equal performance"

    st.subheader("Performance Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Free App Score", f"{free_score:.2f}/100")
    c2.metric("Paid App Score", f"{paid_score:.2f}/100")
    c3.metric("Better Performing", better_type)

    st.success(
        f"{better_type} apps perform better in the "
        f"{selected_category} comparison based on the average "
        "of six app-level percentile-normalized metrics."
    )

    metric_labels = [
        "Average Installs",
        "Weighted Rating",
        "Total Reviews",
        "Average Size",
        "Revenue",
        "Engagement Rate",
    ]
    fig = go.Figure()
    for app_type in ["Free", "Paid"]:
        row = comparison.loc[app_type]
        values = [float(row[column]) for column in normalized_columns]
        values.append(values[0])
        labels = metric_labels + [metric_labels[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=labels,
                fill="toself",
                name=app_type,
                opacity=0.65,
                hovertemplate=(
                    f"<b>{app_type} Apps</b><br>"
                    "%{theta}: %{r:.2f} percentile"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title=f"Free vs Paid App Performance — {selected_category}",
        template="plotly_dark",
        height=700,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                title="Percentile Score"
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=80, r=80, t=100, b=60)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Performance Comparison")
    display_columns = [
        "Average_Installs",
        "Weighted_Rating",
        "Total_Reviews",
        "Average_Size",
        "Revenue",
        "Engagement_Rate",
        "Composite_Score",
    ]
    display_table = comparison[display_columns].T
    display_table.columns.name = "App Type"
    st.dataframe(
        display_table.style.format("{:,.3f}"),
        use_container_width=True
    )

    st.subheader("Dataset Information")
    a, b, c = st.columns(3)
    a.metric("Eligible Apps", f"{len(df):,}")
    b.metric("Selected Categories", len(top_categories))
    c.metric(
        "Paid App Revenue",
        f"${df.loc[df['Type'].eq('Paid'), 'Revenue'].sum():,.0f}"
    )
    st.caption(
        "Percentile ranks are calculated across eligible apps before "
        "aggregation by category and app type. Weighted rating percentiles "
        "use review counts as weights. Paid revenue is estimated as installs "
        "multiplied by price; Free app revenue is recorded as zero."
    )


# --------------------------------------------------
# TASK 4: MONTHLY CALENDAR HEATMAP
# --------------------------------------------------

def show_calendar_heatmap(apps):
    st.header("Task 4: Monthly Calendar Heatmap")
    st.caption(
        "Explore daily Google Play Store app update activity "
        "using the Last Updated date."
    )

    data = apps.copy()
    data["Updated_Date"] = pd.to_datetime(
        data["Updated_Date"], errors="coerce"
    )
    data = data.dropna(subset=["Updated_Date"]).copy()

    if data.empty:
        st.warning("No valid update dates are available in this dataset.")
        return

    data["Update_Year"] = data["Updated_Date"].dt.year
    data["Update_Month"] = data["Updated_Date"].dt.month

    available_years = sorted(
        data["Update_Year"].dropna().astype(int).unique().tolist()
    )
    if not available_years:
        st.warning("No valid update years were found.")
        return

    control1, control2 = st.columns(2)
    with control1:
        selected_year = st.selectbox(
            "Select year",
            available_years,
            index=len(available_years) - 1,
            key="task4_year"
        )
    months_with_data = sorted(
        data.loc[data["Update_Year"] == selected_year, "Update_Month"]
        .dropna().astype(int).unique().tolist()
    )
    if not months_with_data:
        st.warning("No update records are available for this year.")
        return

    month_options = {
        month: calendar.month_name[month] for month in months_with_data
    }
    with control2:
        selected_month = st.selectbox(
            "Select month",
            months_with_data,
            format_func=lambda month: month_options[month],
            key="task4_month"
        )

    month_data = data[
        (data["Update_Year"] == selected_year)
        & (data["Update_Month"] == selected_month)
    ].copy()

    # Count distinct app records updated on each calendar date.
    daily = (
        month_data.groupby(month_data["Updated_Date"].dt.day)
        .agg(
            Updated_Apps=("App", "nunique"),
            Average_Rating=("Rating", "mean"),
            Total_Installs=("Installs", "sum")
        )
    )

    days_in_month = calendar.monthrange(
        int(selected_year), int(selected_month)
    )[1]
    first_weekday, _ = calendar.monthrange(
        int(selected_year), int(selected_month)
    )

    # Monday-first calendar grid. Rows are weekdays; columns are weeks.
    weeks = (first_weekday + days_in_month + 6) // 7
    z_values = [[None for _ in range(weeks)] for _ in range(7)]
    hover_text = [["" for _ in range(weeks)] for _ in range(7)]
    day_labels = [["" for _ in range(weeks)] for _ in range(7)]

    for day in range(1, days_in_month + 1):
        date = pd.Timestamp(
            year=int(selected_year),
            month=int(selected_month),
            day=day
        )
        weekday = date.weekday()  # Monday = 0
        week_index = (first_weekday + day - 1) // 7

        if day in daily.index:
            app_count = int(daily.loc[day, "Updated_Apps"])
            avg_rating = daily.loc[day, "Average_Rating"]
            installs = daily.loc[day, "Total_Installs"]
            z_values[weekday][week_index] = app_count
            rating_text = (
                f"{avg_rating:.2f}" if pd.notna(avg_rating) else "N/A"
            )
            hover_text[weekday][week_index] = (
                f"<b>{date.strftime('%d %B %Y')}</b><br>"
                f"Apps updated: {app_count:,}<br>"
                f"Average rating: {rating_text}<br>"
                f"Total installs: {installs:,.0f}"
            )
            day_labels[weekday][week_index] = str(day)
        else:
            # No row for this date means the uploaded snapshot has no
            # observations for that day; it does NOT prove zero updates.
            # None renders as a neutral grey cell (plot background).
            z_values[weekday][week_index] = None
            hover_text[weekday][week_index] = (
                f"<b>{date.strftime('%d %B %Y')}</b><br>"
                "Status: No records in uploaded dataset<br>"
                "A true zero-update day cannot be confirmed from this snapshot."
            )
            day_labels[weekday][week_index] = str(day)

    week_labels = [f"Week {i + 1}" for i in range(weeks)]
    weekday_labels = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=week_labels,
            y=weekday_labels,
            text=day_labels,
            texttemplate="%{text}",
            customdata=hover_text,
            colorscale="YlOrRd",
            colorbar=dict(title="Apps updated"),
            hovertemplate="%{customdata}<extra></extra>",
            xgap=5,
            ygap=5,
            zmin=0
        )
    )
    fig.update_layout(
        title=(
            f"Daily App Update Activity — "
            f"{calendar.month_name[int(selected_month)]} {selected_year}"
        ),
        template="plotly_dark",
        height=540,
        xaxis_title="Calendar Week",
        yaxis_title="Day of Week",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#555555",
        paper_bgcolor="#111111",
        margin=dict(l=110, r=55, t=85, b=65)
    )
    st.plotly_chart(fig, use_container_width=True)

    total_updated = int(daily["Updated_Apps"].sum())
    busiest_day = int(daily["Updated_Apps"].idxmax()) if not daily.empty else 0
    busiest_count = int(daily["Updated_Apps"].max()) if not daily.empty else 0
    average_per_active_day = (
        daily["Updated_Apps"].mean() if not daily.empty else 0
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("App Update Records", f"{total_updated:,}")
    k2.metric(
        "Busiest Update Date",
        f"{busiest_day} {calendar.month_abbr[int(selected_month)]}"
        if busiest_day else "N/A"
    )
    k3.metric("Apps on Busiest Date", f"{busiest_count:,}")

    st.subheader("Daily Update Details")
    daily_table = daily.copy()
    daily_table.index.name = "Day of Month"
    daily_table = daily_table.rename(columns={
        "Updated_Apps": "Apps Updated",
        "Average_Rating": "Average Rating",
        "Total_Installs": "Total Installs"
    })
    st.dataframe(
        daily_table.style.format({
            "Apps Updated": "{:,.0f}",
            "Average Rating": "{:.2f}",
            "Total Installs": "{:,.0f}"
        }),
        use_container_width=True
    )

    st.download_button(
        "Download Monthly Calendar Data",
        data=daily_table.to_csv().encode("utf-8"),
        file_name=(
            f"calendar_heatmap_{selected_year}_"
            f"{int(selected_month):02d}.csv"
        ),
        mime="text/csv"
    )

    st.markdown(
        "**Calendar legend:** Colored cells = dates with recorded app updates; "
        "grey cells = no records for that date in the uploaded dataset."
    )
    st.caption(
        "Methodology: Each colored calendar cell counts unique apps whose "
        "Last Updated date falls on that day. A grey cell means the dataset "
        "has no observation for that date; it must not be interpreted as "
        "proof that zero apps were updated. The uploaded file contains "
        "latest listed update dates, not a complete daily update log."
    )


# --------------------------------------------------
# TASK 5: HEXBIN DENSITY PLOT
# --------------------------------------------------

def show_hexbin_density(apps):
    st.header("Task 5: Hexbin Density Plot")
    st.caption(
        "Explore the relationship between app installs and reviews. "
        "Hexagon color and size-independent intensity represent app density."
    )

    data = apps.copy()
    required = ["App", "Category", "Rating", "Reviews", "Installs"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        st.error(f"Missing required columns for Task 5: {missing}")
        return

    for column in ["Rating", "Reviews", "Installs"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(
        subset=["App", "Category", "Rating", "Reviews", "Installs"]
    )
    data = data[
        data["Rating"].between(0, 5)
        & data["Reviews"].gt(0)
        & data["Installs"].gt(0)
    ].copy()

    if data.empty:
        st.warning(
            "No valid apps have positive review and install counts "
            "for the hexbin analysis."
        )
        return

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        categories = ["All categories"] + sorted(
            data["Category"].astype(str).unique().tolist()
        )
        selected_category = st.selectbox(
            "Category",
            categories,
            key="task5_category"
        )
    with filter_col2:
        min_rating = st.slider(
            "Minimum app rating",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.5,
            key="task5_min_rating"
        )

    if selected_category != "All categories":
        data = data[data["Category"].astype(str) == selected_category]
    data = data[data["Rating"] >= min_rating].copy()

    if len(data) < 3:
        st.warning(
            "At least three apps are required after applying these filters."
        )
        return

    # Log scaling makes highly skewed install/review distributions readable.
    data["Log_Installs"] = np.log10(data["Installs"])
    data["Log_Reviews"] = np.log10(data["Reviews"])

    x = data["Log_Installs"].to_numpy(dtype=float)
    y = data["Log_Reviews"].to_numpy(dtype=float)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_span = max(x_max - x_min, 0.25)
    y_span = max(y_max - y_min, 0.25)

    # Build a pointy-top hexagonal grid using axial/cube coordinates.
    hex_radius = max(x_span, y_span) / 32.0
    sqrt3 = math.sqrt(3.0)

    axial_q = (sqrt3 / 3.0 * x - y / 3.0) / hex_radius
    axial_r = (2.0 / 3.0 * y) / hex_radius

    cube_x = axial_q
    cube_z = axial_r
    cube_y = -cube_x - cube_z

    rounded_x = np.round(cube_x)
    rounded_y = np.round(cube_y)
    rounded_z = np.round(cube_z)

    diff_x = np.abs(rounded_x - cube_x)
    diff_y = np.abs(rounded_y - cube_y)
    diff_z = np.abs(rounded_z - cube_z)

    for i in range(len(x)):
        if diff_x[i] > diff_y[i] and diff_x[i] > diff_z[i]:
            rounded_x[i] = -rounded_y[i] - rounded_z[i]
        elif diff_y[i] > diff_z[i]:
            rounded_y[i] = -rounded_x[i] - rounded_z[i]
        else:
            rounded_z[i] = -rounded_x[i] - rounded_y[i]

    q = rounded_x.astype(int)
    r = rounded_z.astype(int)

    bins = pd.DataFrame({
        "q": q,
        "r": r,
        "Rating": data["Rating"].to_numpy(),
        "Category": data["Category"].astype(str).to_numpy(),
        "Installs": data["Installs"].to_numpy(),
        "Reviews": data["Reviews"].to_numpy()
    })
    grouped = (
        bins.groupby(["q", "r"], as_index=False)
        .agg(
            App_Count=("Rating", "size"),
            Average_Rating=("Rating", "mean"),
            Total_Installs=("Installs", "sum"),
            Total_Reviews=("Reviews", "sum"),
            Categories=("Category", "nunique")
        )
    )

    grouped["Center_X"] = (
        sqrt3 * hex_radius * (grouped["q"] + grouped["r"] / 2.0)
    )
    grouped["Center_Y"] = 1.5 * hex_radius * grouped["r"]

    max_count = max(int(grouped["App_Count"].max()), 1)
    min_count = int(grouped["App_Count"].min())

    def density_color(value):
        fraction = (
            (value - min_count) / (max_count - min_count)
            if max_count > min_count else 1.0
        )
        return sample_colorscale("YlOrRd", [float(fraction)])[0]

    shapes = []
    for row in grouped.itertuples(index=False):
        cx, cy = float(row.Center_X), float(row.Center_Y)
        vertices = []
        for vertex in range(6):
            angle = math.radians(30 + 60 * vertex)
            vertices.append((
                cx + hex_radius * math.cos(angle),
                cy + hex_radius * math.sin(angle)
            ))
        path = "M " + " L ".join(
            f"{vx:.6f},{vy:.6f}" for vx, vy in vertices
        ) + " Z"
        shapes.append({
            "type": "path",
            "path": path,
            "fillcolor": density_color(int(row.App_Count)),
            "line": {"color": "rgba(25,25,25,0.75)", "width": 0.6},
            "layer": "below"
        })

    hover_text = [
        (
            f"<b>{int(row.App_Count):,} apps in this hexagon</b><br>"
            f"Average rating: {row.Average_Rating:.2f}<br>"
            f"Total installs: {row.Total_Installs:,.0f}<br>"
            f"Total reviews: {row.Total_Reviews:,.0f}<br>"
            f"Categories represented: {int(row.Categories)}"
        )
        for row in grouped.itertuples(index=False)
    ]

    fig = go.Figure()
    # Invisible hover targets sit at each hexagon's center; the colored
    # polygons themselves are layout shapes for a true hexbin appearance.
    fig.add_trace(go.Scatter(
        x=grouped["Center_X"],
        y=grouped["Center_Y"],
        mode="markers",
        marker=dict(size=hex_radius * 65, opacity=0.001, color="white"),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False
    ))

    fig.update_layout(
        title="App Install vs Review Density (Log10 Scale)",
        template="plotly_dark",
        height=700,
        shapes=shapes,
        xaxis_title="Log10(Installs)",
        yaxis_title="Log10(Reviews)",
        hovermode="closest",
        margin=dict(l=75, r=45, t=80, b=70)
    )
    fig.update_xaxes(
        range=[x_min - hex_radius, x_max + hex_radius],
        showgrid=True,
        gridcolor="rgba(150,150,150,0.15)"
    )
    fig.update_yaxes(
        range=[y_min - hex_radius, y_max + hex_radius],
        showgrid=True,
        gridcolor="rgba(150,150,150,0.15)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Explicit density legend: the same YlOrRd scale used by the hexagons.
    low_count = int(grouped["App_Count"].min())
    median_count = float(grouped["App_Count"].median())
    high_count = int(grouped["App_Count"].max())

    st.markdown("**Hexagon density legend**")
    legend_col1, legend_col2, legend_col3 = st.columns(3)
    with legend_col1:
        st.markdown(
            f"<div style='background:#fff7bc;padding:10px;border-radius:6px;"
            f"color:#222;text-align:center'><b>Low density</b><br>"
            f"{low_count:,} apps per hexagon</div>",
            unsafe_allow_html=True
        )
    with legend_col2:
        st.markdown(
            f"<div style='background:#fd8d3c;padding:10px;border-radius:6px;"
            f"color:#222;text-align:center'><b>Medium density</b><br>"
            f"Around {median_count:,.0f} apps per hexagon</div>",
            unsafe_allow_html=True
        )
    with legend_col3:
        st.markdown(
            f"<div style='background:#bd0026;padding:10px;border-radius:6px;"
            f"color:white;text-align:center'><b>High density</b><br>"
            f"{high_count:,} apps per hexagon</div>",
            unsafe_allow_html=True
        )
    st.caption(
        "Density is the number of apps assigned to each hexagonal bin. "
        "The legend thresholds are calculated from the currently filtered data."
    )

    busiest_bin = grouped.loc[grouped["App_Count"].idxmax()]
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Eligible Apps", f"{len(data):,}")
    metric2.metric("Hexagonal Bins", f"{len(grouped):,}")
    metric3.metric("Highest-Density Bin", f"{int(busiest_bin['App_Count']):,} apps")

    st.subheader("Density Bin Details")
    detail = grouped.sort_values(
        "App_Count", ascending=False
    ).rename(columns={
        "App_Count": "Apps in Bin",
        "Average_Rating": "Average Rating",
        "Total_Installs": "Total Installs",
        "Total_Reviews": "Total Reviews",
        "Categories": "Categories Represented"
    })
    st.dataframe(
        detail[
            [
                "Apps in Bin", "Average Rating", "Total Installs",
                "Total Reviews", "Categories Represented"
            ]
        ].style.format({
            "Apps in Bin": "{:,.0f}",
            "Average Rating": "{:.2f}",
            "Total Installs": "{:,.0f}",
            "Total Reviews": "{:,.0f}",
            "Categories Represented": "{:,.0f}"
        }),
        use_container_width=True
    )

    st.download_button(
        "Download Hexbin Density Data",
        data=detail.to_csv(index=False).encode("utf-8"),
        file_name="hexbin_density_analysis.csv",
        mime="text/csv"
    )

    st.caption(
        "Methodology: Apps are grouped into hexagonal bins using log10 "
        "installs and log10 reviews. Darker/brighter density colors indicate "
        "more apps in a bin. Log scaling helps display the wide range of "
        "install and review counts. This is an observational association, "
        "not evidence that installs cause reviews."
    )


# --------------------------------------------------
# TASK 6: MONTHLY STREAMGRAPH
# --------------------------------------------------

def show_monthly_streamgraph(apps):
    st.header("Task 6: Monthly Streamgraph")
    st.caption(
        "Explore how the distribution of listed app update activity "
        "changes across categories over time."
    )

    data = apps.copy()
    required = ["App", "Category", "Updated_Date"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        st.error(f"Missing required columns for Task 6: {missing}")
        return

    data["Updated_Date"] = pd.to_datetime(
        data["Updated_Date"], errors="coerce"
    )
    data = data.dropna(subset=["App", "Category", "Updated_Date"]).copy()

    if data.empty:
        st.warning("No valid app update dates are available for the streamgraph.")
        return

    data["Month"] = data["Updated_Date"].dt.to_period("M")
    data["Year"] = data["Updated_Date"].dt.year

    available_years = sorted(data["Year"].unique().astype(int).tolist())
    year_options = ["All years"] + available_years

    control1, control2 = st.columns(2)
    with control1:
        selected_year = st.selectbox(
            "Select year",
            year_options,
            index=0,
            key="task6_year"
        )
    with control2:
        top_n = st.slider(
            "Number of categories",
            min_value=3,
            max_value=min(12, max(3, data["Category"].nunique())),
            value=min(8, max(3, data["Category"].nunique())),
            step=1,
            key="task6_top_n"
        )

    if selected_year != "All years":
        data = data[data["Year"] == selected_year].copy()

    if data.empty:
        st.warning("No app update records are available for this selection.")
        return

    # Select categories with the most listed update records in the period.
    top_categories = (
        data.groupby("Category")["App"]
        .nunique()
        .nlargest(top_n)
        .index.tolist()
    )
    data = data[data["Category"].isin(top_categories)].copy()

    start_month = data["Month"].min()
    end_month = data["Month"].max()
    month_index = pd.period_range(start_month, end_month, freq="M")

    monthly = (
        data.groupby(["Month", "Category"])["App"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=month_index, columns=top_categories, fill_value=0)
    )
    monthly.index = monthly.index.to_timestamp()

    if monthly.empty or monthly.to_numpy().sum() == 0:
        st.warning("There is not enough monthly activity to build a streamgraph.")
        return

    # Center the stacked layers around zero to create a streamgraph-style view.
    values = monthly.to_numpy(dtype=float)
    totals = values.sum(axis=1)
    baseline = -totals / 2.0
    lower = baseline.copy()

    palette = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
        "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
        "#FF97FF", "#FECB52", "#2E91E5", "#E15F99"
    ]

    fig = go.Figure()
    x_values = monthly.index

    for index, category in enumerate(top_categories):
        layer_values = values[:, index]
        upper = lower + layer_values

        # Draw each category as a closed polygon between its lower and upper
        # boundaries. This gives a centered stacked-area / streamgraph layout.
        polygon_x = list(x_values) + list(x_values[::-1])
        polygon_y = list(upper) + list(lower[::-1])

        custom_hover = (
            f"<b>{category}</b><br>"
            "Month: %{x|%b %Y}<br>"
            f"Apps with listed update date: %{{customdata:,.0f}}"
            "<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=polygon_x,
            y=polygon_y,
            mode="lines",
            line=dict(color=palette[index % len(palette)], width=0.7),
            fill="toself",
            fillcolor=palette[index % len(palette)],
            name=str(category),
            customdata=list(layer_values) + list(layer_values[::-1]),
            hovertemplate=custom_hover,
            opacity=0.88
        ))

        lower = upper

    period_label = (
        str(selected_year) if selected_year != "All years"
        else f"{start_month.strftime('%b %Y')} – {end_month.strftime('%b %Y')}"
    )

    fig.update_layout(
        title=f"Monthly App Update Activity by Category — {period_label}",
        template="plotly_dark",
        height=680,
        xaxis_title="Month",
        yaxis_title="Centered app update activity",
        hovermode="x",
        legend_title="App Category",
        margin=dict(l=75, r=45, t=85, b=70)
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(150,150,150,0.15)",
        tickformat="%b %Y"
    )
    fig.update_yaxes(
        zeroline=True,
        zerolinecolor="rgba(220,220,220,0.6)",
        showgrid=True,
        gridcolor="rgba(150,150,150,0.15)"
    )

    st.plotly_chart(fig, use_container_width=True)

    total_records = int(monthly.to_numpy().sum())
    active_months = int((monthly.sum(axis=1) > 0).sum())
    leading_category = monthly.sum().idxmax()
    leading_total = int(monthly[leading_category].sum())

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Apps in Selected Categories", f"{total_records:,}")
    metric2.metric("Months with Activity", f"{active_months:,}")
    metric3.metric("Most Active Category", str(leading_category))

    st.subheader("Monthly Category Activity")
    display_table = monthly.copy()
    display_table.index.name = "Month"
    display_table.columns.name = "Category"
    st.dataframe(
        display_table.style.format("{:,.0f}"),
        use_container_width=True
    )

    st.download_button(
        "Download Monthly Streamgraph Data",
        data=display_table.to_csv().encode("utf-8"),
        file_name="monthly_streamgraph_data.csv",
        mime="text/csv"
    )

    st.caption(
        "Methodology: Each month/category value counts unique apps whose "
        "listed Last Updated date falls in that month. The layers are centered "
        "around zero to create a streamgraph-style stacked area. The source "
        "dataset provides each app's latest listed update date, not a complete "
        "history of every update event; interpret this as the distribution of "
        "listed update dates rather than total historical update events."
    )





def build_executive_pdf(data, insight_cards):
    """Create a compact PDF report for the currently filtered overview data."""
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=18*mm, leftMargin=18*mm,
        topMargin=17*mm, bottomMargin=17*mm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=21, leading=26,
        textColor=colors.HexColor("#243b70"), alignment=TA_LEFT,
        spaceAfter=5*mm
    ))
    styles.add(ParagraphStyle(
        name="ReportSub", parent=styles["Normal"],
        fontSize=9, leading=13, textColor=colors.HexColor("#526078"),
        spaceAfter=5*mm
    ))
    styles.add(ParagraphStyle(
        name="SectionBlue", parent=styles["Heading2"],
        textColor=colors.HexColor("#243b70"), fontSize=13,
        spaceBefore=4*mm, spaceAfter=2*mm
    ))
    styles.add(ParagraphStyle(
        name="SmallBody", parent=styles["BodyText"],
        fontSize=8.5, leading=12
    ))

    total_apps = len(data)
    avg_rating = data["Rating"].mean() if "Rating" in data else np.nan
    installs = data["Installs"].sum(min_count=1) if "Installs" in data else np.nan
    reviews = data["Reviews"].sum(min_count=1) if "Reviews" in data else np.nan
    categories = data["Category"].nunique() if "Category" in data else 0
    free_count = int(data["Type"].eq("Free").sum()) if "Type" in data else 0
    paid_count = int(data["Type"].eq("Paid").sum()) if "Type" in data else 0

    story = [
        Paragraph("Google Play Store Analytics", styles["ReportTitle"]),
        Paragraph(
            f"Executive summary | Generated {datetime.now().strftime('%d %b %Y, %I:%M %p')} "
            "| Based on the currently filtered dashboard dataset",
            styles["ReportSub"]
        ),
        Paragraph("Key Performance Indicators", styles["SectionBlue"])
    ]

    kpi_rows = [
        ["Metric", "Value", "Metric", "Value"],
        ["Apps", f"{total_apps:,}", "Categories", f"{categories:,}"],
        ["Average rating", f"{avg_rating:.2f}/5" if pd.notna(avg_rating) else "N/A",
         "Free / Paid", f"{free_count:,} / {paid_count:,}"],
        ["Listed installs", f"{installs:,.0f}" if pd.notna(installs) else "N/A",
         "Reviews", f"{reviews:,.0f}" if pd.notna(reviews) else "N/A"]
    ]
    kpi_table = Table(kpi_rows, colWidths=[36*mm, 48*mm, 36*mm, 48*mm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243b70")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#eef2f8")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#1f2937")),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#cbd5e1")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    story += [kpi_table, Paragraph("Automated Insights", styles["SectionBlue"])]

    if insight_cards:
        for heading, body in insight_cards:
            story.append(KeepTogether([
                Paragraph(f"<b>{heading}</b>", styles["SmallBody"]),
                Paragraph(body.replace("&", "&amp;"), styles["SmallBody"]),
                Spacer(1, 2*mm)
            ]))
    else:
        story.append(Paragraph("No insights available for the selected filters.", styles["SmallBody"]))

    story.append(Paragraph("Top Categories by App Count", styles["SectionBlue"]))
    if "Category" in data.columns:
        counts = data["Category"].value_counts().head(10)
        rows = [["Category", "App count", "Average rating"]]
        for category, count in counts.items():
            subset = data[data["Category"] == category]
            rating = subset["Rating"].mean()
            rows.append([str(category), f"{int(count):,}", f"{rating:.2f}" if pd.notna(rating) else "N/A"])
        table = Table(rows, colWidths=[75*mm, 35*mm, 45*mm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243b70")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f8")]),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#cbd5e1")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 6)
        ]))
        story.append(table)

    story += [
        Spacer(1, 5*mm),
        Paragraph(
            "Note: Installs and reviews are aggregated values from the supplied dataset. "
            "The review-to-install ratio is a dataset-level proxy, not a user-level engagement measure.",
            styles["SmallBody"]
        )
    ]
    doc.build(story)
    return buffer.getvalue()


# --------------------------------------------------
# OVERVIEW: EXECUTIVE DASHBOARD
# --------------------------------------------------

def show_overview(apps):
    st.subheader("Executive Overview")
    st.markdown(
        '<div class="section-note">A high-level snapshot of the uploaded Google Play Store dataset. '
        'Use the task navigator to explore the six detailed analyses.</div>',
        unsafe_allow_html=True
    )

    data = apps.copy()

    # Overview-only interactive filters. Detailed task pages continue to use
    # the full cleaned dataset, so existing task calculations are untouched.
    st.markdown("#### Dashboard Filters")
    filter_a, filter_b, filter_c, filter_d = st.columns([1.25, 1, 1, 1])
    with filter_a:
        category_options = ["All categories"] + sorted(
            data["Category"].dropna().astype(str).unique().tolist()
        )
        selected_overview_category = st.selectbox(
            "App category",
            category_options,
            key="overview_category_filter"
        )
    with filter_b:
        type_options = ["All types"] + sorted(
            data["Type"].dropna().astype(str).unique().tolist()
        ) if "Type" in data.columns else ["All types"]
        selected_overview_type = st.selectbox(
            "App type",
            type_options,
            key="overview_type_filter"
        )
    with filter_c:
        selected_min_rating = st.slider(
            "Minimum rating",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.5,
            key="overview_min_rating"
        )
    with filter_d:
        selected_min_installs = st.selectbox(
            "Minimum installs",
            [0, 1000, 10000, 100000, 1000000],
            format_func=lambda value: f"{value:,}+",
            key="overview_min_installs"
        )

    if selected_overview_category != "All categories":
        data = data[data["Category"].astype(str) == selected_overview_category]
    if selected_overview_type != "All types" and "Type" in data.columns:
        data = data[data["Type"].astype(str) == selected_overview_type]
    data = data[
        (data["Rating"].isna() | data["Rating"].ge(selected_min_rating))
        & (data["Installs"].isna() | data["Installs"].ge(selected_min_installs))
    ].copy()

    if data.empty:
        st.warning("No apps match these filters. Try lowering the rating or install threshold.")
        return

    reset_col, info_col = st.columns([1, 5])
    with reset_col:
        if st.button("↺ Reset filters", key="overview_reset_filters"):
            for state_key in [
                "overview_category_filter",
                "overview_type_filter",
                "overview_min_rating",
                "overview_min_installs",
            ]:
                st.session_state.pop(state_key, None)
            st.rerun()
    with info_col:
        st.caption(f"Showing {len(data):,} of {len(apps):,} apps after applying overview filters.")
    total_apps = len(data)
    categories = data["Category"].nunique()
    avg_rating = data["Rating"].mean()
    total_installs = data["Installs"].sum(min_count=1)
    total_reviews = data["Reviews"].sum(min_count=1)
    free_count = int(data["Type"].eq("Free").sum()) if "Type" in data else 0
    paid_count = int(data["Type"].eq("Paid").sum()) if "Type" in data else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Apps in Dataset", f"{total_apps:,}")
    k2.metric("App Categories", f"{categories:,}")
    k3.metric("Average Rating", f"{avg_rating:.2f} / 5" if pd.notna(avg_rating) else "N/A")

    k4, k5, k6 = st.columns(3)
    k4.metric("Total Listed Installs", f"{total_installs:,.0f}" if pd.notna(total_installs) else "N/A")
    k5.metric("Total Reviews", f"{total_reviews:,.0f}" if pd.notna(total_reviews) else "N/A")
    k6.metric("Free / Paid Apps", f"{free_count:,} / {paid_count:,}")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Top 10 Categories by App Count")
        category_counts = data["Category"].value_counts().head(10).sort_values()
        fig = go.Figure(go.Bar(
            x=category_counts.values, y=category_counts.index,
            orientation="h", marker=dict(
                color=category_counts.values, colorscale="Viridis",
                line=dict(color="rgba(255,255,255,.25)", width=.5)
            ),
            hovertemplate="<b>%{y}</b><br>Apps: %{x:,}<extra></extra>"
        ))
        fig.update_layout(
            template="plotly_dark", height=430,
            margin=dict(l=10, r=15, t=15, b=35),
            xaxis_title="Number of apps", yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Free vs Paid App Mix")
        type_counts = data["Type"].value_counts() if "Type" in data else pd.Series(dtype=int)
        if not type_counts.empty:
            fig = go.Figure(go.Pie(
                labels=type_counts.index.astype(str), values=type_counts.values,
                hole=.62, textinfo="label+percent",
                marker=dict(line=dict(color="#111827", width=2)),
                hovertemplate="%{label}: %{value:,} apps (%{percent})<extra></extra>"
            ))
            fig.update_layout(
                template="plotly_dark", height=430,
                margin=dict(l=10, r=10, t=15, b=15),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-.05)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("App Type information is not available in this dataset.")

    st.markdown("#### Rating Distribution")
    rating_data = data["Rating"].dropna()
    if not rating_data.empty:
        fig = go.Figure(go.Histogram(
            x=rating_data, nbinsx=20,
            marker=dict(color="#818cf8", line=dict(color="#c7d2fe", width=.6)),
            hovertemplate="Rating range: %{x}<br>Apps: %{y:,}<extra></extra>"
        ))
        fig.update_layout(
            template="plotly_dark", height=300,
            xaxis_title="App rating", yaxis_title="Number of apps",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=15, t=15, b=45)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Dataset Preview")
    preview_cols = [c for c in ["App", "Category", "Rating", "Installs", "Reviews", "Type", "Price"] if c in data.columns]
    st.dataframe(data[preview_cols].head(12), use_container_width=True, hide_index=True)


    # Explainable, data-driven insights (no external AI/API required).
    st.markdown("---")
    st.subheader("✨ Automated Insights")
    st.caption(
        "Insights are generated from the uploaded dataset using transparent "
        "summary rules. They update whenever the dataset changes."
    )

    insight_cards = []
    if not data.empty:
        category_apps = data.groupby("Category").size().sort_values(ascending=False)
        if not category_apps.empty:
            top_cat = str(category_apps.index[0])
            top_count = int(category_apps.iloc[0])
            share = top_count / len(data) * 100
            insight_cards.append((
                "🏆 Largest app category",
                f"{top_cat} contains {top_count:,} apps ({share:.1f}% of the uploaded dataset)."
            ))

        category_rating = (
            data.dropna(subset=["Rating"])
            .groupby("Category")["Rating"]
            .agg(["mean", "count"])
        )
        # Avoid spotlighting categories with fewer than 5 rated apps.
        stable_rating = category_rating[category_rating["count"] >= 5]
        if not stable_rating.empty:
            best_rating_cat = stable_rating["mean"].idxmax()
            best_rating = float(stable_rating.loc[best_rating_cat, "mean"])
            insight_cards.append((
                "⭐ Highest average category rating",
                f"{best_rating_cat} has the highest average rating ({best_rating:.2f}/5) "
                f"among categories with at least five rated apps."
            ))

        category_installs = (
            data.dropna(subset=["Installs"])
            .groupby("Category")["Installs"]
            .sum()
            .sort_values(ascending=False)
        )
        if not category_installs.empty:
            install_cat = str(category_installs.index[0])
            install_total = float(category_installs.iloc[0])
            insight_cards.append((
                "📈 Install leader",
                f"{install_cat} leads in summed listed installs with {install_total:,.0f}."
            ))

        if free_count + paid_count > 0:
            paid_share = paid_count / (free_count + paid_count) * 100
            insight_cards.append((
                "💳 Monetization mix",
                f"Paid apps make up {paid_share:.2f}% of records classified as Free or Paid "
                f"({paid_count:,} paid vs {free_count:,} free)."
            ))

        valid_engagement = data[
            data["Installs"].gt(0) & data["Reviews"].notna()
        ].copy()
        if not valid_engagement.empty:
            engagement = valid_engagement["Reviews"].sum() / valid_engagement["Installs"].sum() * 100
            insight_cards.append((
                "💬 Review-to-install ratio",
                f"The dataset-wide ratio is approximately {engagement:.3f}% "
                "using total reviews divided by total listed installs. This is a proxy, not a user-level engagement measure."
            ))

    if insight_cards:
        for offset in range(0, len(insight_cards), 2):
            cols = st.columns(2)
            for col, item in zip(cols, insight_cards[offset:offset + 2]):
                with col:
                    st.markdown(
                        f'<div style="min-height:112px;padding:16px 18px;'
                        f'border:1px solid rgba(129,140,248,.24);border-radius:13px;'
                        f'background:linear-gradient(145deg,rgba(30,41,67,.82),rgba(17,27,47,.88));'
                        f'margin-bottom:12px;">'
                        f'<div style="font-weight:700;color:#c7d2fe;margin-bottom:8px;">{item[0]}</div>'
                        f'<div style="color:#e2e8f0;line-height:1.5;">{item[1]}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
    else:
        st.info("Not enough valid data to generate insights.")


    st.markdown("---")
    st.subheader("📄 Executive Report")
    st.caption(
        "Generate a PDF containing the current filtered KPIs, automated insights "
        "and top category summary."
    )
    try:
        pdf_bytes = build_executive_pdf(data, insight_cards)
        st.download_button(
            "⬇️ Download Executive PDF Report",
            data=pdf_bytes,
            file_name="google_play_store_executive_report.pdf",
            mime="application/pdf",
            key="download_executive_pdf"
        )
    except ImportError:
        st.warning(
            "PDF export needs ReportLab. Install it with: pip install reportlab"
        )
    except Exception as exc:
        st.error(f"Could not generate the PDF report: {exc}")


# --------------------------------------------------
# 12. MAIN DASHBOARD
# --------------------------------------------------

with st.sidebar:
    st.markdown("## PLAYSTORE ANALYTICS")
    st.caption("Business intelligence workspace")
    st.markdown("---")
    apps_file = st.file_uploader(
        "Upload app dataset",
        type="csv",
        help="Upload googleplaystore.csv to activate the dashboard."
    )
    reviews_file = st.file_uploader(
        "Optional user reviews CSV",
        type="csv"
    )

if apps_file is None:
    st.markdown("### Welcome to your analytics workspace")
    st.markdown(
        '<div class="section-note">Upload <b>googleplaystore.csv</b> from the sidebar to unlock '
        'the executive overview and all six interactive analysis modules.</div>',
        unsafe_allow_html=True
    )
    st.info("Your dataset stays in this Streamlit session and is processed for visualization.")
    st.stop()

try:
    apps = clean_apps(apps_file)
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

now = datetime.now(ZoneInfo("Asia/Kolkata"))

with st.sidebar:
    st.markdown("---")
    st.markdown("### Navigation")
    navigation_options = ["Overview"] + list(WINDOWS.keys())
    selected = st.radio(
        "Choose a dashboard page",
        navigation_options,
        key="dashboard_navigation"
    )
    st.markdown("---")
    st.caption(f"Dataset rows: {len(apps):,}")
    st.caption(f"Current IST: {now.strftime('%d %b %Y, %I:%M %p')}")
    preview = st.checkbox(
        "Development preview (testing only)",
        value=False,
        help="Temporarily bypasses task time restrictions. Keep disabled for final evaluation."
    )

if selected == "Overview":
    show_overview(apps)
    st.caption("Select any analysis module from the sidebar to explore the detailed visualizations.")
    st.stop()

start, end = WINDOWS[selected]
if not (start <= now.hour < end) and not preview:
    st.warning(
        f"This visualization is restricted to {start:02d}:00–{end:02d}:00 IST. "
        "It is hidden outside that window. Enable Development preview only while testing."
    )
    st.stop()

task_titles = {
    "1 · Clustered Heatmap": "Category Performance Intelligence",
    "2 · Hierarchical Sunburst": "Global App Ecosystem Explorer",
    "3 · Free vs Paid Radar": "Free vs Paid Benchmarking",
    "4 · Monthly Calendar Heatmap": "App Update Calendar",
    "5 · Hexbin Density": "Install & Review Density",
    "6 · Monthly Streamgraph": "Category Activity Over Time",
}
st.markdown(
    f'<div class="section-note"><b>{task_titles.get(selected, selected)}</b> '
    f'&nbsp; | &nbsp; Data rows available: {len(apps):,}</div>',
    unsafe_allow_html=True
)

if selected == "1 · Clustered Heatmap":
    show_clustered_heatmap(apps)
elif selected == "2 · Hierarchical Sunburst":
    show_sunburst()
elif selected == "3 · Free vs Paid Radar":
    show_free_paid_radar(apps)
elif selected == "4 · Monthly Calendar Heatmap":
    show_calendar_heatmap(apps)
elif selected == "5 · Hexbin Density":
    show_hexbin_density(apps)
elif selected == "6 · Monthly Streamgraph":
    show_monthly_streamgraph(apps)
else:
    st.info("Select a dashboard module from the sidebar.")
