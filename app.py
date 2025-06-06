import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from datetime import datetime
from scipy import stats

OPENROUTER_URL = "https://openrouter.ai/api/frontend/models/find"


def get_tokens_per_dollar(model):
    dollars = (
        model["completion"] * model["total_completion_tokens"]
        + model["prompt"] * model["total_prompt_tokens"]
        # Use completion if internal reasoning cost is 0
        + (
            model["internal_reasoning"]
            if model["internal_reasoning"] > 0
            else model["completion"]
        )
        * model["total_native_tokens_reasoning"]
        + model["request"] * model["count"]
    )
    return model["total_tokens"] / dollars


def calculate_model_scores(df):
    # Work with log values since we're using log scales
    log_tokens_per_dollar = np.log10(df["tokens_per_dollar"])
    log_total_tokens = np.log10(df["total_tokens"])

    # Remove any infinite or NaN values
    mask = np.isfinite(log_tokens_per_dollar) & np.isfinite(log_total_tokens)
    log_tokens_per_dollar_clean = log_tokens_per_dollar[mask]
    log_total_tokens_clean = log_total_tokens[mask]

    # Fit linear regression using scipy.stats.linregress
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_tokens_per_dollar_clean, log_total_tokens_clean
    )

    # Predict for all data points
    log_predicted = slope * log_tokens_per_dollar + intercept

    # Calculate residuals (actual - predicted in log space)
    residuals = log_total_tokens - log_predicted

    # Convert to z-scores (standardize residuals)
    residual_mean = np.nanmean(residuals)
    residual_std = np.nanstd(residuals)
    z_scores = (residuals - residual_mean) / residual_std

    # Create results dataframe
    results_df = df.copy()
    results_df["predicted_total_tokens"] = 10**log_predicted
    results_df["score"] = z_scores
    results_df["log_residual"] = residuals

    return results_df


def find_pareto_frontier(df):
    # Create a copy for manipulation
    data = df[["tokens_per_dollar", "total_tokens"]].copy()

    # Find Pareto optimal points
    pareto_mask = np.ones(len(data), dtype=bool)

    for i in range(len(data)):
        if pareto_mask[i]:
            # Check if any other point dominates this one
            # A point dominates if it has both higher tokens_per_dollar AND higher total_tokens
            dominated_mask = (
                (data["tokens_per_dollar"] >= data.iloc[i]["tokens_per_dollar"])
                & (data["total_tokens"] >= data.iloc[i]["total_tokens"])
                & (
                    (data["tokens_per_dollar"] > data.iloc[i]["tokens_per_dollar"])
                    | (data["total_tokens"] > data.iloc[i]["total_tokens"])
                )
            )

            if dominated_mask.any():
                pareto_mask[i] = False

    pareto_df = df[pareto_mask].copy()

    # Sort by tokens_per_dollar for proper line connection
    pareto_df = pareto_df.sort_values("tokens_per_dollar")

    return pareto_df


@st.cache_data
def load_data():
    response = requests.get(OPENROUTER_URL)
    data = response.json()

    models = []
    for model in data["data"]["models"]:
        try:
            slug = model["endpoint"].get("model_variant_permaslug", model["slug"])
            models.append(
                {
                    "name": model["name"],
                    "is_free": model["endpoint"]["is_free"],
                    **model["endpoint"]["pricing"],
                    **data["data"]["analytics"][slug],
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(
        models,
    )

    # Drop free models
    df = df[~df["is_free"]]
    df.drop(columns=["is_free"], inplace=True)

    # Index by organization and model names
    df[["organization", "model"]] = (
        df["variant_permaslug"]
        .apply(lambda x: x.split("/"))
        .apply(lambda x: pd.Series(x))
    )
    df.set_index(["organization", "model"], inplace=True)
    df.sort_index(inplace=True)

    # Convert pricing columns to numeric
    pricing_cols = [
        "prompt",
        "completion",
        "image",
        "request",
        "web_search",
        "internal_reasoning",
    ]
    df[pricing_cols] = df[pricing_cols].apply(pd.to_numeric, errors="coerce")

    # Calculate total tokens (per dollar)
    df["total_tokens"] = df[
        [
            "total_completion_tokens",
            "total_prompt_tokens",
            "total_native_tokens_reasoning",
        ]
    ].sum(axis=1)
    df["tokens_per_dollar"] = df.apply(get_tokens_per_dollar, axis=1)

    # Return both dataframe and timestamp
    return df, datetime.now()


# Load data and timestamp
df, last_updated = load_data()

# Calculate model scores
scored_df = calculate_model_scores(df)

st.markdown("# OpenRouter Pareto")

# Add multiselect for organizations
organizations = st.multiselect(
    "organizations",
    options=sorted(scored_df.index.get_level_values(0).unique()),
    default=[
        "anthropic",
        "deepseek",
        "google",
        "meta-llama",
        "mistralai",
        "openai",
        "qwen",
        "x-ai",
    ],
)

if organizations:
    filtered_df = scored_df[scored_df.index.get_level_values(0).isin(organizations)]
else:
    filtered_df = scored_df

# Find Pareto frontier
pareto_df = find_pareto_frontier(filtered_df)

# Create columns for the header area
col1, col2 = st.columns([3, 1])

with col1:
    st.write(f"**Last updated:** {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Create base figure with log scales
fig = go.Figure()

# Add Pareto line
if len(pareto_df) > 1:
    fig.add_trace(
        go.Scatter(
            x=pareto_df["tokens_per_dollar"],
            y=pareto_df["total_tokens"],
            mode="lines",
            name="frontier",
            line=dict(color="white", width=1),
            customdata=pareto_df["score"],
        )
    )

# Add scatter plot points
for org in filtered_df.index.get_level_values(0).unique():
    org_data = filtered_df[filtered_df.index.get_level_values(0) == org]
    fig.add_trace(
        go.Scatter(
            x=org_data["tokens_per_dollar"],
            y=org_data["total_tokens"],
            mode="markers",
            name=org,
            customdata=org_data["score"],
            text=[model for _, model in org_data.index],
            hovertemplate="<b>%{text}</b><br>"
            + "tokens/$: %{x:,.0f}<br>"
            + "total tokens: %{y:,.0f}<br>"
            + "score: %{customdata:.2f}<extra></extra>",
        )
    )

# Set log scales and labels
fig.update_xaxes(type="log", title="tokens/$")
fig.update_yaxes(type="log", title="total tokens")
fig.update_layout(showlegend=True)

st.plotly_chart(fig, use_container_width=True)

# Prepare display dataframe
display_df = filtered_df.reset_index()
display_df = display_df[
    [
        "organization",
        "model",
        "tokens_per_dollar",
        "total_tokens",
        "score",
    ]
]

# add pareto indicator if pareto_df contains the model
display_df["is_pareto"] = display_df.apply(
    lambda row: row["organization"] in pareto_df.index.get_level_values(0)
    and row["model"] in pareto_df.index.get_level_values(1),
    axis=1,
)

# Sort by score (best first)
display_df = display_df.sort_values("score", ascending=False)

# Add Pareto indicator column for display
display_df["pareto"] = display_df["is_pareto"].apply(lambda x: "✨" if x else "")

# Reorder columns
display_df = display_df[
    ["pareto", "organization", "model", "tokens_per_dollar", "total_tokens", "score"]
]

st.dataframe(display_df, use_container_width=True, hide_index=True)
