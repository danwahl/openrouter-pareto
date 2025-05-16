import requests
import pandas as pd
import plotly.express as px
import streamlit as st

OPENROUTER_URL = "https://openrouter.ai/api/frontend/models/find"


def get_tokens_per_dollar(model):
    tokens = model[
        [
            "total_completion_tokens",
            "total_prompt_tokens",
            "total_native_tokens_reasoning",
        ]
    ].sum()
    dollars = (
        model["completion"] * model["total_completion_tokens"]
        + model["prompt"] * model["total_prompt_tokens"]
        + (
            model["internal_reasoning"]
            if model["internal_reasoning"] > 0
            else model["completion"]
        )
        * model["total_native_tokens_reasoning"]
        + model["request"] * model["count"]
    )
    return tokens / dollars


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

    # Calculate tokens per dollar
    df["tokens_per_dollar"] = df.apply(get_tokens_per_dollar, axis=1)

    return df


df = load_data()

st.markdown("# OpenRouter Pareto")

# Add multiselect for organizations
organizations = st.multiselect(
    "Organizations",
    options=sorted(df.index.get_level_values(0).unique()),
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
    filtered_df = df[df.index.get_level_values(0).isin(organizations)]
else:
    filtered_df = df


fig = px.scatter(
    filtered_df,
    x="tokens_per_dollar",
    y="count",
    color=filtered_df.index.get_level_values(0),
    hover_name=filtered_df.index.get_level_values(1),
    hover_data=["tokens_per_dollar", "count"],
    log_x=True,
    log_y=True,
)

st.plotly_chart(fig)
