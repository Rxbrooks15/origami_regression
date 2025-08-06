#Gami model
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import re, os, csv, requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime

# --- Cache Data ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Rxbrooks15/origami_regression/refs/heads/main/Combined_CNN_BERT.csv"
    return pd.read_csv(url)

df = load_data()

# --- Clean Data for GAMI ---
df_clean = df.dropna(subset=["Keyword_Score", "Edge_Count", "Difficulty_Numeric", "time_minutes"]).copy()
df_clean = df_clean[df_clean["time_minutes"] > 0]
df_clean["GAMI"] = df_clean["Keyword_Score"] * df_clean["Edge_Count"] * df_clean["Difficulty_Numeric"]
df_clean["spacer"] = ""


# --- Features & Models ---
X = df_clean[["time_minutes"]].values
y = df_clean["GAMI"].values
x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

lin_model = LinearRegression().fit(X, y)
y_lin, r2_lin = lin_model.predict(x_range), r2_score(y, lin_model.predict(X))

X_log = np.log(X)
log_model = LinearRegression().fit(X_log, y)
y_log, r2_log = log_model.predict(np.log(x_range)), r2_score(y, log_model.predict(X_log))

dt_model = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X, y)
y_dt, r2_dt = dt_model.predict(x_range), r2_score(y, dt_model.predict(X))

rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42).fit(X, y)
y_rf, r2_rf = rf_model.predict(x_range), r2_score(y, rf_model.predict(X))

# --- Sidebar Search & Model Choice ---
search_query = st.sidebar.text_input("🔎 Search Model Name")
model_choice = st.sidebar.radio("Choose Regression Model:",
                                ("🌲 Random Forest", "Linear", "Logarithmic", "Decision Tree"),
                                index=0)

highlight_name = None
if search_query:
    match = df[df["Name"].str.contains(search_query, case=False, na=False)]
    if not match.empty:
        selected = match.iloc[0]
        highlight_name = selected["Name"]
        st.sidebar.image(selected.get("Image_github") or selected.get("Image"),
                         caption=selected["Name"], width=220)
        st.sidebar.write(f"**Creator:** {selected.get('Creator', 'Unknown')}")
        st.sidebar.write(f"**Difficulty:** {selected.get('Difficulty', 'Unknown')}")
        st.sidebar.write(f"**Description:** {selected.get('Description', '')[:150]}...")

# --- Difficulty Colors ---
difficulty_colors = {1: "blue", 2: "lightblue", 3: "orange", 4: "darkorange", 5: "red"}

# 🔹 Function to add highlight marker
def add_highlight(fig, df_target, y_col):
    if highlight_name:
        match = df_target[df_target["Name"].str.lower() == highlight_name.lower()]
        if not match.empty:
            x_val = match["time_minutes"].values[0]
            y_val = match[y_col].values[0]
            name_val = match["Name"].values[0]

            # Outer circle
            fig.add_trace(go.Scatter(
                x=[x_val], y=[y_val],
                mode='markers',
                marker=dict(size=14, color='rgba(255,0,0,0)',
                            line=dict(color='red', width=3)),
                showlegend=False,
                hoverinfo='skip'
            ))
            # Inner dot with label
            fig.add_trace(go.Scatter(
                x=[x_val], y=[y_val],
                mode='markers+text',
                text=[name_val],
                textposition="top center",
                marker=dict(size=6, color='red'),
                textfont=dict(color='red', size=14),
                name="🔴 Highlighted"
            ))
    return fig

# --- GAMI Plot ---
fig_gami = px.scatter(
    df_clean,
    x="time_minutes",
    y="GAMI",
    color=df_clean["Difficulty_Numeric"].astype(str),
    color_discrete_map={str(k): v for k, v in difficulty_colors.items()},
    custom_data=['Name', 'Keyword_Score', 'Edge_Count', 'Difficulty_Numeric', 'GAMI', 'Description'],
    hover_data={
        "Name": True,
        "GAMI": True,
        "time_minutes": True,
        "Difficulty_Numeric": True,
        "spacer": True,  # <-- Spacer here
        "Edge_Count": True,
        "Keyword_Score": True,
        "Description": True
    },
    labels={
        "Name": "🔖Origami Name",
        "GAMI": "💲 GAMI Score"
        "time_minutes": "🕒 Folding Time",
        
        "Difficulty_Numeric": "😓Difficulty",
        "spacer": " ",  # This line creates the visual gap
        "Edge_Count": "🧩 Edge Count",
        "Keyword_Score": "🔑 Keyword Score",
        "Description": "📜 Description"
    },
    title=f"💲 GAMI vs 🕒 Folding Time | {model_choice}"
)


if model_choice == "Linear":
    fig_gami.add_trace(go.Scatter(x=x_range.flatten(), y=y_lin, mode="lines",
                                  name=f"Linear (R²={r2_lin:.3f})", line=dict(color="blue")))
elif model_choice == "Logarithmic":
    fig_gami.add_trace(go.Scatter(x=x_range.flatten(), y=y_log, mode="lines",
                                  name=f"Logarithmic (R²={r2_log:.3f})", line=dict(color="purple")))
elif model_choice == "Decision Tree":
    fig_gami.add_trace(go.Scatter(x=x_range.flatten(), y=y_dt, mode="lines",
                                  name=f"Decision Tree (R²={r2_dt:.3f})", line=dict(color="red")))
else:
    fig_gami.add_trace(go.Scatter(x=x_range.flatten(), y=y_rf, mode="lines",
                                  name=f"🌲 Random Forest (R²={r2_rf:.3f})", line=dict(color="green", width=4)))

fig_gami = add_highlight(fig_gami, df_clean, "GAMI")
st.plotly_chart(fig_gami, use_container_width=True)
st.markdown("Model experiences overfitting issues with decision trees, try the logarithm regression for a smoother model")
st.markdown("Check the plot for names of models, you might be intersted in, and use the search bar with the origami's model name")

