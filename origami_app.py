import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import urljoin

CSV_PATH = "origami_scrape_final.csv"

# --- Sidebar and Search Query ---
st.sidebar.header("Preview Origami Models")
search_query = st.sidebar.text_input("🔎 Search Model Name")

# --- Scraping functions ---
def scrape_model_detail(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        name = soup.select_one("h1").text.strip()
        image_tag = soup.select_one(".single-model__image img")
        image = urljoin(url, image_tag["src"].strip()) if image_tag else ""
        desc_tag = soup.select_one(".single-model__content p")
        description = desc_tag.text.strip() if desc_tag else "No description available"
        creator_tag = soup.select_one(".single-model__content__creator a")
        creator = creator_tag.text.strip() if creator_tag else "Unknown"

        difficulty = ""
        meta_items = soup.select(".single-model__content__meta__item")
        for item in meta_items:
            text = item.get_text(" ", strip=True)
            match = re.search(r"\b(easy|moderate|intermediate|hard|complex)\b", text, re.I)
            if match:
                difficulty = match.group(1).capitalize()
                break

        time_tag = soup.select_one(
            ".single-model__content__meta__item:nth-child(5) .single-model__content__meta__item__description"
        )
        time_str = time_tag.text.strip() if time_tag else ""

        return {
            "Image": image,
            "Name": name,
            "Creator": creator,
            "Description": description,
            "Difficulty": difficulty,
            "Time": time_str
        }
    except Exception as e:
        st.error(f"Failed to scrape {url}: {e}")
        return None

def get_first_model_url():
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://origami-database.com/models/page/1/"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    first_link = soup.select_one(".model-card:nth-child(1) .model-card__header a")
    if not first_link:
        return None
    return urljoin("https://origami-database.com", first_link["href"])

# --- Utility ---
def convert_to_minutes(time_str):
    if pd.isna(time_str): return 0
    time_str = time_str.lower().replace("hours", "hr").replace("hour", "hr")
    time_str = time_str.replace("minutes", "min").replace("minute", "min").replace(".", "").strip()
    h = int(re.search(r'(\d+)\s*hr', time_str).group(1)) if re.search(r'(\d+)\s*hr', time_str) else 0
    m = int(re.search(r'(\d+)\s*min', time_str).group(1)) if re.search(r'(\d+)\s*min', time_str) else 0
    return h * 60 + m

# --- Main plot function ---
def process_and_plot(df, highlight_name=None):
    df['time_minutes'] = df['Time'].apply(convert_to_minutes)
    df['Difficulty'] = df['Difficulty'].str.strip().str.lower()
    difficulty_map = {'easy': 1, 'moderate': 2, 'intermediate': 3, 'hard': 4, 'complex': 5}
    df['Difficulty_Numeric'] = df['Difficulty'].map(difficulty_map).fillna(1)

    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(""))
    nmf = NMF(n_components=7, random_state=42)
    topic_matrix = nmf.fit_transform(tfidf_matrix)
    df['Dominant_Topic'] = topic_matrix.argmax(axis=1)

    topic_weights = {6: 2.13, 2: 2.57, 3: 2.89, 1: 2.92, 0: 2.95, 5: 3.08, 4: 3.13}
    df['Topic_Weighted_Difficulty'] = df['Dominant_Topic'].map(topic_weights)

    df['Name_Length'] = df['Name'].astype(str).apply(len)
    df['Description_Length'] = df['Description'].astype(str).apply(len)
    df['Name_Score'] = df['Name_Length'] / df.groupby('Difficulty_Numeric')['Name_Length'].transform('mean')
    df['Description_Score'] = df['Description_Length'] / df.groupby('Difficulty_Numeric')['Description_Length'].transform('mean')

    df['Complexity_Score'] = (
        df['Difficulty_Numeric'] + df['Topic_Weighted_Difficulty'] + df['Name_Score'] + df['Description_Score']
    ) / 4

    df["Image_github"] = df["Image"].apply(
        lambda url: "https://raw.githubusercontent.com/rxbrooks15/origami_regression/main/folder/" + os.path.basename(str(url))
    )

    X = df[['time_minutes']].values
    y = df['Complexity_Score'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_degree, best_r2, best_model, best_poly = 1, -np.inf, None, None
    for d in range(1, 7):
        poly = PolynomialFeatures(degree=d, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        model = LinearRegression().fit(X_train_poly, y_train)
        r2 = r2_score(y_test, model.predict(X_test_poly))
        if r2 > best_r2:
            best_degree, best_r2, best_model, best_poly = d, r2, model, poly

    X_full = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_pred = best_model.predict(best_poly.transform(X_full))

    fig = px.scatter(
        df, x='time_minutes', y='Complexity_Score',
        color='Topic_Weighted_Difficulty',
        custom_data=[
            'Image_github', 'Name', 'Creator', 'time_minutes', 'Complexity_Score',
            'Description', 'Dominant_Topic', 'Topic_Weighted_Difficulty',
            'Name_Score', 'Description_Score'
        ],
        title=f"Polynomial Fit (Degree {best_degree}) | R²: {best_r2:.3f}"
    )
    fig.update_traces(
    hovertemplate="""
    🏷️ <b>%{customdata[1]}</b><br>
    🧑‍🎨 <b>%{customdata[2]}</b><br>
    ⏱️ <b>%{customdata[3]:.1f}</b> minutes<br>
    📊 <b>Complexity:</b> %{customdata[4]:.2f}<br>
    <b>Topic Group:</b> %{customdata[6]}<br>
    <b>Topic Weight:</b> %{customdata[7]:.2f}<br>
    <b>Name Score:</b> %{customdata[8]:.2f}<br>
    <b>Description Score:</b> %{customdata[9]:.2f}<br>
    📃<b>Description:</b> %{customdata[5]}<br>
    <extra></extra>
    """,
    marker=dict(size=9, opacity=0.8),
    hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial")
)
    fig.add_trace(go.Scatter(x=X_full.flatten(), y=y_pred, mode='lines', name='Fit', line=dict(color='black')))

    if highlight_name:
        match = df[df["Name"].str.lower() == highlight_name.lower()]
        if not match.empty:
            fig.add_trace(go.Scatter(
                x=match["time_minutes"],
                y=match["Complexity_Score"],
                mode='markers+text',
                name='🟢 Highlighted',
                text=match["Name"],
                textposition="top center",
                marker=dict(color='green', size=14, symbol='circle-open-dot')
            ))

    st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"### Total Observations: {df.shape[0]-1}")
    st.markdown("### Most difficult models:")
    st.dataframe(
        df.sort_values('Complexity_Score', ascending=False)
          .head(5)[['Name', 'Difficulty', 'Complexity_Score']],
        use_container_width=True
    )
    st.markdown("### Most recent models:")
    st.dataframe(
        df.head(5)[['Name', 'Difficulty', 'Complexity_Score']],
        use_container_width=True
    )
    st.markdown("### Validation R² Scores for Polynomial Degrees 1 to 6:")
    for degree, r2 in r2_scores.items():
        st.write(f"Degree {degree}: R² = {r2:.4f}")
    st.markdown(f"### Best Polynomial Degree: {best_degree} with Validation R²: {best_r2_val:.4f}")


# --- Streamlit UI ---
st.title("📐 Origami Model Complexity Tracker")

# Load CSV
df = pd.read_csv(CSV_PATH)

# Sidebar preview logic
highlight_name = None
if search_query:
    match = df[df["Name"].str.contains(search_query, case=False, na=False)]
    if not match.empty:
        selected = match.iloc[0]
        highlight_name = selected["Name"]
        st.sidebar.image(selected.get("Image_github") or selected.get("Image"), caption=selected["Name"], width=220)
        st.sidebar.write(f"**Creator:** {selected.get('Creator')}")
        st.sidebar.write(f"**Difficulty:** {selected.get('Difficulty')}")
        st.sidebar.write(f"**Description:** {selected.get('Description')[:150]}...")
else:
    try:
        sample = df.sample(1).iloc[0]
        highlight_name = sample["Name"]
        st.sidebar.image(sample.get("Image_github") or sample.get("Image"), caption="Random Origami", width=220)
        st.sidebar.write(f"**Name:** {sample.get('Name')}")
        st.sidebar.write(f"**Creator:** {sample.get('Creator')}")
        st.sidebar.write(f"**Difficulty:** {sample.get('Difficulty')}")
        st.sidebar.write(f"**Description:** {sample.get('Description')[:150]}...")
    except Exception as e:
        st.sidebar.error(f"Sidebar error: {e}")

# Button to update
if st.button("📥 Scrape Latest Model & Update Dataset"):
    url = get_first_model_url()
    if url:
        new_model = scrape_model_detail(url)
        if new_model and new_model['Name'].lower() not in set(df['Name'].dropna().str.lower()):
            st.success(f"🆕 Adding new model: {new_model['Name']}")
            df = pd.concat([pd.DataFrame([new_model]), df], ignore_index=True)
            df.to_csv(CSV_PATH, index=False)
            highlight_name = new_model["Name"]
        else:
            st.info(f"ℹ️ The '{new_model['Name']}' is the most recent origami model.")
    else:
        st.error("❌ Could not find latest model.")

# Plot the data
process_and_plot(df, highlight_name=highlight_name)
