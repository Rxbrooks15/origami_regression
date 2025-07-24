import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
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
        st.error("No model card found.")
        return None
    model_url = first_link["href"]
    if model_url.startswith("/"):
        model_url = "https://origami-database.com" + model_url
    return model_url

# --- Data processing functions ---
def convert_to_minutes(time_str):
    if pd.isna(time_str): return 0
    time_str = time_str.lower().replace("hours", "hr").replace("hour", "hr")
    time_str = time_str.replace("minutes", "min").replace("minute", "min")
    time_str = time_str.replace(".", "").strip()
    hours = re.search(r'(\d+)\s*hr', time_str)
    minutes = re.search(r'(\d+)\s*min', time_str)
    h = int(hours.group(1)) if hours else 0
    m = int(minutes.group(1)) if minutes else 0
    return h * 60 + m

def process_and_plot(df):
    import os

    df['time_minutes'] = df['Time'].apply(convert_to_minutes)
    df['Difficulty'] = df['Difficulty'].str.strip().str.lower()
    difficulty_map = {'easy': 1, 'moderate': 2, 'intermediate': 3, 'hard': 4, 'complex': 5}
    df['Difficulty_Numeric'] = df['Difficulty'].map(difficulty_map).fillna(1)

    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(""))
    nmf = NMF(n_components=7, random_state=42)
    topic_matrix = nmf.fit_transform(tfidf_matrix)
    df['Dominant_Topic'] = topic_matrix.argmax(axis=1)

    topic_difficulty_weights = {6: 2.13, 2: 2.57, 3: 2.89, 1: 2.92, 0: 2.95, 5: 3.08, 4: 3.13}
    df['Topic_Weighted_Difficulty'] = df['Dominant_Topic'].map(topic_difficulty_weights)

    df['Name_Length'] = df['Name'].astype(str).apply(len)
    df['Description_Length'] = df['Description'].astype(str).apply(len)

    name_avg = df.groupby('Difficulty_Numeric')['Name_Length'].transform('mean')
    desc_avg = df.groupby('Difficulty_Numeric')['Description_Length'].transform('mean')

    df['Name_Score'] = df['Name_Length'] / name_avg
    df['Description_Score'] = df['Description_Length'] / desc_avg

    df['Complexity_Score'] = (
        df['Difficulty_Numeric'] +
        df['Topic_Weighted_Difficulty'] +
        df['Name_Score'] +
        df['Description_Score']
    ) / 4

    # Add GitHub-hosted image path
    df["Image_github"] = df["Image"].apply(
        lambda url: "https://raw.githubusercontent.com/rxbrooks15/origami_regression/main/folder/" + os.path.basename(str(url))
    )

    X = df[['time_minutes']].values
    y = df['Complexity_Score'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    r2_scores = {}
    best_degree = 1
    best_r2_val = -np.inf
    best_model = None
    best_poly = None

    for degree in range(1, 7):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        y_val_pred = model.predict(X_test_poly)
        r2_val = r2_score(y_test, y_val_pred)
        r2_scores[degree] = r2_val

        if r2_val > best_r2_val:
            best_degree = degree
            best_r2_val = r2_val
            best_model = model
            best_poly = poly

    X_full_sorted = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    X_full_poly = best_poly.transform(X_full_sorted)
    y_full_pred = best_model.predict(X_full_poly)

    fig = px.scatter(
        df,
        x='time_minutes',
        y='Complexity_Score',
        color='Topic_Weighted_Difficulty',
        custom_data=[
            'Image_github', 'Name', 'Creator', 'time_minutes', 'Complexity_Score',
            'Description', 'Dominant_Topic', 'Topic_Weighted_Difficulty',
            'Name_Score', 'Description_Score'
        ],
        title=f'Polynomial Fit (Degree {best_degree}) | Validation R²: {best_r2_val:.3f}'
    )

    fig.update_traces(
    hovertemplate="""
    <b>%{customdata[1]}</b><br>
    Creator: %{customdata[2]}<br>
    Time: %{customdata[3]:.1f} min<br>
    Complexity Score: %{customdata[4]:.2f}<br>
    Topic: %{customdata[6]} | Weight: %{customdata[7]:.2f}<br>
    Name Score: %{customdata[8]:.2f} | Desc Score: %{customdata[9]:.2f}<br>
    <br><br>
    <i>Description:</i> %{customdata[5]}<br>
    <extra></extra>
    """,
    marker=dict(size=9, opacity=0.85),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
)

    fig.add_trace(go.Scatter(
        x=X_full_sorted.flatten(),
        y=y_full_pred,
        mode='lines',
        name='Best Fit Curve',
        line=dict(color='black')
    ))

    fig.update_layout(
        xaxis_title='⏱️ Time (minutes)',
        yaxis_title='📊 Complexity Score'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### Total Observations: {df.shape[0]}")
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

if st.button("📥 Scrape Latest Model & Update Dataset"):
    df = pd.read_csv(CSV_PATH)
    existing_names = set(df['Name'].dropna().str.lower())
    url = get_first_model_url()
    if url:
        new_model = scrape_model_detail(url)
        if new_model:
            if new_model['Name'].lower() not in existing_names:
                st.success(f"🆕 Adding new model: {new_model['Name']}")
                df_new = pd.DataFrame([new_model])
                df = pd.concat([df_new, df], ignore_index=True)
                df.to_csv(CSV_PATH, index=False)
            else:
                st.info(f"ℹ️ The '{new_model['Name']}' is the most recent origami model.")
        else:
            st.error("❌ Failed to scrape the new model details.")
    else:
        st.error("❌ Failed to find new model URL.")
    df = pd.read_csv(CSV_PATH)
    process_and_plot(df)

else:
    df = pd.read_csv(CSV_PATH)
    process_and_plot(df)


# --- Streamlit Sidebar for Preview ---
st.sidebar.header("🔍 Preview Image Test")

try:
    sample_row = df.sample(1).iloc[0]
    sample_image = sample_row["Image_github"] if "Image_github" in df.columns else sample_row.get("Image", None)
except Exception as e:
    sample_row = None
    sample_image = None
    st.sidebar.error(f"Sidebar error: {e}")

if sample_image:
    st.sidebar.image(sample_image, caption="Sample Origami Image", width=220)
else:
    st.sidebar.write("No image found to preview.")

st.sidebar.markdown("---")
st.sidebar.markdown("Random model info:")
if sample_row is not None:
    st.sidebar.write(f"**Name:** {sample_row.get('Name', 'N/A')}")
    st.sidebar.write(f"**Creator:** {sample_row.get('Creator', 'N/A')}")
    st.sidebar.write(f"**Difficulty:** {sample_row.get('Difficulty', 'N/A')}")
    st.sidebar.write(f"**Description:** {sample_row.get('Description', 'N/A')[:150]}...")
else:
    st.sidebar.write("No data available.")
