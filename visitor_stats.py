import streamlit as st
import pandas as pd
import os

LOG_PATH = "visitor_log.csv"

st.title("📈 Visitor Interaction Statistics")

if os.path.exists(LOG_PATH):
    df = pd.read_csv(LOG_PATH, names=["timestamp", "event_type", "detail"], parse_dates=["timestamp"])

    total_visits = (df["event_type"] == "visit").sum()
    total_searches = (df["event_type"] == "search").sum()
    total_randoms = (df["event_type"] == "random_click").sum()
    total_scrapes = (df["event_type"] == "scrape_click").sum()

    st.markdown(f"- **Total Visits:** {total_visits}")
    st.markdown(f"- **Model Searches:** {total_searches}")
    st.markdown(f"- **Random Button Clicks:** {total_randoms}")
    st.markdown(f"- **Scrape Attempts:** {total_scrapes}")

    # Most searched model names
    st.markdown("### 🔍 Most Searched Origami Models")
    search_names = (
        df[df["event_type"] == "search"]["detail"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Model Name", "detail": "Search Count"})
    )
    st.dataframe(search_names, use_container_width=True)

    chart_data = pd.DataFrame({
        "Event": ["Visits", "Searches", "Random Clicks", "Scrape Clicks"],
        "Count": [total_visits, total_searches, total_randoms, total_scrapes]
    })

    st.markdown("### 📊 Interaction Chart")
    st.bar_chart(chart_data.set_index("Event"))

    with st.expander("📝 View Raw Event Log"):
        st.dataframe(df.tail(20), use_container_width=True)
else:
    st.info("No interaction data yet. Visit the main page and try searching or clicking buttons.")
