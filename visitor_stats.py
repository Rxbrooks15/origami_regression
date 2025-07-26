import streamlit as st
import pandas as pd

st.title("📈 Visitor Interaction Statistics")

# Use session_state to retrieve live stats
stats = {
    "visits": st.session_state.get("visits", 0),
    "searches": st.session_state.get("searches", 0),
    "random_clicks": st.session_state.get("random_clicks", 0),
    "scrapes": st.session_state.get("scrapes", 0),
}

st.markdown(f"- **Total Visits (this session):** {stats['visits']}")
st.markdown(f"- **Model Searches:** {stats['searches']}")
st.markdown(f"- **Random Button Clicks:** {stats['random_clicks']}")
st.markdown(f"- **Scrape Attempts:** {stats['scrapes']}")

# Optional bar chart
df = pd.DataFrame.from_dict(stats, orient='index', columns=['Count'])
st.bar_chart(df)
