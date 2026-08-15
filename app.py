import time
import random
import pandas as pd
import plotly.express as px
import streamlit as st
from sentiment_engine import SentimentEngine

st.set_page_config(page_title="Real-Time Sentiment Analysis", layout="wide")

st.title("📊 Real-Time Twitter Sentiment Analysis")
st.caption("Live NLP Sentiment Pipeline powered by VADER and Streamlit")

# Sample tweet stream for real-time simulation
SAMPLE_TWEETS = [
    "Loving the performance of this new app release! Absolutely brilliant work 🔥",
    "This system update broke my workflow completely. Worst customer experience ever.",
    "Just reading through some data science articles for the weekend.",
    "Outstanding support team! Quick response and resolved my issue immediately.",
    "I have mixed feelings about the latest stock market trends today.",
    "The new UI design looks sleek and smooth. Great job by the dev team!",
    "Extremely disappointed with the product quality. Will ask for a refund.",
    "Analyzing sentiment pipelines in Python using Streamlit and VADER."
]

# Initialize engine and session memory
engine = SentimentEngine()
if "df_data" not in st.session_state:
    st.session_state.df_data = pd.DataFrame(columns=["timestamp", "raw_text", "sentiment", "compound"])

# Control buttons
col_start, col_stop = st.columns([1, 5])
run_stream = col_start.button("▶️ Start Live Stream", type="primary")

# Metric containers
kpi1, kpi2, kpi3 = st.columns(3)
chart_col1, chart_col2 = st.columns(2)
table_placeholder = st.empty()

if run_stream:
    for _ in range(20):  # Simulates 20 live updates
        time.sleep(1.5)  # Fetch delay
        
        # Simulate incoming tweet
        tweet_text = random.choice(SAMPLE_TWEETS)
        result = engine.analyze(tweet_text)

        # Update dataframe state
        new_row = {
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
            "raw_text": result["raw_text"],
            "sentiment": result["sentiment"],
            "compound": result["compound"]
        }
        st.session_state.df_data = pd.concat(
            [st.session_state.df_data, pd.DataFrame([new_row])], ignore_index=True
        )

        df = st.session_state.df_data

        # Render KPI Cards
        kpi1.metric("Total Ingested", len(df))
        kpi2.metric("Positive Sentiment", f"{(df['sentiment']=='Positive').mean()*100:.1f}%")
        kpi3.metric("Avg Compound Score", f"{df['compound'].mean():.2f}")

        # Render Pie Chart
        fig_pie = px.pie(df, names="sentiment", title="Sentiment Distribution", color_discrete_sequence=px.colors.qualitative.Set2)
        chart_col1.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{len(df)}")

        # Render Time Series Chart
        fig_line = px.line(df, x="timestamp", y="compound", title="Sentiment Compound Score Trend")
        chart_col2.plotly_chart(fig_line, use_container_width=True, key=f"line_{len(df)}")

        # Render Live Feed Table
        table_placeholder.dataframe(df.tail(10), use_container_width=True)