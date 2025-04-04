# ui/reward_dashboard.py
import streamlit as st
import json
import time
import pandas as pd
import altair as alt

st.set_page_config(page_title="Reward Tracker", layout="wide")
st.title("📈 DQN Reward Trend")

DATA_PATH = "reward_data.json"
SMOOTHING_WINDOW = 10

placeholder = st.empty()

while True:
    try:
        with open(DATA_PATH, "r") as f:
            data = json.load(f)

        df = pd.DataFrame({
            "Episode": data["episode"],
            "Total Reward": data["total_reward"],
            "Avg Reward": data["avg_reward"]
        })

        df["Smoothed Avg Reward"] = df["Avg Reward"].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

        with placeholder.container():
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Total Reward per Episode")
                total_chart = alt.Chart(df).mark_line().encode(
                    x="Episode", y="Total Reward"
                ).properties(width=600, height=400)
                st.altair_chart(total_chart)

            with col2:
                st.subheader(f"Avg Reward & Smoothed Avg Reward (Window={SMOOTHING_WINDOW})")

                avg_line = alt.Chart(df).mark_line(opacity=0.5).encode(
                    x="Episode", y="Avg Reward"
                )

                smoothed_line = alt.Chart(df).mark_line(color="red").encode(
                    x="Episode", y="Smoothed Avg Reward"
                )

                combined_chart = alt.layer(avg_line, smoothed_line).properties(width=600, height=400)
                st.altair_chart(combined_chart)

        time.sleep(2)

    except Exception as e:
        st.warning(f"等待資料中... 錯誤：{e}")
        time.sleep(2)
