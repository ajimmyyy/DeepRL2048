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
            # 計算時間資訊
            if "timestamp" in data and len(data["timestamp"]) >= 2:
                start_time = data["timestamp"][0]
                end_time = data["timestamp"][-1]
                elapsed_seconds = end_time - start_time
                avg_time_per_episode = elapsed_seconds / (len(data["episode"]) - 1)

                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
                avg_str = f"{avg_time_per_episode:.2f} 秒"
            else:
                elapsed_str = "無法計算"
                avg_str = "無法計算"

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

            st.markdown("---")
            st.subheader("⏱️ 執行時間統計")
            st.markdown(f"- **總經過時間**：{elapsed_str}")
            st.markdown(f"- **平均每集時間**：{avg_str}")

        time.sleep(2)

    except Exception as e:
        st.warning(f"等待資料中... 錯誤：{e}")
        time.sleep(2)
