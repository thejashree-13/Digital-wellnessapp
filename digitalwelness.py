# Full polished Streamlit app with requested features
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os
import logging
from filelock import FileLock
import io
import math
import wave
import struct

# ---------- Config ----------
DATA_FILE = "wellness_data.csv"
SCORE_MIN, SCORE_MAX = 0, 100
LOCK_TIMEOUT = 10  # seconds

# ---------- Logging ----------
logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO
)

# ---------- Page config ----------
st.set_page_config(page_title="🌿 Digital Wellness App", layout="wide")

# ---------- Page-wide CSS ----------
st.markdown(
    """
    <style>
    /* Sidebar styling */
    .css-1d391kg {padding-top: 1rem;} /* layout spacing */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071029, #0b1220);
        color: #e6eef6;
        padding: 1rem 0.8rem;
        border-right: 1px solid rgba(255,255,255,0.03);
    }
    [data-testid="stSidebar"] .stRadio > div label {
        color: #cfe8ff;
    }
    .stRadio input[type="radio"]:checked + label {
        background: linear-gradient(90deg, rgba(99,102,241,0.12), rgba(56,189,248,0.04));
        border-radius: 8px;
        padding: 6px 8px;
    }

    /* Card styling used for leaderboard/past-entries */
    .card-black {
        background-color: #0b0b0b;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .card-title { margin: 0; color: #fff; font-weight:600; }
    .card-sub { margin: 0; color: #cfcfcf; font-size:13px; }
    .medal { font-size:20px; margin-right:6px; vertical-align:middle; }

    /* Goals card tweaks */
    .goals {
        background: linear-gradient(90deg, rgba(34,197,94,0.06), rgba(16,185,129,0.02));
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
    }

    @media (max-width: 600px) {
        .css-1d391kg { padding-top: .5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Helper utilities ----------
def ensure_datafile():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=[
            "username", "date", "sleep_hours", "screen_time", "stress_level",
            "mood", "wellness_score", "tip", "journal"
        ])
        df.to_csv(DATA_FILE, index=False)

@st.cache_data
def load_data():
    ensure_datafile()
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        logging.error("Failed to read data file: %s", e)
        return pd.DataFrame(columns=[
            "username", "date", "sleep_hours", "screen_time", "stress_level",
            "mood", "wellness_score", "tip", "journal"
        ])

    # Normalize date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.date.astype(str)

    for col in ["sleep_hours", "screen_time", "stress_level", "wellness_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["username", "mood", "tip", "journal"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    # Keep last entry for username+date
    df = df.drop_duplicates(subset=["username", "date"], keep="last")
    return df

def save_entry(entry):
    lock = FileLock(f"{DATA_FILE}.lock", timeout=LOCK_TIMEOUT)
    try:
        with lock:
            df = load_data()
            df['date_only'] = pd.to_datetime(df['date'], errors='coerce').dt.date
            entry_date = pd.to_datetime(entry['date']).date()
            exists = ((df['username'] == entry['username']) &
                      (df['date_only'] == entry_date)).any()
            if exists:
                st.warning("You’ve already submitted an entry for this date.")
                return False

            entry_to_save = entry.copy()
            entry_to_save['date'] = pd.to_datetime(entry_to_save['date']).date().isoformat()
            df_new = pd.concat(
                [df.drop(columns=['date_only'], errors='ignore'),
                 pd.DataFrame([entry_to_save])],
                ignore_index=True
            )
            df_new.to_csv(DATA_FILE, index=False)
            try:
                st.cache_data.clear()
            except Exception:
                pass
            return True
    except Exception as e:
        logging.error("Error saving entry: %s", e)
        st.error("An error occurred while saving. Please try again.")
        return False

def compute_wellness_score(sleep, screen, stress):
    sleep_score = np.clip((sleep / 8.0) * 40, 0, 40)
    stress_score = np.clip((10 - stress) / 10.0 * 30, 0, 30)
    screen_score = 30 if screen <= 3 else max(0, 30 - (screen - 3) * (30 / 9))
    return int(np.clip(sleep_score + stress_score + screen_score, SCORE_MIN, SCORE_MAX))

def generate_tip(sleep, screen, stress, mood):
    tip = ""
    try:
        mood_text = str(mood).lower()
    except Exception:
        mood_text = ""
    if sleep < 6: tip += "🛌 Try sleeping 7–8 hours. "
    if screen > 8: tip += "📱 Try reducing screen time. "
    if stress >= 7: tip += "😣 Do short breathing exercises. "
    if mood_text in ["tired", "exhausted"]: tip += "💤 Take a brief power nap. "
    if tip == "": tip = "🌟 Keep going — small, consistent habits help!"
    return tip.strip()

def render_card(title, value, delta=None, color="#4CAF50", emoji=""):
    delta_text = f"<br><span style='font-size:15px; color:white;'>Δ {delta}</span>" if delta else ""
    st.markdown(f"""
    <div style='background-color:{color}; padding:18px; border-radius:12px; text-align:center;'>
        <h3 style='color:white; margin:0; font-size:16px;'>{emoji} {title}</h3>
        <p style='font-size:24px; font-weight:bold; color:white; margin:6px 0;'>{value}</p>
        {delta_text}
    </div>
    """, unsafe_allow_html=True)

def get_last_n_days(df, n=7, username=None):
    df_user = df[df["username"]==username] if username else df
    df_user = df_user.copy()
    df_user["date_only"] = pd.to_datetime(df_user["date"], errors='coerce').dt.date
    df_user = df_user.dropna(subset=["date_only"])
    today = pd.Timestamp(datetime.now().date())
    start = today - pd.Timedelta(days=n-1)
    mask = df_user["date_only"] >= start.date()
    df_filtered = df_user[mask]
    return df_filtered.sort_values("date_only").tail(n)

# ---------- Small beep generator for confirmation sound ----------
def generate_beep(duration_ms=220, freq=880.0, volume=0.18, sample_rate=22050):
    duration_s = duration_ms / 1000.0
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        max_amp = 32767 * volume
        for i in range(n_samples):
            t = float(i) / sample_rate
            envelope = math.sin(math.pi * (i / n_samples))
            val = int(max_amp * envelope * math.sin(2 * math.pi * freq * t))
            wf.writeframes(struct.pack('<h', val))
    buf.seek(0)
    return buf.read()

# ---------- Session initialization ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "show_balloons" not in st.session_state:
    st.session_state.show_balloons = False

# ---------- LOGIN PAGE ----------
if not st.session_state.logged_in:
    st.markdown("""
    <div style='background-color:#fff7ed; padding:36px; border-radius:12px; text-align:center;'>
        <h1 style='color:#FF4500; font-size:42px; margin-bottom:6px;'>👤 Digital Wellness Login</h1>
        <p style='margin-top:0; color:#6b7280;'>Quick check-in to track your daily wellness</p>
    </div>
    """, unsafe_allow_html=True)

    # Name first, then date below (vertical layout)
    username = st.text_input("Your Name:", max_chars=30)
    st.markdown("")  # spacer
    date_input = st.date_input("Select Date:")
    st.markdown("")  # spacer

    if st.button("Continue"):
        if username:
            st.session_state.logged_in = True
            st.session_state.username = username.strip()
            st.session_state.date_input = date_input
            st.session_state.page = "dashboard"
            # safe rerun
            try:
                st.experimental_rerun()
            except Exception as e:
                logging.warning("experimental_rerun failed after login: %s", e)
                st.success("Login saved — please refresh the page if it doesn't update automatically.")
                st.stop()
        else:
            st.error("Please enter your name!")
    st.stop()

# ---------- DASHBOARD ----------
username = st.session_state.username
date_input = st.session_state.date_input
data = load_data()

# Show stored balloons flag (multiple)
if st.session_state.get("show_balloons", False):
    for _ in range(3):
        try:
            st.balloons()
        except Exception:
            pass
    st.session_state.show_balloons = False

if "dashboard_page" not in st.session_state:
    st.session_state.dashboard_page = "Today's Check-in"

# ---------- Sidebar navigation (vertical) ----------
option = st.sidebar.radio(
    "Navigate",
    ["Today's Check-in", "Weekly Overview", "Leaderboard", "View Past Entries",
     "Clear All Past Entries", "Switch Account", "Exit App"],
    index=["Today's Check-in", "Weekly Overview", "Leaderboard", "View Past Entries",
           "Clear All Past Entries", "Switch Account", "Exit App"].index(st.session_state.dashboard_page)
)
st.session_state.dashboard_page = option

# ---------- Today's Check-in ----------
if option == "Today's Check-in":
    df_user = data[data["username"] == username].copy()
    df_user['date_only'] = pd.to_datetime(df_user['date'], errors='coerce').dt.date
    today_entry = df_user[df_user['date_only'] == pd.to_datetime(date_input).date()]

    c1, c2 = st.columns([1,3])
    with c1:
        st.markdown("<div class='goals'><h4 style='margin:0;'>🎯 Your Goals</h4><ul style='margin:6px 0 0 18px;'><li>Sleep: 8.0 hrs</li><li>Screen time: ≤ 3 hrs</li><li>Stress: ≤ 4</li></ul></div>", unsafe_allow_html=True)
    with c2:
        checkin_key = f"checkin_done_{username}_{str(date_input)}"
        already_done = st.session_state.get(checkin_key, False) or (not today_entry.empty)

        if not already_done:
            with st.form("checkin_form", clear_on_submit=False):
                sleep_hours = st.number_input("Sleep Hours (0-12)", min_value=0.0, max_value=12.0, value=8.0, step=0.5)
                screen_time = st.number_input("Screen Time (0-24)", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
                stress_level = st.slider("Stress Level (0-10)", min_value=0, max_value=10, value=5)
                mood = st.selectbox("Mood", ["Happy", "Tired", "Sad", "Anxious", "Stressed"])
                journal = st.text_area("Journal / Notes")
                submitted = st.form_submit_button("Submit Today's Check-in")

            if submitted:
                wellness_score = compute_wellness_score(sleep_hours, screen_time, stress_level)
                tip = generate_tip(sleep_hours, screen_time, stress_level, mood)
                entry = {
                    "username": username,
                    "date": pd.Timestamp(date_input),
                    "sleep_hours": float(sleep_hours),
                    "screen_time": float(screen_time),
                    "stress_level": int(stress_level),
                    "mood": mood,
                    "wellness_score": int(wellness_score),
                    "tip": tip,
                    "journal": journal
                }

                success = save_entry(entry)
                if success:
                    st.session_state[checkin_key] = True
                    st.session_state.show_balloons = True

                    # Play confirmation beep
                    try:
                        beep_bytes = generate_beep(duration_ms=260, freq=720.0, volume=0.18)
                        st.audio(beep_bytes, format='audio/wav')
                    except Exception as e:
                        logging.warning("Audio playback failed: %s", e)

                    # Extra balloons
                    try:
                        for _ in range(2):
                            st.balloons()
                    except Exception:
                        pass

                    st.success("✅ Today's check-in saved!")
                    # reload data and show immediate graph + tips
                    data = load_data()

                    st.markdown("### 📈 Quick Mini Dashboard")
                    recent = get_last_n_days(data, 7, username)
                    if not recent.empty:
                        recent["date"] = pd.to_datetime(recent["date"], errors='coerce')
                        recent_melt = recent.melt(
                            id_vars="date",
                            value_vars=["stress_level", "screen_time", "sleep_hours", "wellness_score"],
                            var_name="Metric",
                            value_name="Value"
                        )
                        fig = px.line(
                            recent_melt,
                            x=recent_melt["date"].dt.strftime('%b %d'),
                            y="Value",
                            color="Metric",
                            markers=True
                        )
                        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=340)
                        left, right = st.columns([2,1])
                        with left:
                            st.plotly_chart(fig, use_container_width=True)
                        with right:
                            st.markdown("#### 💡 Personalized Tips")
                            st.markdown(f"> {tip}")
                    else:
                        st.info("No historical data to show trends yet.")
                else:
                    st.error("Failed to save entry.")
        else:
            st.info("✅ You have already submitted today's check-in.")

    # If an entry exists for today, show metric cards
    if not today_entry.empty:
        row = today_entry.iloc[-1]
        st.subheader("📊 Today’s Analysis")
        c1, c2, c3, c4 = st.columns(4)
        with c1: render_card("Stress", int(row["stress_level"]), color="#FF4B4B", emoji="😣")
        with c2: render_card("Screen", float(row["screen_time"]), color="#FFA500", emoji="📱")
        with c3: render_card("Sleep", float(row["sleep_hours"]), color="#1E90FF", emoji="🛌")
        with c4: render_card("Score", int(row["wellness_score"]), color="#16A34A", emoji="🌿")

# ---------- Weekly Overview ----------
elif option == "Weekly Overview":
    st.header("📊 Weekly Overview (Last 7 Days)")
    last7 = get_last_n_days(data, 7, username)
    if last7.empty:
        st.info("No entries yet for weekly overview.")
    else:
        last7["date"] = pd.to_datetime(last7["date"], errors='coerce')
        last7_melt = last7.melt(
            id_vars="date",
            value_vars=["stress_level", "screen_time", "sleep_hours", "wellness_score"],
            var_name="Metric",
            value_name="Value"
        )
        fig = px.line(
            last7_melt,
            x=last7_melt["date"].dt.strftime('%b %d'),
            y="Value",
            color="Metric",
            markers=True,
            color_discrete_map={
                "stress_level": "red", "screen_time": "orange",
                "sleep_hours": "blue", "wellness_score": "green"
            }
        )
        fig.update_layout(
            title="📈 Weekly Trend - Stress, Screen, Sleep, Wellness",
            yaxis_title="Level / Hours / Score",
            plot_bgcolor="white", paper_bgcolor="white",
            height=440
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------- Leaderboard ----------
elif option == "Leaderboard":
    st.header("🏆 Leaderboard")
    df_score_type = st.selectbox("Select leaderboard type:", ["Daily", "Weekly"])
    today = pd.Timestamp(datetime.now().date())

    if df_score_type == "Daily":
        df_today = data[pd.to_datetime(data["date"], errors="coerce").dt.date == today.date()]
        df_score = df_today.groupby("username", as_index=False)["wellness_score"].mean()
    else:
        week_ago = today - pd.Timedelta(days=6)
        df_week = data[
            (pd.to_datetime(data["date"], errors="coerce").dt.date >= week_ago.date()) &
            (pd.to_datetime(data["date"], errors="coerce").dt.date <= today.date())
        ]
        df_score = df_week.groupby("username", as_index=False)["wellness_score"].mean()

    if df_score.empty:
        st.info("No leaderboard records yet.")
    else:
        df_score = df_score.sort_values("wellness_score", ascending=False).reset_index(drop=True)
        df_score["Rank"] = df_score.index + 1
        df_score["Medal"] = df_score["Rank"].apply(lambda r: ["🥇","🥈","🥉"][r-1] if r <= 3 else "")
        for _, row in df_score.iterrows():
            rank_color = "gold" if row["Rank"] == 1 else ("silver" if row["Rank"] == 2 else ("#cd7f32" if row["Rank"] == 3 else "white"))
            st.markdown(f"""
            <div class='card-black'>
                <h4 class='card-title' style='color:{rank_color}; margin:0;'>{row['Medal']} Rank {int(row['Rank'])}</h4>
                <p class='card-sub'>User: <strong style='color:#fff'>{row['username']}</strong> | Score: <strong style='color:#9ae6b4'>{row['wellness_score']:.1f}</strong></p>
            </div>
            """, unsafe_allow_html=True)

# ---------- View Past Entries (black leaderboard cards) ----------
elif option == "View Past Entries":
    st.header("📜 Past Entries (Your history)")
    df_user = data[data["username"] == username].sort_values("date", ascending=False)
    if df_user.empty:
        st.info("No past entries found.")
    else:
        for i, row in enumerate(df_user.itertuples(), start=1):
            date_str = pd.to_datetime(row.date, errors="coerce").strftime("%B %d, %Y") if pd.notnull(row.date) else "Date Missing"
            medal = "🔷"
            if i == 1: medal = "🥇"
            elif i == 2: medal = "🥈"
            elif i == 3: medal = "🥉"
            st.markdown(f"""
            <div class='card-black'>
                <h4 class='card-title' style='color:#f59e0b; margin:0;'>{medal} {i}. {date_str}</h4>
                <p class='card-sub'>Sleep: <strong>{row.sleep_hours}</strong> | Screen: <strong>{row.screen_time}</strong> | Stress: <strong>{row.stress_level}</strong> | Score: <strong style='color:#9ae6b4'>{row.wellness_score}</strong></p>
                <p class='card-sub'>Mood: <strong>{row.mood}</strong></p>
                <p class='card-sub'>Journal: <span style='color:#cfcfcf'>{row.journal}</span></p>
            </div>
            """, unsafe_allow_html=True)

# ---------- Clear / Switch / Exit ----------
elif option == "Clear All Past Entries":
    if st.button("⚠ Delete All Data"):
        ensure_datafile()
        pd.DataFrame(columns=[
            "username","date","sleep_hours","screen_time","stress_level",
            "mood","wellness_score","tip","journal"
        ]).to_csv(DATA_FILE, index=False)
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.success("✅ All entries deleted!")
        st.stop()

elif option == "Switch Account":
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    try:
        st.experimental_rerun()
    except Exception as e:
        logging.warning("experimental_rerun failed on Switch Account: %s", e)
        st.success("Signed out. Please refresh the page to sign in with another account.")
        st.stop()

elif option == "Exit App":
    st.stop()