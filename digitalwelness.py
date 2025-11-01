import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import time
from filelock import FileLock
import os
import streamlit.components.v1 as components

# ---------- Config ----------
DATA_FILE = "wellness_data.csv"
LOCK_FILE = DATA_FILE + ".lock"
SCORE_MIN, SCORE_MAX = 0, 100
INACTIVITY_LIMIT = 300  # seconds (5 minutes)

# ---------- Helpers ----------
def ensure_datafile():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=[
            "username", "date", "sleep_hours", "screen_time", "stress_level",
            "mood", "wellness_score", "tip", "journal"
        ])
        with FileLock(LOCK_FILE, timeout=10):
            df.to_csv(DATA_FILE, index=False)

def load_data():
    ensure_datafile()
    with FileLock(LOCK_FILE, timeout=10):
        df = pd.read_csv(DATA_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    expected = ["username","date","sleep_hours","screen_time","stress_level","mood","wellness_score","tip","journal"]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA
    return df.drop_duplicates(subset=["username","date"], keep="last")

def save_data(df):
    with FileLock(LOCK_FILE, timeout=10):
        df.to_csv(DATA_FILE, index=False)

def save_entry(entry):
    df = load_data()
    exists = ((df["username"] == entry["username"]) & (df["date"] == entry["date"])).any()
    if exists:
        st.warning("You’ve already submitted an entry for this date.")
        return False
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    save_data(df)
    return True

def compute_wellness_score(sleep, screen, stress):
    sleep_score = np.clip((sleep / 8.0) * 40, 0, 40)
    stress_score = np.clip((10 - stress) / 10.0 * 30, 0, 30)
    screen_score = 30 if screen <= 3 else max(0, 30 - (screen - 3) * (30 / 9))
    return int(np.clip(sleep_score + stress_score + screen_score, SCORE_MIN, SCORE_MAX))

def generate_tip(sleep, screen, stress, mood):
    tip_parts = []
    if sleep < 6:
        tip_parts.append("🛌 Try sleeping 7–8 hours.")
    if screen > 8:
        tip_parts.append("📱 Too much screen time — reduce it.")
    if stress >= 7:
        tip_parts.append("😣 High stress — try breathing exercises.")
    if mood.lower() in ["tired","exhausted"]:
        tip_parts.append("💤 Consider a short power nap.")
    if not tip_parts:
        tip_parts.append("👍 Looking good — keep it up!")
    return " ".join(tip_parts)

def render_card(title, value, color="#4CAF50", emoji=""):
    st.markdown(f"""
    <div style='background-color:{color}; padding:14px; border-radius:12px; text-align:center;'>  
    <h4 style='color:white; margin:0;'>{emoji} {title}</h4>  
    <p style='font-size:24px; font-weight:bold; color:white; margin:6px 0 0 0;'>{value}</p>  
    </div>  
    """, unsafe_allow_html=True)

def get_last_n_days(df, n=7, username=None):
    df_user = df[df["username"] == username] if username else df.copy()
    df_user["date_only"] = pd.to_datetime(df_user["date"]).dt.normalize()
    today = pd.Timestamp(datetime.now().date())
    start = today - pd.Timedelta(days=n-1)
    return df_user[df_user["date_only"] >= start].sort_values("date_only").tail(n)

def weekly_summary_table(df_user):
    if df_user.empty:
        return None
    df = df_user.copy()
    df["weekday"] = df["date"].dt.day_name()
    week_avg = df.groupby("weekday")[["stress_level","screen_time","sleep_hours","wellness_score"]].mean()
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    week_avg = week_avg.reindex(weekdays).dropna(how="all")
    return week_avg.round(2)

def trend_tip_for_user(df_user):
    if df_user.shape[0] >= 2:
        last_two = df_user.sort_values("date").tail(2)
        prev = float(last_two.iloc[0]["wellness_score"])
        curr = float(last_two.iloc[1]["wellness_score"])
        diff = curr - prev
        pct = (diff / prev * 100) if prev != 0 else (diff * 100)
        if diff > 0:
            return f"🎉 Nice — wellness improved by {pct:.1f}% vs previous entry."
        elif diff < 0:
            return f"⚠ Wellness dropped by {abs(pct):.1f}%. Try improving sleep or reducing stress."
        else:
            return "😌 Wellness unchanged vs previous entry."
    return None

# ---------- Inactivity ----------
def check_inactivity():
    if "last_active" not in st.session_state:
        st.session_state.last_active = time.time()
    if time.time() - st.session_state.last_active > INACTIVITY_LIMIT:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.warning("You were logged out due to inactivity. Refresh and start again.")
        st.stop()
    else:
        st.session_state.last_active = time.time()

# ---------- Sound ----------
def play_beep():
    js = """
    <script>  
    try {  
    const ctx = new (window.AudioContext || window.webkitAudioContext)();  
    const o = ctx.createOscillator();  
    const g = ctx.createGain();  
    o.type = 'sine';  
    o.frequency.value = 880;  
    g.gain.value = 0.05;  
    o.connect(g);  
    g.connect(ctx.destination);  
    o.start();  
    setTimeout(()=>{ o.stop(); ctx.close(); }, 150);  
    } catch(e) { console.log('beep error', e); }  
    </script>  
    """
    components.html(js, height=0)

# ---------- App setup ----------
st.set_page_config(page_title="🌿 Digital Wellness App", layout="wide")
check_inactivity()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "date_input" not in st.session_state:
    st.session_state.date_input = pd.Timestamp(datetime.now().date()).normalize()
if "page" not in st.session_state:
    st.session_state.page = "Today's Check-in"
if "checkin_done" not in st.session_state:
    st.session_state.checkin_done = False

# ---------- LOGIN ----------
if not st.session_state.logged_in:
    st.markdown("<div style='background-color:#fff3e0; padding:40px; border-radius:12px; text-align:center;'><h1 style='color:#FF4500;margin:0;'>👤 Digital Wellness Login</h1><p>Enter your name and choose date to continue.</p></div>", unsafe_allow_html=True)
    name = st.text_input("Your name:", max_chars=30, placeholder="e.g., thejashree")
    date_sel = st.date_input("Date:", value=datetime.now().date())
    if st.button("Continue"):
        if name and name.strip():
            st.session_state.logged_in = True
            st.session_state.username = name.strip()
            st.session_state.date_input = pd.Timestamp(date_sel).normalize()
            st.session_state.page = "Today's Check-in"
            st.rerun()
        else:
            st.error("Please enter your name.")
    st.stop()

# ---------- MAIN ----------
check_inactivity()
username = st.session_state.username
date_input = st.session_state.date_input
data = load_data()

st.markdown(f"<div style='background-color:#e0f7fa; padding:16px; border-radius:10px;'><h2 style='color:#FF4500; margin:0;'>Welcome, <span style='color:#FFD700'>{username}</span>!</h2><p style='margin:0;color:#FF8C00;'>Selected date: {date_input.strftime('%B %d, %Y')}</p></div>", unsafe_allow_html=True)

option = st.selectbox("Choose an option:", ["Today's Check-in","Weekly Overview","Leaderboard","View Past Entries","Clear My Past Entries","Edit / Delete Entries","Switch Account","Exit App"], key="menu_option")
st.session_state.page = option
check_inactivity()

# ---------------- Today's Check-in ----------------
if st.session_state.page == "Today's Check-in":
    today_ts = pd.Timestamp(date_input).normalize()
    data = load_data()
    df_user = data[data["username"] == username].sort_values("date")
    today_entry = data[(data["username"] == username) & (data["date"] == today_ts)]

    left, right = st.columns([1,3])
    with left:
        st.markdown("### 🎯 Targets")
        st.markdown("- Sleep: 8 hrs")
        st.markdown("- Screen: ≤3 hrs")
        st.markdown("- Stress: ≤4")
    with right:
        if today_entry.empty and not st.session_state.checkin_done:
            sleep_hours = st.number_input("Sleep Hours (0-12)", min_value=0, max_value=12, value=8, step=1)
            screen_time = st.number_input("Screen Time (hrs, 0-24)", min_value=0, max_value=24, value=3, step=1)
            stress_level = st.slider("Stress Level (0-10)", min_value=0, max_value=10, value=5)
            mood = st.selectbox("Mood", ["Happy","Tired","Sad","Anxious","Stressed"])
            journal = st.text_area("Journal / Notes", value="")

            if st.button("Submit Today's Check-in"):
                score = compute_wellness_score(sleep_hours, screen_time, stress_level)
                tip = generate_tip(sleep_hours, screen_time, stress_level, mood)
                entry = {
                    "username": username,
                    "date": today_ts,
                    "sleep_hours": sleep_hours,
                    "screen_time": screen_time,
                    "stress_level": stress_level,
                    "mood": mood,
                    "wellness_score": score,
                    "tip": tip,
                    "journal": journal
                }
                ok = save_entry(entry)
                if ok:
                    st.balloons()
                    play_beep()
                    st.success("✅ Today's check-in saved!")
                    st.session_state.checkin_done = True
                    st.rerun()

        else:
            st.info("✅ Already submitted for this date — showing analysis below.")

    data = load_data()
    today_entry = data[(data["username"] == username) & (data["date"] == today_ts)]
    if not today_entry.empty:
        row = today_entry.iloc[-1]
        st.subheader("📊 Today's Analysis")
        c1,c2,c3,c4 = st.columns(4)
        with c1: render_card("Stress", row["stress_level"], color="#FF4B4B", emoji="😣")
        with c2: render_card("Screen", row["screen_time"], color="#FFA500", emoji="📱")
        with c3: render_card("Sleep", row["sleep_hours"], color="#1E90FF", emoji="🛌")
        with c4: render_card("Score", row["wellness_score"], color="#4CAF50", emoji="🌿")

        stress_pct = (float(row["stress_level"]) / 10.0) * 100
        screen_pct = (float(row["screen_time"]) / 24.0) * 100
        sleep_pct = min(100, (float(row["sleep_hours"]) / 8.0) * 100)
        score_pct = float(row["wellness_score"])
        metrics = ["Stress (%)","Screen (%)","Sleep (%)","Wellness Score (%)"]
        vals = [round(stress_pct,1), round(screen_pct,1), round(sleep_pct,1), round(score_pct,1)]
        cmap = {"Stress (%)":"#FF4B4B","Screen (%)":"#FFA500","Sleep (%)":"#1E90FF","Wellness Score (%)":"#4CAF50"}
        fig = px.bar(x=metrics, y=vals, text=vals, color=metrics, color_discrete_map=cmap, labels={"y":"Percent (%)"})
        fig.update_traces(textposition='outside', marker_line_width=0)
        fig.update_layout(title="📊 Today's Metrics (as %)", yaxis=dict(range=[0,110]), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"*Tip (based on your latest entry):* {row.get('tip','')}")
        tip_note = trend_tip_for_user(df_user)
        if tip_note:
            st.info(tip_note)

        if row.get("journal",""):
            st.markdown(f"*Journal:* {row.get('journal','')}")

# ---------------- View Past Entries (now like Leaderboard cards) ----------------
elif st.session_state.page == "View Past Entries":
    st.header("📜 Past Entries")
    data = load_data()
    if data.empty:
        st.info("No entries.")
    else:
        users = ["All Users"] + sorted(data["username"].dropna().unique().tolist())
        sel = st.selectbox("Filter by user:", users, index=users.index(username) if username in users else 0)
        if sel == "All Users":
            display = data.sort_values(["username","date"], ascending=[True,False]).reset_index(drop=True)
        else:
            display = data[data["username"] == sel].sort_values("date", ascending=False).reset_index(drop=True)

        for _, row in display.iterrows():
            st.markdown(f"<div style='background:#1a1a1a; padding:10px; border-radius:8px; margin-bottom:6px;'>"
                        f"<h4 style='color:#FFD700;margin:0;'>📅 {pd.to_datetime(row['date']).strftime('%B %d, %Y')}</h4>"
                        f"<p style='color:white;margin:2px 0;'>User: {row['username']} | "
                        f"Score: {row['wellness_score']} | Sleep: {row['sleep_hours']}h | "
                        f"Screen: {row['screen_time']}h | Stress: {row['stress_level']} | Mood: {row['mood']}</p>"
                        f"<p style='color:#bbb;margin:0;'>Tip: {row['tip']}</p></div>", unsafe_allow_html=True)
# ---------------- Weekly Overview ----------------
elif st.session_state.page == "Weekly Overview":
    st.header("📅 Weekly Overview")
    df_user = get_last_n_days(data, 7, username)
    if df_user.empty:
        st.info("No data available for the past week.")
    else:
        st.subheader("📈 Wellness Trend (Last 7 Days)")
        fig = px.line(df_user, x="date", y="wellness_score", markers=True,
                      title="Wellness Score Trend", labels={"wellness_score": "Score", "date": "Date"})
        fig.update_traces(line_color="#4CAF50", marker=dict(size=8))
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Weekly Averages by Day")
        week_table = weekly_summary_table(df_user)
        if week_table is not None:
            st.dataframe(week_table, use_container_width=True)

# ---------------- Leaderboard ----------------
elif st.session_state.page == "Leaderboard":
    st.header("🏆 Wellness Leaderboard")
    if data.empty:
        st.info("No users yet.")
    else:
        latest_scores = data.sort_values("date").groupby("username").last().reset_index()
        top_users = latest_scores.sort_values("wellness_score", ascending=False)
        fig = px.bar(top_users, x="username", y="wellness_score", color="wellness_score",
                     color_continuous_scale="greens", title="Top Wellness Scores")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_users[["username", "wellness_score", "sleep_hours", "screen_time", "stress_level"]], use_container_width=True)

# ---------------- Clear My Past Entries ----------------
elif st.session_state.page == "Clear My Past Entries":
    st.header("🧹 Clear My Past Entries")
    df_user = data[data["username"] == username]
    if df_user.empty:
        st.info("You have no saved entries.")
    else:
        if st.button("⚠ Delete all my past entries"):
            confirm = st.checkbox("I confirm I want to delete my data.")
            if confirm:
                df_new = data[data["username"] != username]
                save_data(df_new)
                st.success("All your entries were deleted.")
                play_beep()
                st.rerun()

# ---------------- Edit / Delete Entries ----------------
elif st.session_state.page == "Edit / Delete Entries":
    st.header("✏ Edit / Delete Entries")
    df_user = data[data["username"] == username].sort_values("date", ascending=False)
    if df_user.empty:
        st.info("You have no entries to edit.")
    else:
        dates = df_user["date"].dt.strftime("%Y-%m-%d").tolist()
        selected = st.selectbox("Select date to edit/delete:", dates)
        if selected:
            entry = df_user[df_user["date"].dt.strftime("%Y-%m-%d") == selected].iloc[0]
            new_sleep = st.number_input("Sleep Hours", 0, 12, int(entry["sleep_hours"]))
            new_screen = st.number_input("Screen Time (hrs)", 0, 24, int(entry["screen_time"]))
            new_stress = st.slider("Stress Level (0–10)", 0, 10, int(entry["stress_level"]))
            new_mood = st.selectbox("Mood", ["Happy", "Tired", "Sad", "Anxious", "Stressed"], index=["Happy","Tired","Sad","Anxious","Stressed"].index(entry["mood"]))
            new_journal = st.text_area("Journal", value=entry["journal"])

            if st.button("💾 Save Changes"):
                df_user.loc[df_user["date"] == entry["date"], ["sleep_hours","screen_time","stress_level","mood","journal"]] = \
                    [new_sleep, new_screen, new_stress, new_mood, new_journal]
                df_user["wellness_score"] = df_user.apply(lambda r: compute_wellness_score(r["sleep_hours"], r["screen_time"], r["stress_level"]), axis=1)
                df_user["tip"] = df_user.apply(lambda r: generate_tip(r["sleep_hours"], r["screen_time"], r["stress_level"], r["mood"]), axis=1)
                df_other = data[data["username"] != username]
                save_data(pd.concat([df_other, df_user], ignore_index=True))
                st.success("✅ Entry updated successfully!")
                play_beep()
                st.rerun()

            if st.button("🗑 Delete This Entry"):
                confirm = st.checkbox("I confirm deletion for this entry")
                if confirm:
                    df_new = data[~((data["username"] == username) & (data["date"].dt.strftime("%Y-%m-%d") == selected))]
                    save_data(df_new)
                    st.success("Entry deleted.")
                    play_beep()
                    st.rerun()

# ---------------- Switch Account ----------------
elif st.session_state.page == "Switch Account":
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.page = "Today's Check-in"
    st.success("Switched account successfully.")
    st.rerun()

# ---------------- Exit App ----------------
elif st.session_state.page == "Exit App":
    st.markdown("<h3>👋 You have exited the app. Refresh to restart.</h3>", unsafe_allow_html=True)
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.stop()