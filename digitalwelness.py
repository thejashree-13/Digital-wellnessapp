# app.py
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
    """
    Returns:
      - True if saved (new)
      - False if an entry for that user/date already exists (not overwritten)
    """
    df = load_data()
    exists = ((df["username"] == entry["username"]) & (df["date"] == entry["date"])).any()
    if exists:
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
    if isinstance(mood, str) and mood.lower() in ["tired","exhausted"]:
        tip_parts.append("💤 Consider a short power nap.")
    if not tip_parts:
        tip_parts.append("👍 Looking good — keep it up!")
    return " ".join(tip_parts)

def render_card(title, value, color="#4CAF50", emoji=""):
    st.markdown(f"""
    <div style='background-color:{color}; padding:12px; border-radius:10px; text-align:center;'>    
      <h4 style='color:white; margin:0;'>{emoji} {title}</h4>    
      <p style='font-size:20px; font-weight:bold; color:white; margin:6px 0 0 0;'>{value}</p>    
    </div>    
    """, unsafe_allow_html=True)

def get_last_n_days(df, n=7, username=None):
    df_user = df[df["username"] == username] if username else df.copy()
    df_user = df_user.copy()
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

# ---------- Sound helper (plays a short beep using WebAudio) ----------
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
    } catch(e) {    
    console.log('beep error', e);    
    }    
    </script>  
    """
    components.html(js, height=0)

# ---------- App setup ----------
st.set_page_config(page_title="🌿 Digital Wellness App", layout="wide")
check_inactivity()

# session init
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
    st.markdown(
        "<div style='background-color:#fff3e0; padding:40px; border-radius:12px; text-align:center;'><h1 style='color:#FF4500;margin:0;'>👤 Digital Wellness Login</h1><p>Enter your name and choose date to continue.</p></div>",
        unsafe_allow_html=True
    )
    name = st.text_input("Your name:", max_chars=30, placeholder="e.g., thejashree")
    # date BELOW name (vertical layout)
    date_sel = st.date_input("Date:", value=datetime.now().date())

    if st.button("Continue"):
        if name and name.strip():
            # set new session values (fresh)
            st.session_state.logged_in = True
            st.session_state.username = name.strip()
            st.session_state.date_input = pd.Timestamp(date_sel).normalize()
            st.session_state.page = "Today's Check-in"
            st.session_state.checkin_done = False
            st.rerun()
        else:
            st.error("Please enter your name.")
    # stop rendering the rest until logged in
    st.stop()

# ---------- MAIN ----------
check_inactivity()
username = st.session_state.username
date_input = st.session_state.date_input
data = load_data()

# header
st.markdown(
    f"<div style='background-color:#e0f7fa; padding:16px; border-radius:10px;'><h2 style='color:#FF4500; margin:0;'>Welcome, <span style='color:#FFD700'>{username}</span>!</h2><p style='margin:0;color:#FF8C00;'>Selected date: {date_input.strftime('%B %d, %Y')}</p></div>",
    unsafe_allow_html=True
)

# menu
option = st.selectbox(
    "Choose an option:",
    ["Today's Check-in","Weekly Overview","Leaderboard","View Past Entries","Clear My Past Entries","Edit / Delete Entries","Switch Account","Exit App"],
    key="menu_option"
)
st.session_state.page = option
check_inactivity()
# ======================================================================
# -------------------- PAGE HANDLER: WELLNESS APP -----------------------
# ======================================================================

# ---------- TODAY'S CHECK-IN ----------
if st.session_state.page == "Today's Check-in":

    today_ts = pd.Timestamp(st.session_state.date_input).normalize()
    data = load_data()
    df_user = data[data["username"] == st.session_state.username].sort_values("date")
    today_entry = data[(data["username"] == st.session_state.username) & (data["date"] == today_ts)]

    left, right = st.columns([1, 3])
    with left:
        st.markdown("### 🎯 Targets")
        st.markdown("- Sleep: 8 hrs")
        st.markdown("- Screen: ≤3 hrs")
        st.markdown("- Stress: ≤4")

    with right:
        # ----- Check if today's entry already exists -----
        if not today_entry.empty:
            st.success("🎉 You have already submitted your entry for this date!")
            st.balloons()
            row = today_entry.iloc[-1]

            journal_html = ""
            if str(row.get("journal", "")).strip():
                journal_html = f"<div style='margin-top:6px; color:#cfcfcf; font-size:13px;'>Journal: {row.get('journal')}</div>"

            html = f"""
            <div style='background:#0b0b0b; color:white; padding:12px; border-radius:10px; margin-bottom:10px;'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                  <strong style='font-size:16px'>{row.get('username')}</strong>
                  <div style='font-size:12px; color:#bdbdbd'>{pd.to_datetime(row.get('date')).strftime('%b %d, %Y')}</div>
                </div>
                <div style='text-align:right;'>
                  <div style='font-size:20px; font-weight:bold; color:#ffd700'>{row.get('wellness_score')}</div>
                  <div style='font-size:12px; color:#bdbdbd'>Score</div>
                </div>
              </div>

              <div style='margin-top:8px; display:flex; gap:8px;'>
                <div style='background:#111; padding:8px; border-radius:8px; min-width:80px; text-align:center;'>
                  Stress<br><strong>{row.get('stress_level')}</strong>
                </div>
                <div style='background:#111; padding:8px; border-radius:8px; min-width:80px; text-align:center;'>
                  Screen<br><strong>{row.get('screen_time')}h</strong>
                </div>
                <div style='background:#111; padding:8px; border-radius:8px; min-width:80px; text-align:center;'>
                  Sleep<br><strong>{row.get('sleep_hours')}h</strong>
                </div>
                <div style='background:#111; padding:8px; border-radius:8px; min-width:80px; text-align:center;'>
                  Mood<br><strong>{row.get('mood')}</strong>
                </div>
              </div>

              <div style='margin-top:8px; color:#e0e0e0; font-size:13px;'>Tip: {row.get('tip','—')}</div>
              {journal_html}
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

        # ----- New entry submission form -----
        else:
            sleep_hours = st.number_input("Sleep Hours (0-12)", min_value=0, max_value=12, value=8, step=1)
            screen_time = st.number_input("Screen Time (hrs, 0-24)", min_value=0, max_value=24, value=3, step=1)
            stress_level = st.slider("Stress Level (0-10)", min_value=0, max_value=10, value=5)
            mood = st.selectbox("Mood", ["Happy", "Tired", "Sad", "Anxious", "Stressed"])
            journal = st.text_area("Journal / Notes", value="")

            if st.button("Submit Today's Check-in"):
                score = compute_wellness_score(sleep_hours, screen_time, stress_level)
                tip = generate_tip(sleep_hours, screen_time, stress_level, mood)
                entry = {
                    "username": st.session_state.username,
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
                    st.success("✅ Today's check-in saved!")
                    st.balloons()
                    play_beep()
                    st.session_state.checkin_done = True
                    st.rerun()
                else:
                    st.warning("⚠ You've already submitted for this date. Showing your previous entry below.")
                    st.balloons()
                    st.rerun()


# ---------- WEEKLY OVERVIEW ----------
elif st.session_state.page == "Weekly Overview":
    st.header("📊 Weekly Overview (Last 7 Days)")
    data = load_data()
    last7 = get_last_n_days(data, 7, st.session_state.username)
    if last7.empty:
        st.info("No entries for weekly overview.")
    else:
        for _, r in last7.iterrows():
            st.markdown(f"### {pd.to_datetime(r['date']).strftime('%B %d, %Y')}")
            a, b, c, d = st.columns(4)
            with a:
                render_card("Stress", r["stress_level"], color="#FF4B4B", emoji="😣")
            with b:
                render_card("Screen (hrs)", r["screen_time"], color="#FFA500", emoji="📱")
            with c:
                render_card("Sleep (hrs)", r["sleep_hours"], color="#1E90FF", emoji="🛌")
            with d:
                render_card("Score", r["wellness_score"], color="#4CAF50", emoji="🌿")

        melt = last7.melt(id_vars="date", value_vars=["stress_level", "screen_time", "sleep_hours", "wellness_score"],
                          var_name="Metric", value_name="Value")
        fig = px.line(melt, x=melt["date"].dt.strftime('%b %d'), y="Value", color="Metric", markers=True,
                      color_discrete_map={"stress_level": "red", "screen_time": "orange",
                                          "sleep_hours": "blue", "wellness_score": "green"})
        fig.update_layout(title="📈 Weekly Trend", yaxis_title="Level / Hours / Score",
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)


# ---------- LEADERBOARD ----------
elif st.session_state.page == "Leaderboard":
    st.header("🏆 Leaderboard")
    data = load_data()
    if data.empty:
        st.info("No data yet.")
    else:
        today = pd.Timestamp(datetime.now().date())
        board_type = st.selectbox("Daily or Weekly:", ["Daily", "Weekly"], key="board_type")
        if board_type == "Daily":
            df_today = data[data["date"] == today]
            df_rank = df_today.groupby("username", as_index=False)["wellness_score"].mean()
        else:
            week_ago = today - pd.Timedelta(days=6)
            df_week = data[(data["date"] >= week_ago) & (data["date"] <= today)]
            df_rank = df_week.groupby("username", as_index=False)["wellness_score"].mean()

        if df_rank.empty:
            st.info("No leaderboard records.")
        else:
            df_rank = df_rank.sort_values("wellness_score", ascending=False).reset_index(drop=True)
            df_rank["Rank"] = df_rank.index + 1
            df_rank["Medal"] = df_rank["Rank"].apply(lambda r: ["🥇", "🥈", "🥉"][r - 1] if r <= 3 else "")
            for _, row in df_rank.head(20).iterrows():
                rc = "gold" if row["Rank"] == 1 else ("silver" if row["Rank"] == 2 else (
                    "#cd7f32" if row["Rank"] == 3 else "white"))
                st.markdown(
                    f"<div style='background:#111; padding:12px; border-radius:10px; margin-bottom:8px; color:white;'>"
                    f"<h4 style='margin:0;'>Rank: <span style='color:{rc}; font-weight:bold'>{row['Rank']}</span> {row['Medal']}</h4>"
                    f"<p style='margin:4px 0 0 0;'>User: {row['username']} | Score: {row['wellness_score']:.1f}</p>"
                    "</div>",
                    unsafe_allow_html=True
                )


# ---------- VIEW PAST ENTRIES ----------
elif st.session_state.page == "View Past Entries":
    st.header("📜 Past Entries (card view)")
    data = load_data()
    if data.empty:
        st.info("No entries.")
    else:
        users = ["All Users"] + sorted(data["username"].dropna().unique().tolist())
        sel = st.selectbox("Filter by user:", users,
                           index=users.index(st.session_state.username) if st.session_state.username in users else 0)
        period = st.selectbox("Period:", ["All", "Daily", "Weekly"], index=0, key="past_period")
        if sel == "All Users":
            display = data.copy()
        else:
            display = data[data["username"] == sel].copy()

        today = pd.Timestamp(datetime.now().date())
        if period == "Daily":
            display = display[display["date"] == today]
        elif period == "Weekly":
            week_ago = today - pd.Timedelta(days=6)
            display = display[(display["date"] >= week_ago) & (display["date"] <= today)]

        display = display.sort_values(["wellness_score"], ascending=False).reset_index(drop=True)

        for _, r in display.iterrows():
            uname = r.get("username", "")
            dstr = pd.to_datetime(r.get("date")).strftime("%b %d, %Y")
            score = r.get("wellness_score", "")
            stress = r.get("stress_level", "")
            screen = r.get("screen_time", "")
            sleep = r.get("sleep_hours", "")
            mood = r.get("mood", "")
            tip = r.get("tip", "")
            journal = r.get("journal", "")
            journal_html = f"<div style='margin-top:6px; color:#cfcfcf; font-size:13px;'>Journal: {journal}</div>" if str(
                journal).strip() else ""
            html = f"""
            <div style='background:#0b0b0b; color:white; padding:12px; border-radius:10px; margin-bottom:10px;'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                  <strong style='font-size:16px'>{uname}</strong>
                  <div style='font-size:12px; color:#bdbdbd'>{dstr}</div>
                </div>
                <div style='text-align:right;'>
                  <div style='font-size:20px; font-weight:bold; color:#ffd700'>{score}</div>
                  <div style='font-size:12px; color:#bdbdbd'>Score</div>
                </div>
              </div>
              <div style='margin-top:8px; display:flex; gap:8px;'>
                <div style='background:#111; padding:8px; border-radius:8px; min-width:80px; text-align:center;'>Stress<br><strong>{stress}</strong></div>
                <div style='background:#111; padding:8px; border-radius:8px; min-width:80px; text-align:center;'>Screen<br><strong>{screen}h</strong></div>
                <div style='background:#111; padding:8px; border-radius:8px; min-width:80px; text-align:center;'>Sleep<br><strong>{sleep}h</strong></div>
                <div style='background:#111; padding:8px; border-radius:8px; min-width:80px; text-align:center;'>Mood<br><strong>{mood}</strong></div>
              </div>
              <div style='margin-top:8px; color:#e0e0e0; font-size:13px;'>Tip: {tip if tip else '—'}</div>
              {journal_html}
            </div>"""
            st.markdown(html, unsafe_allow_html=True)


# --------------- Clear My Past Entries ---------------
elif st.session_state.page == "Clear My Past Entries":
    st.header("🧹 Clear My Past Entries")

    # Two-step confirmation flag
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = False

    if not st.session_state.confirm_delete:
        if st.button("⚠ Delete only my entries"):
            st.session_state.confirm_delete = True
            st.rerun()  # show confirmation UI immediately
    else:
        st.warning("Are you sure? This will permanently delete ALL entries for your username.")
        c1, c2 = st.columns([1, 1])

        with c1:
            if st.button("✅ Yes — Delete my entries"):
                df_all = load_data()
                before = df_all.shape[0]
                df_all = df_all[df_all["username"] != username]
                save_data(df_all)
                after = df_all.shape[0]
                deleted = before - after
                st.success(f"✅ Deleted {deleted} entries for '{username}'.")
                st.balloons()
                time.sleep(2)
                st.session_state.confirm_delete = False
                st.rerun()

        with c2:
            if st.button("❌ Cancel"):
                st.session_state.confirm_delete = False
                st.info("Deletion canceled.")
                st.rerun()


# --------------- Edit / Delete Entries ---------------
elif st.session_state.page == "Edit / Delete Entries":
    st.header("✏ Edit or Delete Your Entries")

    data = load_data()
    df_user = data[data["username"] == username].sort_values("date", ascending=False).reset_index(drop=True)

    if df_user.empty:
        st.info("No records to edit for you.")
    else:
        edited = st.data_editor(df_user, num_rows="dynamic")

        if st.button("💾 Save my edits"):
            other = data[data["username"] != username]
            merged = pd.concat([other, edited], ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.normalize()
            merged = merged.drop_duplicates(subset=["username", "date"], keep="last")
            save_data(merged)
            st.success("✅ Changes saved successfully!")
            st.balloons()
            time.sleep(2)  # wait 2 seconds before refresh
            st.rerun()

        st.caption("📝 To delete a row: remove it in the editor and click 'Save my edits'.")


# ----------- Switch Account / Exit -----------
elif st.session_state.page == "Switch Account":
    # clear session fully so next user gets a clean start
    for k in list(st.session_state.keys()):
        del st.session_state[k]

    # reinitialize defaults then rerun (ensures login page is shown)
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.date_input = pd.Timestamp(datetime.now().date()).normalize()
    st.session_state.page = "Today's Check-in"
    st.session_state.checkin_done = False
    st.rerun()

elif st.session_state.page == "Exit App":
    st.write("👋 Thanks — close the tab to exit.")
    st.stop()