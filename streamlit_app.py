import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from textblob import TextBlob

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="DSAT Intelligence", layout="wide", page_icon="🧠")

# -------------------------------
# CUSTOM CSS — Professional Dark Theme
# -------------------------------
st.markdown("""
<style>
    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #0f1117;
        color: #e8eaf0;
    }

    /* ── Title ── */
    h1 { color: #ffffff; font-size: 2rem; font-weight: 700; }
    h2, h3 { color: #c9d1e8; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #1a1f2e;
        border: 1px solid #2e3555;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] { color: #7eb8f7; font-size: 1.7rem; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 0.85rem; }
    [data-testid="stMetricLabel"] { color: #8b92ab; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }

    /* ── Section divider ── */
    .section-header {
        background: linear-gradient(90deg, #1e2540, #131826);
        border-left: 4px solid #4f7bf7;
        padding: 10px 18px;
        border-radius: 6px;
        margin: 24px 0 12px 0;
        font-size: 1.05rem;
        font-weight: 600;
        color: #c9d1e8;
    }

    /* ── Morning brief card ── */
    .morning-brief {
        background: linear-gradient(135deg, #1a2744, #1a1f2e);
        border: 1px solid #3a4870;
        border-radius: 14px;
        padding: 20px 26px;
        margin-bottom: 20px;
    }
    .brief-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #7eb8f7;
        margin-bottom: 8px;
    }
    .brief-body { font-size: 1.05rem; color: #dce3f5; line-height: 1.7; }

    /* ── 2×2 matrix cell ── */
    .matrix-cell {
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    .cell-red    { background: #2c1414; border: 1px solid #7f2222; }
    .cell-orange { background: #2c1f10; border: 1px solid #8c5a1a; }
    .cell-yellow { background: #252412; border: 1px solid #7a7020; }
    .cell-green  { background: #102414; border: 1px solid #1e6e2a; }

    /* ── Insight box ── */
    .insight-box {
        background: #141926;
        border: 1px solid #2e3555;
        border-radius: 12px;
        padding: 20px 24px;
        line-height: 1.8;
    }
    .insight-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .tag-critical { background: #4a1010; color: #f87272; }
    .tag-watchlist { background: #3a2b0a; color: #fbbf24; }
    .tag-strong   { background: #0e2e1a; color: #34d399; }

    /* ── Coaching card ── */
    .coaching-card {
        background: #141d2e;
        border: 1px solid #2c4a80;
        border-radius: 12px;
        padding: 18px 22px;
    }
    .coaching-item {
        padding: 8px 0;
        border-bottom: 1px solid #1e2a40;
        color: #c9d1e8;
        font-size: 0.92rem;
    }
    .coaching-item:last-child { border-bottom: none; }

    /* ── Risk badge ── */
    .risk-badge-high   { color: #f87272; font-weight: 700; }
    .risk-badge-medium { color: #fbbf24; font-weight: 700; }
    .risk-badge-low    { color: #34d399; font-weight: 700; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* ── Selectbox ── */
    .stSelectbox label { color: #8b92ab; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { background: #1a1f2e; border-radius: 10px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8b92ab; border-radius: 8px; padding: 6px 18px; }
    .stTabs [aria-selected="true"] { background: #2e3f6e !important; color: #ffffff !important; }

    /* ── Change indicator ── */
    .delta-up   { color: #f87272; }
    .delta-down { color: #34d399; }
    .delta-flat { color: #8b92ab; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("# 🧠 DSAT Intelligence Dashboard")
st.markdown("<p style='color:#8b92ab; margin-top:-14px; font-size:0.9rem;'>AI-powered agent performance monitoring · Prediction · Coaching</p>", unsafe_allow_html=True)

# -------------------------------
# SAFE CSV LOADER
# -------------------------------
file_path = "bpo_customer_experience_dataset.csv"

if not os.path.exists(file_path):
    st.error("❌ CSV file not found. Please place 'bpo_customer_experience_dataset.csv' in the same folder.")
    st.stop()

df = None
for encoding in ["utf-8", "latin1", "utf-16"]:
    for sep in [",", "|", ";", "\t"]:
        try:
            temp = pd.read_csv(file_path, encoding=encoding, sep=sep)
            if temp.shape[1] > 1:
                df = temp
                break
        except:
            continue
    if df is not None:
        break

if df is None or df.empty:
    st.error("❌ Failed to load dataset")
    st.stop()

df.columns = df.columns.str.strip()

# -------------------------------
# CLEANING
# -------------------------------
df['Week'] = pd.to_datetime(df['Week'], errors='coerce')
df = df.dropna(subset=['Week'])
df['DSAT'] = df['Customer_Effortless'].apply(lambda x: 1 if str(x).lower() == "no" else 0)
df['Sentiment'] = df['Customer_Comment'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# -------------------------------
# LABEL FUNCTION
# -------------------------------
def label_issue(comment):
    c = str(comment).lower()
    if any(w in c for w in ["rude", "bad", "angry", "unhelpful", "attitude", "unprofessional"]):
        return "Communication"
    elif any(w in c for w in ["delay", "wait", "slow", "long", "transfer", "hold"]):
        return "Process"
    elif any(w in c for w in ["error", "bug", "failed", "not working", "broken", "issue"]):
        return "Product"
    else:
        return "Other"

df['Issue_Label'] = df['Customer_Comment'].apply(label_issue)

# -------------------------------
# NLP MODEL
# -------------------------------
df_clean = df[df['Issue_Label'] != "Other"].copy()
nlp_accuracy = 0.0
vectorizer = None
issue_model = None

if len(df_clean) > 50:
    noise_ratio = 0.2
    noise_idx = np.random.choice(df_clean.index, int(len(df_clean) * noise_ratio), replace=False)
    df_clean.loc[noise_idx, 'Issue_Label'] = np.random.choice(["Communication", "Process", "Product"], len(noise_idx))

    train_df, test_df = train_test_split(df_clean, test_size=0.3, stratify=df_clean['Issue_Label'], random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    X_train = vectorizer.fit_transform(train_df['Customer_Comment'])
    y_train = train_df['Issue_Label']
    X_test = vectorizer.transform(test_df['Customer_Comment'])
    y_test = test_df['Issue_Label']

    issue_model = LogisticRegression(max_iter=200, C=0.5)
    issue_model.fit(X_train, y_train)
    y_pred = issue_model.predict(X_test)
    nlp_accuracy = accuracy_score(y_test, y_pred)

# -------------------------------
# DSAT MODEL
# -------------------------------
weekly_df = df.groupby(['Agent_Name', 'Week']).agg(
    DSAT_Count=('DSAT', 'sum'),
    Total_Tickets=('Ticket_ID', 'count')
).reset_index()

for i in range(1, 5):
    weekly_df[f'DSAT_lag_{i}'] = weekly_df.groupby('Agent_Name')['DSAT_Count'].shift(i)
    weekly_df[f'Tickets_lag_{i}'] = weekly_df.groupby('Agent_Name')['Total_Tickets'].shift(i)

weekly_df = weekly_df.dropna()
features = [f'DSAT_lag_{i}' for i in range(1, 5)] + [f'Tickets_lag_{i}' for i in range(1, 5)]

X = weekly_df[features]
y = weekly_df['DSAT_Count']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------------
# AGENT SUMMARY WITH RISK + DELTA
# -------------------------------
def compute_agent_summary(weekly_df, model, features, y):
    summary_rows = []
    for agent_name, grp in weekly_df.groupby('Agent_Name'):
        grp = grp.sort_values('Week')
        if len(grp) < 2:
            continue
        latest = grp.iloc[-1]
        prev = grp.iloc[-2]

        pred_latest = model.predict([latest[features]])[0]
        pred_prev = model.predict([prev[features]])[0]

        risk_latest = (pred_latest - y.mean()) / y.std() * 10 + 50
        risk_prev = (pred_prev - y.mean()) / y.std() * 10 + 50
        delta_risk = risk_latest - risk_prev

        avg_dsat = grp['DSAT_Count'].mean()
        trend = latest['DSAT_lag_1'] - latest['DSAT_lag_4']

        summary_rows.append({
            'Agent': agent_name,
            'Avg DSAT': round(avg_dsat, 1),
            'Predicted DSAT': int(pred_latest),
            'Risk Score': int(risk_latest),
            'Risk Δ': round(delta_risk, 1),
            'Trend': trend
        })
    return pd.DataFrame(summary_rows)

agent_summary_df = compute_agent_summary(weekly_df, model, features, y)

# ── Quadrant assignment
def assign_quadrant(row):
    high_dsat = row['Avg DSAT'] >= agent_summary_df['Avg DSAT'].median()
    worsening = row['Trend'] > 0
    if high_dsat and worsening:
        return "🔴 Intervene Now"
    elif not high_dsat and worsening:
        return "🟡 Watch Closely"
    elif high_dsat and not worsening:
        return "🟠 Coach & Monitor"
    else:
        return "🟢 Acknowledge"

agent_summary_df['Focus Zone'] = agent_summary_df.apply(assign_quadrant, axis=1)

# ── Week-over-week change label
def delta_label(d):
    if d > 2:
        return f"<span class='delta-up'>↑ +{d:.1f}</span>"
    elif d < -2:
        return f"<span class='delta-down'>↓ {d:.1f}</span>"
    else:
        return f"<span class='delta-flat'>→ {d:.1f}</span>"

# -------------------------------------------------------
# ⭐ SECTION 1 — MANAGER'S MORNING BRIEF
# -------------------------------------------------------
st.markdown('<div class="section-header">☀️ Manager\'s Morning Brief</div>', unsafe_allow_html=True)

n_critical = (agent_summary_df['Focus Zone'] == "🔴 Intervene Now").sum()
n_watch = (agent_summary_df['Focus Zone'] == "🟡 Watch Closely").sum()
n_worsening = (agent_summary_df['Risk Δ'] > 3).sum()
n_improving = (agent_summary_df['Risk Δ'] < -3).sum()
team_avg_dsat = round(agent_summary_df['Avg DSAT'].mean(), 1)
team_pred_dsat = agent_summary_df['Predicted DSAT'].sum()

top_critical = agent_summary_df[agent_summary_df['Focus Zone'] == "🔴 Intervene Now"]['Agent'].tolist()
top_critical_str = ", ".join(top_critical[:3]) + ("..." if len(top_critical) > 3 else "") if top_critical else "None"

brief_text = f"""
<b>{n_critical} agent(s) need immediate intervention</b> · {n_watch} on watchlist · {n_worsening} worsening vs last week · {n_improving} recovering<br>
📌 Immediate focus: <b>{top_critical_str}</b><br>
📊 Team's predicted DSAT this week: <b>{team_pred_dsat}</b> · Average DSAT per agent: <b>{team_avg_dsat}</b>
"""

st.markdown(f"""
<div class="morning-brief">
  <div class="brief-title">🗓️ Today's Situation · Auto-generated</div>
  <div class="brief-body">{brief_text}</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# ⭐ SECTION 2 — TEAM KPIs
# -------------------------------------------------------
st.markdown('<div class="section-header">📊 System Accuracy & Team Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Issue Detection Accuracy", f"{round(nlp_accuracy * 100, 1)}%")
c2.metric("Agents in Critical Zone", n_critical, delta=f"{n_critical} need action", delta_color="inverse")
c3.metric("Worsening This Week", n_worsening, delta_color="inverse")
c4.metric("Recovering This Week", n_improving, delta_color="normal")
c5.metric("Team Avg DSAT", team_avg_dsat)

# -------------------------------------------------------
# ⭐ SECTION 3 — 2×2 FOCUS MATRIX
# -------------------------------------------------------
st.markdown('<div class="section-header">🎯 Manager Focus Matrix</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#8b92ab; font-size:0.85rem; margin-top:-8px;'>Each agent is placed based on their DSAT level and weekly trend direction.</p>", unsafe_allow_html=True)

quadrant_agents = {q: agent_summary_df[agent_summary_df['Focus Zone'] == q]['Agent'].tolist()
                   for q in ["🔴 Intervene Now", "🟡 Watch Closely", "🟠 Coach & Monitor", "🟢 Acknowledge"]}

col1, col2 = st.columns(2)

with col1:
    names_red = "<br>".join([f"• {a}" for a in quadrant_agents["🔴 Intervene Now"]]) or "None"
    st.markdown(f"""
    <div class="matrix-cell cell-red">
        <b style='color:#f87272'>🔴 Intervene Now</b><br>
        <span style='color:#8b92ab; font-size:0.75rem'>High DSAT · Worsening</span><br><br>
        {names_red}
    </div>""", unsafe_allow_html=True)

    names_orange = "<br>".join([f"• {a}" for a in quadrant_agents["🟠 Coach & Monitor"]]) or "None"
    st.markdown(f"""
    <div class="matrix-cell cell-orange" style="margin-top:10px">
        <b style='color:#fb923c'>🟠 Coach & Monitor</b><br>
        <span style='color:#8b92ab; font-size:0.75rem'>High DSAT · Improving</span><br><br>
        {names_orange}
    </div>""", unsafe_allow_html=True)

with col2:
    names_yellow = "<br>".join([f"• {a}" for a in quadrant_agents["🟡 Watch Closely"]]) or "None"
    st.markdown(f"""
    <div class="matrix-cell cell-yellow">
        <b style='color:#facc15'>🟡 Watch Closely</b><br>
        <span style='color:#8b92ab; font-size:0.75rem'>Low DSAT · Worsening</span><br><br>
        {names_yellow}
    </div>""", unsafe_allow_html=True)

    names_green = "<br>".join([f"• {a}" for a in quadrant_agents["🟢 Acknowledge"]]) or "None"
    st.markdown(f"""
    <div class="matrix-cell cell-green" style="margin-top:10px">
        <b style='color:#34d399'>🟢 Acknowledge</b><br>
        <span style='color:#8b92ab; font-size:0.75rem'>Low DSAT · Stable/Improving</span><br><br>
        {names_green}
    </div>""", unsafe_allow_html=True)

# -------------------------------------------------------
# ⭐ SECTION 4 — ALERTS TABLE (with delta)
# -------------------------------------------------------
st.markdown('<div class="section-header">🚨 Risk Leaderboard</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔴 Top 10 High Risk", "🟢 Top 10 Low Risk"])

display_cols = ['Agent', 'Avg DSAT', 'Predicted DSAT', 'Risk Score', 'Risk Δ', 'Focus Zone']

with tab1:
    high_risk = agent_summary_df.sort_values('Risk Score', ascending=False).head(10)[display_cols]
    st.dataframe(high_risk, use_container_width=True, hide_index=True)

with tab2:
    low_risk = agent_summary_df.sort_values('Risk Score').head(10)[display_cols]
    st.dataframe(low_risk, use_container_width=True, hide_index=True)

# -------------------------------------------------------
# ⭐ SECTION 5 — AGENT DEEP DIVE
# -------------------------------------------------------
st.markdown('<div class="section-header">🔍 Agent Deep Dive</div>', unsafe_allow_html=True)

agent = st.selectbox("Select Agent", sorted(weekly_df['Agent_Name'].unique()))

agent_data = weekly_df[weekly_df['Agent_Name'] == agent].sort_values('Week')
latest = agent_data.iloc[-1]
prev = agent_data.iloc[-2] if len(agent_data) >= 2 else latest

prediction = model.predict([latest[features]])[0]
risk = (prediction - y.mean()) / y.std() * 10 + 50

prev_pred = model.predict([prev[features]])[0]
prev_risk = (prev_pred - y.mean()) / y.std() * 10 + 50
risk_delta = risk - prev_risk

trend = latest['DSAT_lag_1'] - latest['DSAT_lag_4']

agent_actual = agent_data['DSAT_Count'].values
agent_pred_vals = model.predict(agent_data[features])
agent_mae = mean_absolute_error(agent_actual, agent_pred_vals)

agent_variance = np.var(agent_actual)
agent_error_var = np.var(agent_actual - agent_pred_vals)
agent_r2 = 1 - (agent_error_var / agent_variance) if agent_variance != 0 else 0

# Agent KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Predicted DSAT", int(prediction), delta=f"{int(prediction - prev_pred):+d} vs last week", delta_color="inverse")
k2.metric("Risk Score", int(risk), delta=f"{risk_delta:+.1f} vs last week", delta_color="inverse")
k3.metric("Prediction Reliability (R²)", round(agent_r2, 2))
k4.metric("Avg Prediction Error", round(agent_mae, 2))

# DSAT trend chart
st.markdown("**📈 DSAT Trend Over Time**")
chart_data = agent_data.set_index('Week')[['DSAT_Count']].copy()
chart_data.columns = ['DSAT Count']
st.line_chart(chart_data)

# -------------------------------------------------------
# ⭐ SECTION 6 — ISSUE BREAKDOWN (with top phrases)
# -------------------------------------------------------
st.markdown('<div class="section-header">🔍 Issue Breakdown</div>', unsafe_allow_html=True)

agent_comments = df[df['Agent_Name'] == agent]
dsat_comments = agent_comments[agent_comments['DSAT'] == 1]['Customer_Comment']
issues = ["Communication", "Process", "Product"]

if len(dsat_comments) > 0 and vectorizer is not None and issue_model is not None:
    X_test_agent = vectorizer.transform(dsat_comments)
    pred_issues = issue_model.predict(X_test_agent)
    issue_counts = pd.Series(pred_issues).value_counts()
    issue_df = issue_counts.reset_index()
    issue_df.columns = ["Issue", "Count"]
    for i in issues:
        if i not in issue_df["Issue"].values:
            issue_df = pd.concat([issue_df, pd.DataFrame({"Issue": [i], "Count": [0]})])
else:
    issue_df = pd.DataFrame({"Issue": issues, "Count": [0, 0, 0]})

issue_df = issue_df.sort_values("Count", ascending=False).reset_index(drop=True)
top_issue = issue_df.iloc[0]["Issue"]
total_issues = issue_df["Count"].sum()

ic1, ic2 = st.columns([1, 2])
with ic1:
    st.dataframe(issue_df, use_container_width=True, hide_index=True)
with ic2:
    st.bar_chart(issue_df.set_index("Issue")["Count"])

# Top phrases per issue (from raw label function)
st.markdown(f"**🔑 What's driving `{top_issue}` issues for this agent?**")
issue_keyword_map = {
    "Communication": ["rude", "unhelpful", "attitude", "angry", "bad", "unprofessional"],
    "Process": ["delay", "wait", "slow", "long hold", "transfer", "multiple transfers"],
    "Product": ["error", "bug", "failed", "not working", "broken", "recurring issue"]
}
phrases = issue_keyword_map.get(top_issue, [])
matched = []
for comment in dsat_comments:
    for phrase in phrases:
        if phrase in str(comment).lower():
            matched.append(phrase)
            break

if matched:
    from collections import Counter
    phrase_counts = Counter(matched).most_common(5)
    phrase_df = pd.DataFrame(phrase_counts, columns=["Keyword", "Occurrences"])
    st.dataframe(phrase_df, use_container_width=True, hide_index=True)
else:
    st.info("No specific keyword patterns detected for this agent.")

# -------------------------------------------------------
# ⭐ SECTION 7 — AI INSIGHT (narrative style)
# -------------------------------------------------------
def generate_ai_insight(agent, pred, risk, issue_df, sentiment, trend, risk_delta):
    top_issue = issue_df.sort_values("Count", ascending=False).iloc[0]["Issue"]
    total_issues = int(issue_df["Count"].sum())

    if risk < 45:
        level = "Strong Performer"
        tag_class = "tag-strong"
    elif risk < 60:
        level = "Watchlist"
        tag_class = "tag-watchlist"
    else:
        level = "Critical"
        tag_class = "tag-critical"

    risk_dir = "worsening" if risk_delta > 2 else ("improving" if risk_delta < -2 else "stable")

    if trend > 0:
        trend_msg = "DSAT has been rising week-over-week — this agent is on a declining performance path."
        sentiment_msg = "Customer sentiment is turning negative in line with the DSAT increase."
    elif trend < 0:
        trend_msg = "DSAT is decreasing — there are signs of performance recovery."
        sentiment_msg = "Customer sentiment appears to be improving alongside performance."
    else:
        trend_msg = "DSAT has been stable with no significant movement recently."
        sentiment_msg = "Customer sentiment is relatively flat."

    coaching_map = {
        ("up", "Communication"): [
            "🗣️ Run empathy and active listening workshops",
            "🎧 Review recent call recordings for tone and language",
            "📋 Introduce post-call customer confirmation checklist",
        ],
        ("up", "Process"): [
            "⏱️ Audit average hold/transfer time for this agent",
            "🔁 Train on first-call resolution techniques",
            "📞 Reduce unnecessary escalations by strengthening product knowledge",
        ],
        ("up", "Product"): [
            "📚 Refresh product and system training",
            "🐛 Create a recurring-issue escalation protocol",
            "🤝 Pair with a high-performer for knowledge shadowing",
        ],
    }

    direction = "up" if trend > 0 else ("down" if trend < 0 else "flat")
    coaching_items = coaching_map.get((direction, top_issue), [
        "✅ Maintain current performance",
        "📊 Monitor trends weekly",
        "🌟 Share best practices with the team",
    ])

    coaching_html = "".join([f"<div class='coaching-item'>{item}</div>" for item in coaching_items])

    return f"""
<div class="insight-box">
  <div><span class='insight-tag {tag_class}'>{level}</span></div>
  <b style='font-size:1.05rem'>{agent}</b><br><br>
  <b>📊 Performance Summary</b><br>
  Predicted DSAT this week: <b>{int(pred)}</b> &nbsp;|&nbsp; Risk Score: <b>{int(risk)}</b> ({risk_dir})<br>
  {trend_msg}<br>
  {sentiment_msg}<br><br>
  <b>🔍 Root Cause</b><br>
  The dominant issue is <b>{top_issue}</b>, accounting for most of the {total_issues} flagged complaints.
  {"This is a pattern that typically worsens without direct coaching intervention." if trend > 0 else "Improvements in this area are contributing to the recovery."}<br><br>
  <b>💡 Recommended Coaching Actions</b>
  <div class="coaching-card" style="margin-top:8px">{coaching_html}</div>
</div>
"""

sentiment_val = agent_comments['Sentiment'].mean()

st.markdown('<div class="section-header">🤖 AI Insight & Coaching Plan</div>', unsafe_allow_html=True)
st.markdown(
    generate_ai_insight(agent, prediction, risk, issue_df, sentiment_val, trend, risk_delta),
    unsafe_allow_html=True
)

# -------------------------------------------------------
# ⭐ SECTION 8 — WHAT CHANGED THIS WEEK
# -------------------------------------------------------
st.markdown('<div class="section-header">🔄 What Changed This Week?</div>', unsafe_allow_html=True)

new_critical = agent_summary_df[
    (agent_summary_df['Focus Zone'] == "🔴 Intervene Now") &
    (agent_summary_df['Risk Δ'] > 5)
]['Agent'].tolist()

newly_improved = agent_summary_df[
    (agent_summary_df['Risk Δ'] < -5)
]['Agent'].tolist()

dominant_issue_overall = df[df['DSAT'] == 1]['Issue_Label'].value_counts().idxmax() if len(df[df['DSAT'] == 1]) > 0 else "N/A"

change_lines = []
if new_critical:
    change_lines.append(f"🔴 <b>{len(new_critical)} agent(s)</b> newly entered the high-risk zone: {', '.join(new_critical[:3])}")
if newly_improved:
    change_lines.append(f"🟢 <b>{len(newly_improved)} agent(s)</b> showed significant improvement this week: {', '.join(newly_improved[:3])}")
change_lines.append(f"📌 Team's dominant complaint category this week: <b>{dominant_issue_overall}</b>")
change_lines.append(f"📊 Team-wide predicted DSAT total: <b>{team_pred_dsat}</b>")

change_html = "<br>".join(change_lines)
st.markdown(f"""
<div class="morning-brief">
  <div class="brief-title">📅 Weekly Delta Summary</div>
  <div class="brief-body">{change_html}</div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='color:#3a4060; font-size:0.75rem; text-align:center;'>DSAT Intelligence Dashboard · Powered by Random Forest + NLP · Auto-refreshes on data update</p>", unsafe_allow_html=True)
