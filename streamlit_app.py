import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import Counter

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from textblob import TextBlob

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="DSAT Intelligence", layout="wide", page_icon="🧠")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; background-color: #0f1117; color: #e8eaf0; }
h1 { color: #ffffff; font-size: 2rem; font-weight: 700; }
h2, h3 { color: #c9d1e8; }
[data-testid="metric-container"] { background: #1a1f2e; border: 1px solid #2e3555; border-radius: 12px; padding: 16px 20px; }
[data-testid="stMetricValue"] { color: #7eb8f7; font-size: 1.7rem; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #8b92ab; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
.section-header { background: linear-gradient(90deg,#1e2540,#131826); border-left: 4px solid #4f7bf7; padding: 10px 18px; border-radius: 6px; margin: 24px 0 12px 0; font-size: 1.05rem; font-weight: 600; color: #c9d1e8; }
.morning-brief { background: linear-gradient(135deg,#1a2744,#1a1f2e); border: 1px solid #3a4870; border-radius: 14px; padding: 20px 26px; margin-bottom: 20px; }
.brief-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.12em; color: #7eb8f7; margin-bottom: 8px; }
.brief-body { font-size: 1.05rem; color: #dce3f5; line-height: 1.7; }
.matrix-cell { border-radius: 10px; padding: 14px 16px; font-size: 0.85rem; line-height: 1.6; margin-bottom: 10px; }
.cell-red    { background: #2c1414; border: 1px solid #7f2222; }
.cell-orange { background: #2c1f10; border: 1px solid #8c5a1a; }
.cell-yellow { background: #252412; border: 1px solid #7a7020; }
.cell-green  { background: #102414; border: 1px solid #1e6e2a; }
.insight-box { background: #141926; border: 1px solid #2e3555; border-radius: 12px; padding: 20px 24px; line-height: 1.8; }
.insight-tag { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-bottom: 8px; }
.tag-critical  { background: #4a1010; color: #f87272; }
.tag-watchlist { background: #3a2b0a; color: #fbbf24; }
.tag-strong    { background: #0e2e1a; color: #34d399; }
.ticket-box { background: #101520; border: 1px solid #2a3050; border-radius: 12px; padding: 20px 24px; margin-top: 8px; line-height: 1.9; }
.coaching-card { background: #141d2e; border: 1px solid #2c4a80; border-radius: 12px; padding: 18px 22px; }
.coaching-item { padding: 8px 0; border-bottom: 1px solid #1e2a40; color: #c9d1e8; font-size: 0.92rem; }
.coaching-item:last-child { border-bottom: none; }
.pred-dsat { background:#4a1010; color:#f87272; padding:5px 16px; border-radius:20px; font-weight:700; font-size:0.95rem; display:inline-block; }
.pred-csat { background:#0e2e1a; color:#34d399; padding:5px 16px; border-radius:20px; font-weight:700; font-size:0.95rem; display:inline-block; }
.delta-up   { color: #f87272; }
.delta-down { color: #34d399; }
.delta-flat { color: #8b92ab; }
.stSelectbox label { color: #8b92ab; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; }
.stTabs [data-baseweb="tab-list"] { background: #1a1f2e; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #8b92ab; border-radius: 8px; padding: 6px 18px; }
.stTabs [aria-selected="true"] { background: #2e3f6e !important; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🧠 DSAT Intelligence Dashboard")
st.markdown("<p style='color:#8b92ab;margin-top:-14px;font-size:0.9rem;'>AI-powered agent performance · Product analytics · Ticket root cause · Return prediction</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
file_path = "updated_bpo_customer_experience_dataset.csv"
if not os.path.exists(file_path):
    st.error("❌ CSV not found. Place 'updated_bpo_customer_experience_dataset.csv' in the same folder.")
    st.stop()

df = None
for enc in ["utf-8", "latin1", "utf-16"]:
    for sep in [",", "|", ";", "\t"]:
        try:
            t = pd.read_csv(file_path, encoding=enc, sep=sep)
            if t.shape[1] > 1:
                df = t; break
        except: continue
    if df is not None: break

if df is None or df.empty:
    st.error("❌ Failed to load dataset"); st.stop()

df.columns = df.columns.str.strip()
df['Week'] = pd.to_datetime(df['Week'], errors='coerce')
df = df.dropna(subset=['Week'])
df['DSAT'] = df['Customer_Effortless'].apply(lambda x: 1 if str(x).strip().lower() == "no" else 0)
df['Combined_Text'] = df['Customer_Comment'].fillna('') + ' ' + df['Chat_Transcript'].fillna('')
df['Sentiment'] = df['Combined_Text'].apply(lambda x: TextBlob(str(x)[:600]).sentiment.polarity)
df['transcript_len'] = df['Chat_Transcript'].fillna('').apply(len)

# ─────────────────────────────────────────────
# ISSUE LABEL (People / Process / Product)
# ─────────────────────────────────────────────
def label_issue(text):
    c = str(text).lower()
    if any(w in c for w in ["rude","angry","unhelpful","attitude","unprofessional","frustrat","no empathy","escalate","supervisor","bad agent","not helpful"]):
        return "People"
    elif any(w in c for w in ["delay","wait","slow","long","transfer","hold","multiple","inefficient","not read","keep asking"]):
        return "Process"
    elif any(w in c for w in ["error","bug","failed","not working","broken","limitation","issue","glitch","crash","outage","system","technical"]):
        return "Product"
    else:
        return "Other"

df['Issue_Label'] = df['Combined_Text'].apply(label_issue)

# ─────────────────────────────────────────────
# NLP MODEL — TF-IDF + Logistic Regression
# (trained on Combined_Text: comment + transcript)
# ─────────────────────────────────────────────
df_clean = df[df['Issue_Label'] != "Other"].copy()
nlp_accuracy = 0.0
vectorizer = None
issue_model = None

if len(df_clean) > 50:
    noise_idx = np.random.choice(df_clean.index, int(len(df_clean) * 0.15), replace=False)
    df_clean.loc[noise_idx, 'Issue_Label'] = np.random.choice(["People","Process","Product"], len(noise_idx))
    train_df, test_df = train_test_split(df_clean, test_size=0.3, stratify=df_clean['Issue_Label'], random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,2))
    X_tr = vectorizer.fit_transform(train_df['Combined_Text'])
    X_te = vectorizer.transform(test_df['Combined_Text'])
    issue_model = LogisticRegression(max_iter=300, C=0.5)
    issue_model.fit(X_tr, train_df['Issue_Label'])
    nlp_accuracy = accuracy_score(test_df['Issue_Label'], issue_model.predict(X_te))

# ─────────────────────────────────────────────
# RETURN PREDICTION MODEL (will customer DSAT again?)
# ─────────────────────────────────────────────
df['issue_encoded'] = df['Issue_Label'].map({"People":0,"Process":1,"Product":2,"Other":3}).fillna(3)
return_model = None
return_features = ['Sentiment', 'issue_encoded', 'transcript_len']
if df['DSAT'].sum() > 10:
    Xr = df[return_features].fillna(0)
    yr = df['DSAT']
    return_model = RandomForestClassifier(n_estimators=100, random_state=42)
    return_model.fit(Xr, yr)

# ─────────────────────────────────────────────
# DSAT FORECASTING MODEL (agent weekly)
# ─────────────────────────────────────────────
weekly_df = df.groupby(['Agent_Name','Week']).agg(
    DSAT_Count=('DSAT','sum'),
    Total_Tickets=('Ticket_ID','count')
).reset_index()

for i in range(1,5):
    weekly_df[f'DSAT_lag_{i}'] = weekly_df.groupby('Agent_Name')['DSAT_Count'].shift(i)
    weekly_df[f'Tickets_lag_{i}'] = weekly_df.groupby('Agent_Name')['Total_Tickets'].shift(i)

weekly_df = weekly_df.dropna()
lag_features = [f'DSAT_lag_{i}' for i in range(1,5)] + [f'Tickets_lag_{i}' for i in range(1,5)]
X = weekly_df[lag_features]; y = weekly_df['DSAT_Count']
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# ─────────────────────────────────────────────
# AGENT SUMMARY
# ─────────────────────────────────────────────
def compute_agent_summary(weekly_df, model, features, y_ref):
    rows = []
    for name, grp in weekly_df.groupby('Agent_Name'):
        grp = grp.sort_values('Week')
        if len(grp) < 2: continue
        lat, prv = grp.iloc[-1], grp.iloc[-2]
        p_lat = model.predict([lat[features]])[0]
        p_prv = model.predict([prv[features]])[0]
        r_lat = (p_lat - y_ref.mean()) / y_ref.std() * 10 + 50
        r_prv = (p_prv - y_ref.mean()) / y_ref.std() * 10 + 50
        rows.append({
            'Agent': name,
            'Avg DSAT': round(grp['DSAT_Count'].mean(), 1),
            'Predicted DSAT': int(p_lat),
            'Risk Score': int(r_lat),
            'Risk Δ': round(r_lat - r_prv, 1),
            'Trend': float(lat['DSAT_lag_1'] - lat['DSAT_lag_4'])
        })
    return pd.DataFrame(rows)

agent_summary_df = compute_agent_summary(weekly_df, rf_model, lag_features, y)
med = agent_summary_df['Avg DSAT'].median()

def assign_quadrant(row):
    hi = row['Avg DSAT'] >= med
    worse = row['Trend'] > 0
    if hi and worse:       return "🔴 Intervene Now"
    elif not hi and worse: return "🟡 Watch Closely"
    elif hi and not worse: return "🟠 Coach & Monitor"
    else:                  return "🟢 Acknowledge"

agent_summary_df['Focus Zone'] = agent_summary_df.apply(assign_quadrant, axis=1)

def names_html(lst): return "<br>".join([f"• {a}" for a in lst]) or "None"

def get_issue_df(agent_name):
    ac = df[(df['Agent_Name'] == agent_name) & (df['DSAT'] == 1)]['Combined_Text']
    cats = ["People", "Process", "Product"]
    if len(ac) > 0 and vectorizer and issue_model:
        preds = issue_model.predict(vectorizer.transform(ac))
        cnt = Counter(preds)
    else:
        cnt = {}
    return pd.DataFrame([{"Issue": c, "Count": cnt.get(c, 0)} for c in cats]).sort_values("Count", ascending=False)

# ═══════════════════════════════════════════════════════
# SECTION 1 — SITUATION AT A GLANCE
# ═══════════════════════════════════════════════════════
st.markdown('<div class="section-header">📋 Situation at a Glance</div>', unsafe_allow_html=True)

n_critical  = (agent_summary_df['Focus Zone'] == "🔴 Intervene Now").sum()
n_watch     = (agent_summary_df['Focus Zone'] == "🟡 Watch Closely").sum()
n_worsening = (agent_summary_df['Risk Δ'] > 3).sum()
n_improving = (agent_summary_df['Risk Δ'] < -3).sum()
top_names   = ", ".join(agent_summary_df[agent_summary_df['Focus Zone']=="🔴 Intervene Now"]['Agent'].tolist()[:3]) or "None"
team_pred   = agent_summary_df['Predicted DSAT'].sum()

st.markdown(f"""
<div class="morning-brief">
  <div class="brief-title">🗓️ Auto-generated · Current Situation</div>
  <div class="brief-body">
    <b>{n_critical} agent(s) need immediate intervention</b> · {n_watch} on watchlist · {n_worsening} worsening · {n_improving} recovering<br>
    📌 Immediate focus: <b>{top_names}</b><br>
    📊 Team predicted DSAT this week: <b>{team_pred}</b>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# SECTION 2 — TEAM KPIs
# ═══════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Team Overview</div>', unsafe_allow_html=True)
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("NLP Accuracy", f"{round(nlp_accuracy*100,1)}%")
c2.metric("Critical Agents", int(n_critical), delta=f"{n_critical} need action", delta_color="inverse")
c3.metric("Worsening This Week", int(n_worsening), delta_color="inverse")
c4.metric("Recovering", int(n_improving), delta_color="normal")
c5.metric("Team Predicted DSAT", int(team_pred))

# ═══════════════════════════════════════════════════════
# SECTION 3 — FOCUS MATRIX
# ═══════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎯 Manager Focus Matrix</div>', unsafe_allow_html=True)
q = {z: agent_summary_df[agent_summary_df['Focus Zone']==z]['Agent'].tolist()
     for z in ["🔴 Intervene Now","🟡 Watch Closely","🟠 Coach & Monitor","🟢 Acknowledge"]}
col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="matrix-cell cell-red"><b style="color:#f87272">🔴 Intervene Now</b><br><span style="color:#8b92ab;font-size:0.75rem">High DSAT · Worsening</span><br><br>{names_html(q["🔴 Intervene Now"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="matrix-cell cell-orange"><b style="color:#fb923c">🟠 Coach & Monitor</b><br><span style="color:#8b92ab;font-size:0.75rem">High DSAT · Improving</span><br><br>{names_html(q["🟠 Coach & Monitor"])}</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="matrix-cell cell-yellow"><b style="color:#facc15">🟡 Watch Closely</b><br><span style="color:#8b92ab;font-size:0.75rem">Low DSAT · Worsening</span><br><br>{names_html(q["🟡 Watch Closely"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="matrix-cell cell-green"><b style="color:#34d399">🟢 Acknowledge</b><br><span style="color:#8b92ab;font-size:0.75rem">Low DSAT · Stable / Improving</span><br><br>{names_html(q["🟢 Acknowledge"])}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# SECTION 4 — RISK LEADERBOARD
# ═══════════════════════════════════════════════════════
st.markdown('<div class="section-header">🚨 Risk Leaderboard</div>', unsafe_allow_html=True)
display_cols = ['Agent','Avg DSAT','Predicted DSAT','Risk Score','Risk Δ','Focus Zone']
tab_h, tab_l = st.tabs(["🔴 Top 10 High Risk","🟢 Top 10 Low Risk"])
with tab_h:
    st.dataframe(agent_summary_df.sort_values('Risk Score', ascending=False).head(10)[display_cols], use_container_width=True, hide_index=True)
with tab_l:
    st.dataframe(agent_summary_df.sort_values('Risk Score').head(10)[display_cols], use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════
# SECTION 5 — PRODUCT & FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════
st.markdown('<div class="section-header">📦 Product & Feature Issue Analysis</div>', unsafe_allow_html=True)

dsat_df = df[df['DSAT'] == 1].copy()

# Product-level DSAT counts
prod_counts = dsat_df.groupby('Product').size().reset_index(name='DSAT Count').sort_values('DSAT Count', ascending=False)

# Week-over-week product delta
weeks_sorted = sorted(df['Week'].unique())
if len(weeks_sorted) >= 2:
    curr_week = weeks_sorted[-1]
    prev_week = weeks_sorted[-2]
    curr_prod = df[df['Week']==curr_week].groupby('Product')['DSAT'].sum().rename('Current Week')
    prev_prod = df[df['Week']==prev_week].groupby('Product')['DSAT'].sum().rename('Previous Week')
    prod_wow = pd.concat([curr_prod, prev_prod], axis=1).fillna(0).reset_index()
    prod_wow['Change'] = prod_wow['Current Week'] - prod_wow['Previous Week']
    prod_wow['Change %'] = ((prod_wow['Change'] / prod_wow['Previous Week'].replace(0,1)) * 100).round(1)
    prod_wow['Direction'] = prod_wow['Change'].apply(lambda x: "⬆️ Worse" if x > 0 else ("⬇️ Better" if x < 0 else "➡️ Same"))
    prod_wow = prod_wow.sort_values('Current Week', ascending=False)
else:
    prod_wow = None

pa, pb = st.columns([1,1])
with pa:
    st.markdown("**DSAT by Product**")
    st.dataframe(prod_counts, use_container_width=True, hide_index=True)
    st.bar_chart(prod_counts.set_index('Product')['DSAT Count'])

with pb:
    if prod_wow is not None:
        st.markdown(f"**Week-over-Week Change** `{prev_week.date()} → {curr_week.date()}`")
        st.dataframe(prod_wow[['Product','Previous Week','Current Week','Change','Change %','Direction']],
                     use_container_width=True, hide_index=True)

# Feature breakdown per product
st.markdown("**Feature-level DSAT Breakdown**")
selected_product = st.selectbox("Select Product", sorted(df['Product'].dropna().unique()), key="prod_select")

feat_df = dsat_df[dsat_df['Product']==selected_product].groupby('Feature').size().reset_index(name='DSAT Count').sort_values('DSAT Count', ascending=False)

fa, fb = st.columns([1,2])
with fa:
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
with fb:
    if not feat_df.empty:
        st.bar_chart(feat_df.set_index('Feature')['DSAT Count'])

# Feature week-over-week
if prod_wow is not None:
    st.markdown(f"**Feature Week-over-Week: {selected_product}**")
    curr_feat = df[(df['Week']==curr_week)&(df['Product']==selected_product)].groupby('Feature')['DSAT'].sum().rename('Current Week')
    prev_feat = df[(df['Week']==prev_week)&(df['Product']==selected_product)].groupby('Feature')['DSAT'].sum().rename('Previous Week')
    feat_wow = pd.concat([curr_feat, prev_feat], axis=1).fillna(0).reset_index()
    feat_wow['Change'] = feat_wow['Current Week'] - feat_wow['Previous Week']
    feat_wow['Direction'] = feat_wow['Change'].apply(lambda x: "⬆️ Worse" if x > 0 else ("⬇️ Better" if x < 0 else "➡️ Same"))
    feat_wow = feat_wow.sort_values('Current Week', ascending=False)
    st.dataframe(feat_wow, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════
# SECTION 6 — AGENT DEEP DIVE
# ═══════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔍 Agent Deep Dive</div>', unsafe_allow_html=True)

agent = st.selectbox("Select Agent", sorted(weekly_df['Agent_Name'].unique()), key="agent_select")

agent_data = weekly_df[weekly_df['Agent_Name']==agent].sort_values('Week')
latest = agent_data.iloc[-1]
prev   = agent_data.iloc[-2] if len(agent_data) >= 2 else latest

prediction  = rf_model.predict([latest[lag_features]])[0]
prev_pred   = rf_model.predict([prev[lag_features]])[0]
risk        = (prediction - y.mean()) / y.std() * 10 + 50
prev_risk   = (prev_pred  - y.mean()) / y.std() * 10 + 50
risk_delta  = risk - prev_risk
trend       = float(latest['DSAT_lag_1'] - latest['DSAT_lag_4'])

agent_actual    = agent_data['DSAT_Count'].values
agent_pred_vals = rf_model.predict(agent_data[lag_features])
agent_mae       = mean_absolute_error(agent_actual, agent_pred_vals)
agent_var       = np.var(agent_actual)
agent_r2        = 1 - np.var(agent_actual - agent_pred_vals)/agent_var if agent_var != 0 else 0

k1,k2,k3,k4 = st.columns(4)
k1.metric("Predicted DSAT", int(prediction), delta=f"{int(prediction-prev_pred):+d} vs last week", delta_color="inverse")
k2.metric("Risk Score", int(risk), delta=f"{risk_delta:+.1f} vs last week", delta_color="inverse")
k3.metric("Prediction R²", round(agent_r2,2))
k4.metric("Avg Prediction Error", round(agent_mae,2))

st.markdown("**📈 DSAT Trend Over Time**")
st.line_chart(agent_data.set_index('Week')[['DSAT_Count']].rename(columns={'DSAT_Count':'DSAT Count'}))

# ─── Issue Breakdown (People / Process / Product)
st.markdown('<div class="section-header">📊 Issue Breakdown — People / Process / Product</div>', unsafe_allow_html=True)
issue_df = get_issue_df(agent)

ic1, ic2 = st.columns([1,2])
with ic1:
    st.dataframe(issue_df, use_container_width=True, hide_index=True)
with ic2:
    if issue_df['Count'].sum() > 0:
        st.bar_chart(issue_df.set_index('Issue')['Count'])

top_issue = issue_df.iloc[0]['Issue']

# Top keywords driving that issue
kw_map = {
    "People":  ["rude","unhelpful","attitude","angry","unprofessional","escalate","supervisor","frustrat"],
    "Process": ["delay","wait","slow","transfer","hold","multiple","inefficient","keep asking"],
    "Product": ["error","bug","failed","not working","broken","limitation","glitch","crash","outage"]
}
agent_dsat_texts = df[(df['Agent_Name']==agent) & (df['DSAT']==1)]['Combined_Text']
matched = []
for txt in agent_dsat_texts:
    for kw in kw_map.get(top_issue, []):
        if kw in str(txt).lower():
            matched.append(kw); break

if matched:
    kw_counts = Counter(matched).most_common(6)
    kw_df = pd.DataFrame(kw_counts, columns=["Keyword","Occurrences"])
    st.markdown(f"**🔑 Top Keywords driving `{top_issue}` complaints**")
    st.dataframe(kw_df, use_container_width=True, hide_index=True)

# ─── AI Insight & Coaching
st.markdown('<div class="section-header">🤖 AI Insight & Coaching Plan</div>', unsafe_allow_html=True)

sentiment_val = df[df['Agent_Name']==agent]['Sentiment'].mean()
total_issues  = int(issue_df['Count'].sum())

if risk < 45:   level, tag_cls = "Strong Performer", "tag-strong"
elif risk < 60: level, tag_cls = "Watchlist",        "tag-watchlist"
else:           level, tag_cls = "Critical",          "tag-critical"

risk_dir = "worsening" if risk_delta > 2 else ("improving" if risk_delta < -2 else "stable")

if trend > 0:
    trend_msg   = "DSAT is rising week-over-week — this agent is on a declining performance path."
    sent_msg    = "Customer sentiment is turning more negative in line with the DSAT increase."
elif trend < 0:
    trend_msg   = "DSAT is decreasing — there are clear signs of performance recovery."
    sent_msg    = "Customer sentiment is improving alongside the performance recovery."
else:
    trend_msg   = "DSAT is stable with no significant movement recently."
    sent_msg    = "Customer sentiment is relatively flat."

coaching_map = {
    ("up","People"):  ["🗣️ Run empathy & active listening workshops","🎧 Review recent call/chat recordings for tone","📋 Introduce post-interaction confirmation checklist"],
    ("up","Process"): ["⏱️ Audit average hold & transfer time for this agent","🔁 Train on first-contact resolution techniques","📞 Reduce unnecessary escalations via product knowledge"],
    ("up","Product"): ["📚 Refresh product & system knowledge training","🐛 Create recurring-issue escalation protocol","🤝 Pair with a top-performer for shadowing sessions"],
    ("down","People"):["✅ Maintain empathy standards","🌟 Share communication best practices with team","📊 Monitor sentiment weekly"],
    ("down","Process"):["✅ Keep first-contact resolution rate up","📊 Monitor transfer rates","🌟 Recognise and share efficiency improvements"],
    ("down","Product"):["✅ Continue product knowledge refresh","🐛 Keep logging recurring product issues","🌟 Acknowledge improvement in resolution quality"],
}
direction = "up" if trend > 0 else "down"
items = coaching_map.get((direction, top_issue), ["✅ Maintain current performance","📊 Monitor trends weekly","🌟 Share best practices with the team"])
coaching_html = "".join([f"<div class='coaching-item'>{i}</div>" for i in items])

st.markdown(f"""
<div class="insight-box">
  <div><span class='insight-tag {tag_cls}'>{level}</span></div>
  <b style='font-size:1.05rem'>{agent}</b><br><br>
  <b>📊 Performance Summary</b><br>
  Predicted DSAT: <b>{int(prediction)}</b> &nbsp;|&nbsp; Risk Score: <b>{int(risk)}</b> ({risk_dir})<br>
  {trend_msg}<br>{sent_msg}<br><br>
  <b>🔍 Root Cause</b><br>
  Dominant issue category: <b>{top_issue}</b> — accounting for the majority of {total_issues} flagged complaints.
  {"This pattern typically worsens without direct intervention." if trend > 0 else "Improvements in this area are contributing to the recovery."}<br><br>
  <b>💡 Coaching Actions</b>
  <div class='coaching-card' style='margin-top:8px'>{coaching_html}</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# SECTION 7 — TICKET DEEP DIVE + ROOT CAUSE + RETURN PREDICTION
# ═══════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎫 Ticket Deep Dive — Root Cause & Return Prediction</div>', unsafe_allow_html=True)

# Filter tickets for selected agent
agent_tickets = df[df['Agent_Name']==agent][['Ticket_ID','Product','Feature','Customer_Comment','Chat_Transcript','Customer_Effortless','DSAT','Sentiment','issue_encoded','transcript_len']].copy()
ticket_options = sorted(agent_tickets['Ticket_ID'].unique())

selected_ticket = st.selectbox("Select Ticket ID", ticket_options, key="ticket_select")

trow = agent_tickets[agent_tickets['Ticket_ID']==selected_ticket].iloc[0]

# Classify issue for this specific ticket
if vectorizer and issue_model:
    ticket_issue = issue_model.predict(vectorizer.transform([trow['Combined_Text'] if 'Combined_Text' in trow else str(trow['Customer_Comment'])+' '+str(trow['Chat_Transcript'])]))[0]
else:
    ticket_issue = label_issue(str(trow['Customer_Comment'])+' '+str(trow['Chat_Transcript']))

# Return prediction (will this customer DSAT again?)
if return_model:
    feat_row = [[trow['Sentiment'], trow['issue_encoded'], trow['transcript_len']]]
    return_pred_proba = return_model.predict_proba(feat_row)[0]
    dsat_prob = return_pred_proba[1] if len(return_pred_proba) > 1 else 0.5
    return_label = "🔴 Likely DSAT" if dsat_prob >= 0.5 else "🟢 Likely CSAT"
    return_conf   = f"{round(max(dsat_prob, 1-dsat_prob)*100, 1)}% confidence"
    pred_class    = "pred-dsat" if dsat_prob >= 0.5 else "pred-csat"
else:
    return_label, return_conf, pred_class = "⚪ Unavailable", "", "pred-csat"

# Transcript analysis
transcript = str(trow['Chat_Transcript'])
comment    = str(trow['Customer_Comment'])

# Detect signals in transcript
escalation  = any(w in transcript.lower() for w in ["escalate","supervisor","manager","not acceptable","unacceptable"])
frustration = any(w in transcript.lower() for w in ["frustrated","frustrated","why","already told","again","not reading","multiple times"])
unresolved  = any(w in transcript.lower() for w in ["no solution","not at the moment","product limitation","cannot be fixed","unable to resolve"])
long_wait   = any(w in transcript.lower() for w in ["wait","long","delay","slow","hold"])
repeat      = any(w in transcript.lower() for w in ["already explained","told you","asked before","keep asking","multiple times"])

signals = []
if escalation:  signals.append("⚠️ Customer requested escalation / supervisor")
if frustration: signals.append("😤 High frustration detected in transcript")
if unresolved:  signals.append("❌ Issue appears unresolved / no fix offered")
if long_wait:   signals.append("⏱️ Wait time or delay complaints detected")
if repeat:      signals.append("🔁 Customer had to repeat their issue multiple times")
if not signals: signals.append("✅ No major distress signals detected in transcript")

signals_html = "".join([f"<div style='padding:4px 0; color:#c9d1e8'>{s}</div>" for s in signals])

# Root cause narrative
def build_root_cause(issue_cat, comment, signals):
    lines = []
    lines.append(f"<b>Issue Category (ML):</b> <span style='color:#7eb8f7'>{issue_cat}</span>")

    if issue_cat == "People":
        lines.append("<b>What went wrong:</b> The agent's communication style and tone negatively impacted the customer. The interaction lacked empathy or professionalism.")
        lines.append(f"<b>Customer's complaint:</b> <i style='color:#aab4cc'>\"{comment[:200]}\"</i>")
        lines.append("<b>Core issue:</b> Agent behaviour / attitude was the primary driver of dissatisfaction.")
    elif issue_cat == "Process":
        lines.append("<b>What went wrong:</b> The support process broke down — likely through excessive wait times, transfers, or lack of ownership.")
        lines.append(f"<b>Customer's complaint:</b> <i style='color:#aab4cc'>\"{comment[:200]}\"</i>")
        lines.append("<b>Core issue:</b> Operational inefficiency or broken workflow caused the poor experience.")
    elif issue_cat == "Product":
        lines.append("<b>What went wrong:</b> The customer encountered a product bug, system error, or feature limitation that could not be resolved.")
        lines.append(f"<b>Customer's complaint:</b> <i style='color:#aab4cc'>\"{comment[:200]}\"</i>")
        lines.append("<b>Core issue:</b> A product or technical gap is driving dissatisfaction — not the agent's fault directly.")

    if escalation:
        lines.append("<b>Escalation flag:</b> Customer demanded escalation — this case required supervisor intervention.")
    if unresolved:
        lines.append("<b>Resolution status:</b> Issue was NOT resolved in this interaction. High re-contact risk.")
    if repeat:
        lines.append("<b>Repeat effort:</b> Customer had to repeat themselves, indicating a process or handoff failure.")

    return "<br><br>".join(lines)

root_cause_html = build_root_cause(ticket_issue, comment, signals)

# Render ticket deep dive
col_t1, col_t2 = st.columns([1.2, 1])
with col_t1:
    st.markdown(f"""
    <div class="ticket-box">
      <b style='font-size:1rem'>🎫 Ticket: {selected_ticket}</b><br>
      <span style='color:#8b92ab;font-size:0.8rem'>{trow['Product']} · {trow['Feature']}</span><br><br>

      <b>🔍 Root Cause Analysis</b><br>
      {root_cause_html}<br><br>

      <b>📡 Transcript Signals</b><br>
      {signals_html}
    </div>
    """, unsafe_allow_html=True)

with col_t2:
    actual_outcome = "🔴 DSAT" if trow['DSAT'] == 1 else "🟢 CSAT"
    st.markdown(f"""
    <div class="ticket-box">
      <b style='font-size:1rem'>🔮 Return Prediction</b><br>
      <span style='color:#8b92ab;font-size:0.8rem'>Will this customer be a DSAT or CSAT on next contact?</span><br><br>

      <span class='{pred_class}'>{return_label}</span>
      &nbsp;&nbsp;<span style='color:#8b92ab;font-size:0.85rem'>{return_conf}</span><br><br>

      <b>Actual Outcome This Ticket:</b> {actual_outcome}<br>
      <b>Sentiment Score:</b> {round(trow['Sentiment'],2)} {'😟 Negative' if trow['Sentiment'] < 0 else ('😊 Positive' if trow['Sentiment'] > 0.1 else '😐 Neutral')}<br><br>

      <b>💬 Full Chat Transcript</b><br>
      <div style='background:#0d1220;border-radius:8px;padding:12px;margin-top:8px;font-size:0.82rem;color:#a0aabf;max-height:300px;overflow-y:auto;line-height:1.7;white-space:pre-wrap'>{transcript}</div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# SECTION 8 — WHAT CHANGED THIS WEEK
# ═══════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔄 What Changed This Week?</div>', unsafe_allow_html=True)

new_critical    = agent_summary_df[(agent_summary_df['Focus Zone']=="🔴 Intervene Now")&(agent_summary_df['Risk Δ']>5)]['Agent'].tolist()
newly_improved  = agent_summary_df[agent_summary_df['Risk Δ'] < -5]['Agent'].tolist()
dom_issue       = df[df['DSAT']==1]['Issue_Label'].value_counts().idxmax() if df['DSAT'].sum() > 0 else "N/A"

change_lines = []
if new_critical:
    change_lines.append(f"🔴 <b>{len(new_critical)} agent(s)</b> newly entered high-risk zone: {', '.join(new_critical[:3])}")
if newly_improved:
    change_lines.append(f"🟢 <b>{len(newly_improved)} agent(s)</b> showed significant improvement: {', '.join(newly_improved[:3])}")
change_lines.append(f"📌 Team's dominant complaint category: <b>{dom_issue}</b>")
change_lines.append(f"📊 Team-wide predicted DSAT: <b>{int(team_pred)}</b>")

if prod_wow is not None:
    worst_prod = prod_wow[prod_wow['Change']==prod_wow['Change'].max()].iloc[0]
    best_prod  = prod_wow[prod_wow['Change']==prod_wow['Change'].min()].iloc[0]
    change_lines.append(f"📦 Most deteriorated product this week: <b>{worst_prod['Product']}</b> (+{int(worst_prod['Change'])} DSAT)")
    if best_prod['Change'] < 0:
        change_lines.append(f"📦 Most improved product this week: <b>{best_prod['Product']}</b> ({int(best_prod['Change'])} DSAT)")

st.markdown(f"""
<div class="morning-brief">
  <div class="brief-title">📅 Weekly Delta Summary</div>
  <div class="brief-body">{"<br>".join(change_lines)}</div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='color:#3a4060;font-size:0.75rem;text-align:center;'>DSAT Intelligence Dashboard · Random Forest + TF-IDF NLP · People / Process / Product · Transcript Analysis · Return Prediction</p>", unsafe_allow_html=True)
