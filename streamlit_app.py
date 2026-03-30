import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import Counter
import google.generativeai as genai

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG & SIDEBAR (GEMINI API)
# ─────────────────────────────────────────────
st.set_page_config(page_title="DSAT Intelligence", layout="wide", page_icon="🧠")

st.sidebar.markdown("### 🤖 AI Analysis Settings")
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Required for dynamic Ticket Root Cause analysis. Get it from Google AI Studio.")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; background-color: #0f1117; color: #e8eaf0; }
h1 { color: #ffffff; font-size: 2rem; font-weight: 700; }
h2, h3 { color: #c9d1e8; }
[data-testid="metric-container"] { background: #1a1f2e; border: 1px solid #2e3555; border-radius: 12px; padding: 16px 20px; }
[data-testid="stMetricValue"] { color: #7eb8f7; font-size: 1.7rem; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #8b92ab; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
.section-header { background: linear-gradient(90deg,#1e2540,#131826); border-left: 4px solid #4f7bf7; padding: 10px 18px; border-radius: 6px; margin: 28px 0 14px 0; font-size: 1.05rem; font-weight: 600; color: #c9d1e8; }
.sub-header { border-left: 3px solid #3a4870; padding: 6px 14px; margin: 18px 0 10px 0; font-size: 0.95rem; font-weight: 600; color: #a0adc8; background: #13182a; border-radius: 0 6px 6px 0; }
.morning-brief { background: linear-gradient(135deg,#1a2744,#1a1f2e); border: 1px solid #3a4870; border-radius: 14px; padding: 20px 26px; margin-bottom: 20px; }
.brief-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.12em; color: #7eb8f7; margin-bottom: 8px; }
.brief-body  { font-size: 1.02rem; color: #dce3f5; line-height: 1.8; }
.story-card { background: #141926; border: 1px solid #2e3555; border-radius: 14px; padding: 22px 26px; margin-bottom: 14px; line-height: 1.85; }
.product-story { background: #101520; border: 1px solid #232d45; border-radius: 12px; padding: 16px 20px; margin-bottom: 10px; line-height: 1.8; }
.prod-name { font-size: 1rem; font-weight: 700; color: #7eb8f7; margin-bottom: 6px; }
.wow-better { color: #34d399; font-weight: 600; }
.wow-worse  { color: #f87272; font-weight: 600; }
.wow-same   { color: #8b92ab; }
.ppp-row  { display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0; }
.ppp-pill { padding: 5px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
.pill-people  { background:#2c1f10; border:1px solid #8c5a1a; color:#fb923c; }
.pill-process { background:#252412; border:1px solid #7a7020; color:#facc15; }
.pill-product { background:#102414; border:1px solid #1e6e2a; color:#34d399; }
.matrix-cell { border-radius: 10px; padding: 14px 16px; font-size: 0.85rem; line-height: 1.6; margin-bottom: 10px; }
.cell-red    { background: #2c1414; border: 1px solid #7f2222; }
.cell-orange { background: #2c1f10; border: 1px solid #8c5a1a; }
.cell-yellow { background: #252412; border: 1px solid #7a7020; }
.cell-green  { background: #102414; border: 1px solid #1e6e2a; }
.insight-tag { display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-bottom: 10px; }
.tag-critical  { background: #4a1010; color: #f87272; }
.tag-watchlist { background: #3a2b0a; color: #fbbf24; }
.tag-strong    { background: #0e2e1a; color: #34d399; }
.fivey-card { background: #0d1525; border: 1px solid #1e3a5f; border-radius: 14px; padding: 22px 26px; margin-bottom: 14px; line-height: 1.9; }
.fivey-step { background: #111b30; border-left: 3px solid #4f7bf7; border-radius: 0 8px 8px 0; padding: 12px 18px; margin: 10px 0; }
.fivey-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #4f7bf7; font-weight: 700; }
.fivey-text  { color: #c9d1e8; font-size: 0.93rem; margin-top: 4px; }
.coaching-card { background: #0d1220; border: 1px solid #1e2a40; border-radius: 10px; padding: 14px 18px; margin-top: 10px; }
.coaching-item { padding: 7px 0; border-bottom: 1px solid #1a2235; color: #c9d1e8; font-size: 0.91rem; }
.coaching-item:last-child { border-bottom: none; }
.ticket-box { background: #101520; border: 1px solid #2a3050; border-radius: 12px; padding: 20px 24px; line-height: 1.9; }
.pred-dsat { background:#4a1010; color:#f87272; padding:5px 16px; border-radius:20px; font-weight:700; font-size:0.95rem; display:inline-block; }
.pred-csat { background:#0e2e1a; color:#34d399; padding:5px 16px; border-radius:20px; font-weight:700; font-size:0.95rem; display:inline-block; }
.wow-card { background:#101520; border:1px solid #232d45; border-radius:12px; padding:18px 22px; margin-bottom:10px; line-height:1.85; }
.trend-up   { color:#f87272; font-weight:700; }
.trend-down { color:#34d399; font-weight:700; }
.trend-flat { color:#8b92ab; font-weight:600; }
.gemini-box { background: #1a233a; border: 1px solid #4f7bf7; border-radius: 10px; padding: 18px 22px; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🧠 DSAT Intelligence Dashboard")
st.markdown("<p style='color:#8b92ab;margin-top:-14px;font-size:0.9rem;'>ML-powered · 5-Why root cause · Product & feature WoW · Agent risk & refresher training</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
file_path = "updated_bpo_customer_experience_dataset.csv"
if not os.path.exists(file_path):
    st.error("❌ CSV not found."); st.stop()

df = None
for enc in ["utf-8","latin1","utf-16"]:
    for sep in [",","|",";"]:
        try:
            t = pd.read_csv(file_path, encoding=enc, sep=sep)
            if t.shape[1] > 1: df = t; break
        except: continue
    if df is not None: break
if df is None or df.empty:
    st.error("❌ Failed to load dataset"); st.stop()

df.columns          = df.columns.str.strip()
df['Week']          = pd.to_datetime(df['Week'], errors='coerce')
df                  = df.dropna(subset=['Week'])
df['DSAT']          = df['Customer_Effortless'].apply(lambda x: 1 if str(x).strip().lower()=="no" else 0)
df['Combined_Text'] = df['Customer_Comment'].fillna('') + ' ' + df['Chat_Transcript'].fillna('')
df['transcript_len']= df['Chat_Transcript'].fillna('').apply(len)
df['comment_len']   = df['Customer_Comment'].fillna('').apply(len)

weeks_sorted = sorted(df['Week'].unique())
curr_week    = weeks_sorted[-1]
prev_week    = weeks_sorted[-2] if len(weeks_sorted) >= 2 else curr_week

# ─────────────────────────────────────────────
# LABELS & ML MODEL
# ─────────────────────────────────────────────
def simple_sentiment(text):
    pos_w = ["great","excellent","good","happy","satisfied","thank","thanks","love","helpful","resolved","fixed","perfect","awesome","brilliant","quick","easy"]
    neg_w = ["bad","terrible","awful","horrible","poor","worst","hate","angry","frustrated","useless","broken","failed","slow","rude","unhelpful","annoyed","disgusting","pathetic","waste","never","again","unacceptable"]
    t = str(text).lower()
    return sum(1 for w in pos_w if w in t) - sum(1 for w in neg_w if w in t)

df['Sentiment'] = df['Combined_Text'].apply(simple_sentiment)

PEOPLE_KW  = ["rude","angry","unhelpful","attitude","unprofessional","frustrat","no empathy","escalate","supervisor","bad agent","not listening","dismissive","condescending","impatient","arrogant","yelled","didn't care","poor service","terrible agent","incompetent","clueless"]
PROCESS_KW = ["delay","wait","slow","long hold","transfer","transferred","inefficient","keep asking","repeat","already told","waiting too long","put on hold","no follow up","no callback","still waiting","took forever","long wait","asked again","third time"]
PRODUCT_KW = ["error","bug","failed","not working","broken","limitation","glitch","crash","outage","system issue","technical","doesn't work","cannot login","app crash","feature missing","stopped working","platform issue","website down","login issue","reset not working","page not loading","server error","keeps crashing","software bug"]

def label_issue(text):
    c  = str(text).lower()
    if any(w in c for w in PRODUCT_KW): return "Product"
    ps = sum(1 for w in PEOPLE_KW  if w in c)
    rs = sum(1 for w in PROCESS_KW if w in c)
    if ps == 0 and rs == 0: return "Other"
    if ps >= rs: return "People"
    return "Process"

df['Issue_Label'] = df['Combined_Text'].apply(label_issue)

df_labeled   = df[(df['Issue_Label'] != "Other") & (df['DSAT'] == 1)].copy()
nlp_accuracy = 0.0
vectorizer, issue_model = None, None
model_report = ""
label_dist   = {}

if len(df_labeled) >= 30 and df_labeled['Issue_Label'].nunique() >= 2:
    label_dist = df_labeled['Issue_Label'].value_counts().to_dict()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=4000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_all = vectorizer.fit_transform(df_labeled['Combined_Text'])
    y_all = df_labeled['Issue_Label'].values
    n_splits = max(2, min(5, df_labeled['Issue_Label'].value_counts().min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_true_cv, y_pred_cv = [], []
    for tr_idx, te_idx in skf.split(X_all, y_all):
        m = LogisticRegression(max_iter=500, C=0.7, class_weight='balanced', solver='lbfgs')
        m.fit(X_all[tr_idx], y_all[tr_idx])
        y_true_cv.extend(y_all[te_idx])
        y_pred_cv.extend(m.predict(X_all[te_idx]))
    nlp_accuracy = accuracy_score(y_true_cv, y_pred_cv)
    model_report = classification_report(y_true_cv, y_pred_cv)
    issue_model = LogisticRegression(max_iter=500, C=0.7, class_weight='balanced', solver='lbfgs')
    issue_model.fit(X_all, y_all)
    df['Issue_Label'] = issue_model.predict(vectorizer.transform(df['Combined_Text']))

df['issue_encoded'] = df['Issue_Label'].map({"People":0,"Process":1,"Product":2,"Other":3}).fillna(3)
return_model = None
if df['DSAT'].sum() > 10:
    return_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=6)
    return_model.fit(df[['Sentiment','issue_encoded','transcript_len','comment_len']].fillna(0), df['DSAT'])

weekly_df = df.groupby(['Agent_Name','Week']).agg(DSAT_Count=('DSAT','sum'), Total_Tickets=('Ticket_ID','count')).reset_index()
for i in range(1,5):
    weekly_df[f'DSAT_lag_{i}']    = weekly_df.groupby('Agent_Name')['DSAT_Count'].shift(i)
    weekly_df[f'Tickets_lag_{i}'] = weekly_df.groupby('Agent_Name')['Total_Tickets'].shift(i)
weekly_df = weekly_df.dropna()
lag_features = [f'DSAT_lag_{i}' for i in range(1,5)] + [f'Tickets_lag_{i}' for i in range(1,5)]
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(weekly_df[lag_features], weekly_df['DSAT_Count'])

def compute_agent_summary(wdf, model, feats, y_ref):
    rows = []
    for name, grp in wdf.groupby('Agent_Name'):
        grp = grp.sort_values('Week')
        if len(grp) < 2: continue
        lat, prv  = grp.iloc[-1], grp.iloc[-2]
        p_lat = model.predict([lat[feats]])[0]
        p_prv = model.predict([prv[feats]])[0]
        r_lat = (p_lat - y_ref.mean()) / y_ref.std() * 10 + 50
        r_prv = (p_prv - y_ref.mean()) / y_ref.std() * 10 + 50
        rows.append({'Agent':name, 'Avg DSAT':round(grp['DSAT_Count'].mean(),1), 'Predicted DSAT':int(p_lat), 'Risk Score':int(r_lat), 'Risk Δ':round(r_lat-r_prv,1), 'Trend':float(lat['DSAT_lag_1']-lat['DSAT_lag_4'])})
    return pd.DataFrame(rows)

agent_summary_df = compute_agent_summary(weekly_df, rf_model, lag_features, weekly_df['DSAT_Count'])
med = agent_summary_df['Avg DSAT'].median()
agent_summary_df['Focus Zone'] = agent_summary_df.apply(lambda r: "🔴 Intervene Now" if (r['Avg DSAT']>=med and r['Trend']>0) else ("🟡 Watch Closely" if (not r['Avg DSAT']>=med and r['Trend']>0) else ("🟠 Coach & Monitor" if (r['Avg DSAT']>=med and not r['Trend']>0) else "🟢 Acknowledge")), axis=1)

def ppp_counts(texts):
    text_list = list(texts)
    cats = ["People","Process","Product"]
    if len(text_list) > 0 and vectorizer and issue_model: cnt = Counter(issue_model.predict(vectorizer.transform(text_list)))
    else: cnt = Counter([label_issue(t) for t in text_list])
    return cnt, sum(cnt.get(c,0) for c in cats) or 1

# ══════════════════════════════════════════════════════════
# SECTION 1 & 2 — TEAM OVERVIEW & WoW
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📋 Situation at a Glance</div>', unsafe_allow_html=True)
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("CV Model Accuracy", f"{round(nlp_accuracy*100,1)}%")
c2.metric("Critical Agents", (agent_summary_df['Focus Zone']=="🔴 Intervene Now").sum(), delta_color="inverse")
c3.metric("Worsening This Week", (agent_summary_df['Risk Δ'] > 3).sum(), delta_color="inverse")
c4.metric("Recovering", (agent_summary_df['Risk Δ'] < -3).sum(), delta_color="normal")
c5.metric("Team Predicted DSAT", int(agent_summary_df['Predicted DSAT'].sum()))

st.markdown('<div class="section-header">📊 Week-on-Week: Product & Feature Impact (Team Level)</div>', unsafe_allow_html=True)
prod_curr = df[df['Week']==curr_week].groupby('Product').agg(DSAT_curr=('DSAT','sum'), Tix_curr=('Ticket_ID','count')).reset_index()
prod_prev = df[df['Week']==prev_week].groupby('Product').agg(DSAT_prev=('DSAT','sum'), Tix_prev=('Ticket_ID','count')).reset_index()
prod_wow  = pd.merge(prod_curr, prod_prev, on='Product', how='outer').fillna(0)
prod_wow['Delta'] = prod_wow['DSAT_curr'] - prod_wow['DSAT_prev']
prod_wow = prod_wow.sort_values('Delta', ascending=False)

def render_product_wow_card(row):
    delta = int(row['Delta']); tclass = "trend-up" if delta>0 else ("trend-down" if delta<0 else "trend-flat")
    arr = "⬆" if delta>0 else ("⬇" if delta<0 else "➡")
    badge = "🔴 Worsening" if delta>0 else ("🟢 Improving" if delta<0 else "🟡 Stable")
    return f"""<div class="wow-card"><b style="color:#7eb8f7;font-size:1rem">📦 {row['Product']}</b> &nbsp;&nbsp;<span style="background:#1e2a40;padding:2px 10px;border-radius:12px;font-size:0.78rem;color:#c9d1e8">{badge}</span><br>
    <span style="font-size:1.1rem;color:#c9d1e8">DSAT: {int(row['DSAT_curr'])} (Last week: {int(row['DSAT_prev'])})</span> <span class="{tclass}"> {arr} {delta:+d}</span></div>"""

for _, row in prod_wow.head(3).iterrows(): st.markdown(render_product_wow_card(row), unsafe_allow_html=True)
if not prod_wow.iloc[3:].empty:
    with st.expander("🔽 View Full Product Breakdown"):
        for _, row in prod_wow.iloc[3:].iterrows(): st.markdown(render_product_wow_card(row), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SECTION 3 — AGENT DEEP DIVE (Added Weekly Matrices)
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">👤 Agent Deep Dive — Risk · Root Cause · Weekly Trends</div>', unsafe_allow_html=True)
agent = st.selectbox("Select Agent", sorted(weekly_df['Agent_Name'].unique()), key="agent_sel")

ag_all    = df[df['Agent_Name']==agent]
ag_dsat   = ag_all[ag_all['DSAT']==1]

if len(ag_dsat) > 0:
    st.markdown('<div class="sub-header">📅 Agent Weekly Trend: People/Process/Product (DSAT Count)</div>', unsafe_allow_html=True)
    ppp_weekly = ag_dsat.groupby([ag_dsat['Week'].dt.date, 'Issue_Label'])['Ticket_ID'].count().unstack(fill_value=0)
    for col in ['People', 'Process', 'Product']:
        if col not in ppp_weekly.columns: ppp_weekly[col] = 0
    ppp_weekly = ppp_weekly[['People', 'Process', 'Product']].sort_index(ascending=False)
    st.dataframe(ppp_weekly, use_container_width=True)

    st.markdown('<div class="sub-header">📅 Agent Weekly Trend: Product & Feature (DSAT Count)</div>', unsafe_allow_html=True)
    pf_weekly = ag_dsat.groupby([ag_dsat['Week'].dt.date, 'Product', 'Feature'])['Ticket_ID'].count().reset_index()
    pf_weekly = pf_weekly.rename(columns={'Ticket_ID': 'DSAT Count', 'Week': 'Week Commencing'})
    st.dataframe(pf_weekly.sort_values(['Week Commencing', 'DSAT Count'], ascending=[False, False]), use_container_width=True, hide_index=True)
else:
    st.info("No DSAT tickets recorded for this agent to plot weekly trends.")


# ══════════════════════════════════════════════════════════
# SECTION 4 — TICKET DEEP DIVE (GEMINI LLM FIX)
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎫 Ticket Deep Dive — Root Cause & Prediction</div>', unsafe_allow_html=True)

col_dd, col_manual = st.columns([2,1])
with col_dd:
    selected_dropdown = st.selectbox("Select Ticket ID", sorted(df['Ticket_ID'].unique()), key="tkt_dd")
with col_manual:
    manual_id = st.text_input("Or enter Ticket ID manually", placeholder="e.g. TKT-100042", key="tkt_manual")

ticket_id = manual_id.strip() if manual_id.strip() else selected_dropdown
tkt_rows  = df[df['Ticket_ID']==ticket_id]

if not tkt_rows.empty:
    trow       = tkt_rows.iloc[0]
    combined   = str(trow['Customer_Comment']) + ' ' + str(trow['Chat_Transcript'])
    transcript = str(trow['Chat_Transcript'])
    comment    = str(trow['Customer_Comment'])
    actual     = "🔴 DSAT" if trow['DSAT']==1 else "🟢 CSAT"
    ticket_issue = issue_model.predict(vectorizer.transform([combined]))[0] if (vectorizer and issue_model) else label_issue(combined)

    left, right = st.columns([1.3,1])
    
    with left:
        st.markdown(f"""
        <div class="ticket-box">
          <b>🔍 Ticket Analysis ({actual})</b><br>
          <span style="color:#7eb8f7;font-size:0.85rem">ML Class: <b>{ticket_issue}</b></span><br><br>
          <b>Customer's words:</b><br>
          <span style="color:#aab4cc;font-style:italic">"{comment}"</span>
        </div>
        """, unsafe_allow_html=True)
        
        # ==========================================
        # 🤖 AI ROOT CAUSE GENERATOR (ON-DEMAND)
        # ==========================================
        st.markdown('<div class="sub-header">🤖 AI Root Cause Insight</div>', unsafe_allow_html=True)
        
        if st.button("✨ Generate AI Analysis", key="btn_gemini"):
            if not gemini_api_key:
                st.error("🔑 Please enter your Gemini API Key in the sidebar first.")
            else:
                with st.spinner("Gemini is analyzing the transcript and comments..."):
                    try:
                        genai.configure(api_key=gemini_api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        prompt = f"""
                        You are a QA Analyst reviewing a customer support ticket.
                        Outcome: {actual}
                        Product: {trow['Product']}
                        Feature: {trow['Feature']}
                        Customer Comment: {comment}
                        Transcript: {transcript}
                        
                        If Outcome is DSAT, explain:
                        1. What went wrong
                        2. The Core Issue
                        
                        If Outcome is CSAT, explain:
                        1. What went well
                        2. The Core Success Driver
                        
                        Keep it concise, objective, and use bullet points.
                        """
                        response = model.generate_content(prompt)
                        
                        # Render the response natively in Streamlit to preserve Markdown formatting
                        st.markdown('<div class="gemini-box">', unsafe_allow_html=True)
                        st.markdown(response.text)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error calling Gemini API: {e}")
        else:
            # Show static fallback before the user clicks the button
            if trow['DSAT'] == 1:
                st.info(f"**Static Estimate:** Looks like a **{ticket_issue}** issue. Click the button above to have Gemini read the full transcript and confirm.")
            else:
                st.success("**Static Estimate:** Customer was satisfied. Click the button above for Gemini to analyze what the agent did well.")

    with right:
        st.markdown(f"""
        <div class="ticket-box">
          <b>💬 Full Chat Transcript</b>
          <div style="background:#0d1220;border-radius:8px;padding:12px;margin-top:8px;font-size:0.79rem;color:#a0aabf;max-height:340px;overflow-y:auto;line-height:1.75;white-space:pre-wrap">{transcript}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='color:#3a4060;font-size:0.75rem;text-align:center;'>DSAT Intelligence · RF Forecasting · TF-IDF + Balanced LR (5-fold CV) · 5-Why Root Cause · WoW Product & Feature</p>", unsafe_allow_html=True)
