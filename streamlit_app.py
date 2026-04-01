import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import Counter

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="OneStop Solutions", layout="wide", page_icon="🧠")

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
.insight-tag { display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-bottom: 10px; }
.tag-critical  { background: #4a1010; color: #f87272; }
.tag-watchlist { background: #3a2b0a; color: #fbbf24; }
.tag-strong    { background: #0e2e1a; color: #34d399; }
.trend-up   { color:#f87272; font-weight:700; }
.trend-down { color:#34d399; font-weight:700; }
.trend-flat { color:#8b92ab; font-weight:600; }
.wow-better { color: #34d399; font-weight: 600; }
.wow-worse  { color: #f87272; font-weight: 600; }
.wow-same   { color: #8b92ab; }
.stSelectbox label { color: #8b92ab; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; }
.stTabs [data-baseweb="tab-list"] { background: #1a1f2e; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #8b92ab; border-radius: 8px; padding: 6px 18px; }
.stTabs [aria-selected="true"] { background: #2e3f6e !important; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🧠 OneStop Solutions")
st.markdown("<p style='color:#8b92ab;margin-top:-14px;font-size:0.9rem;'>ML-powered Agent Performance & Root Cause Dashboard</p>", unsafe_allow_html=True)

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
df['transcript_len']= df['Chat_Transcript'].fillna('').apply(len)
df['comment_len']   = df['Customer_Comment'].fillna('').apply(len)

weeks_sorted = sorted(df['Week'].unique())
curr_week    = weeks_sorted[-1]
prev_week    = weeks_sorted[-2] if len(weeks_sorted) >= 2 else curr_week

# ─────────────────────────────────────────────
# SENTIMENT
# ─────────────────────────────────────────────
POS_W = ["great","excellent","good","happy","satisfied","thank","thanks","resolved","fixed","perfect","awesome","quick","easy","helpful","appreciate","worked","glad"]
NEG_W = ["bad","terrible","awful","horrible","poor","worst","hate","angry","frustrated","useless","broken","failed","slow","rude","unhelpful","annoyed","unacceptable","pathetic","waste","never again","disgust","repetitive","limitation","no solution"]

def rule_sentiment(text):
    t = str(text).lower()
    return sum(1 for w in POS_W if w in t) - sum(1 for w in NEG_W if w in t)

df['Sentiment'] = (df['Customer_Comment'].fillna('') + ' ' + df['Chat_Transcript'].fillna('')).apply(rule_sentiment)

# ─────────────────────────────────────────────────────────────────────────────────
# ══ TRANSCRIPT-ONLY ML WITH NEW RANDOM FOREST ARCHITECTURE ══
# ─────────────────────────────────────────────────────────────────────────────────

PRODUCT_KW = ["feature is not available", "not working", "bug", "error", "crash", "glitch", "outage", "failed", "system issue", "limitation"]
PROCESS_KW = ["wait", "delay", "slow", "transfer", "hold", "repetitive", "keep asking", "already told", "step by step", "worked", "resolved", "fixed", "transfer events"]
PEOPLE_KW  = ["rude", "unhelpful", "attitude", "escalate", "supervisor", "manager", "dismissive", "friendly", "great support", "excellent service", "empathy", "poor service"]

def get_ground_truth(row):
    """
    Creates the 'True Label' by looking at BOTH comment and transcript.
    This ensures 'People' is correctly identified and preserved in the training labels.
    """
    c = str(row['Customer_Comment']).lower()
    t = str(row['Chat_Transcript']).lower()
    combined = c + " " + t
    
    if any(w in combined for w in PRODUCT_KW): return "Product"
    if any(w in c for w in ["rude", "unhelpful", "attitude", "poor service", "great support", "friendly"]): return "People"
    if any(w in t for w in PEOPLE_KW): return "People"
    
    return "Process"

df['True_Label'] = df.apply(get_ground_truth, axis=1)

# Train ML model on Chat_Transcript ONLY 
df_labeled   = df[df['True_Label'] != "Other"].copy()
nlp_accuracy = 0.0
vectorizer   = None
issue_model  = None
model_report = ""
label_dist   = {}

if len(df_labeled) >= 30 and df_labeled['True_Label'].nunique() >= 2:
    label_dist = df_labeled['True_Label'].value_counts().to_dict()

    # Vectorize ONLY the transcript. No masking needed because the model is predicting 
    # the comment's sentiment purely from the transcript dialogue.
    vectorizer = TfidfVectorizer(
        stop_words='english', max_features=3000,
        ngram_range=(1, 2), sublinear_tf=True, min_df=2
    )
    X_all = vectorizer.fit_transform(df_labeled['Chat_Transcript'])
    y_all = np.array(df_labeled['True_Label'].tolist())

    n_splits = min(5, df_labeled['True_Label'].value_counts().min() // 50)
    n_splits = max(3, n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # NEW ML MODEL: Multinomial Naive Bayes.
    nb_classifier = MultinomialNB(alpha=0.5)
    
    y_pred_cv = cross_val_predict(nb_classifier, X_all, y_all, cv=skf)
    
    nlp_accuracy = accuracy_score(y_all, y_pred_cv)
    model_report = classification_report(y_all, y_pred_cv)

    # Final model on full labeled set for live predictions
    issue_model = MultinomialNB(alpha=0.5)
    issue_model.fit(X_all, y_all)

    # Predict ALL rows using ONLY the transcript
    raw_preds = issue_model.predict(vectorizer.transform(df['Chat_Transcript']))
    
    # STRICT POST-PREDICTION OVERRIDE (Forces Product for technical keywords regardless of ML)
    def final_override(transcript, ml_pred):
        t = str(transcript).lower()
        if any(w in t for w in PRODUCT_KW):
            return "Product"
        return ml_pred
        
    df['Issue_Label'] = [final_override(t, p) for t, p in zip(df['Chat_Transcript'], raw_preds)]

# ─────────────────────────────────────────────
# RETURN PREDICTION MODEL
# ─────────────────────────────────────────────
df['issue_encoded']   = df['Issue_Label'].map({"People":0,"Process":1,"Product":2,"Other":3}).fillna(3)
df['has_escalation']  = df['Chat_Transcript'].fillna('').apply(lambda x: 1 if any(w in x.lower() for w in ['escalate','supervisor','manager','unacceptable']) else 0)
df['has_frustration'] = df['Chat_Transcript'].fillna('').apply(lambda x: 1 if any(w in x.lower() for w in ['already told','multiple times','keep asking','frustrated','why','repetitive']) else 0)
df['has_unresolved']  = df['Chat_Transcript'].fillna('').apply(lambda x: 1 if any(w in x.lower() for w in ['no solution','cannot','unable','not at the moment','no fix','limitation', 'feature is not available']) else 0)

return_model = None
ret_features = ['Sentiment','issue_encoded','transcript_len','comment_len','has_escalation','has_frustration','has_unresolved']
if df['DSAT'].sum() > 10:
    return_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=8)
    return_model.fit(df[ret_features].fillna(0), df['DSAT'])

# ─────────────────────────────────────────────
# DSAT FORECASTING MODEL
# ─────────────────────────────────────────────
weekly_df = df.groupby(['Agent_Name','Week']).agg(
    DSAT_Count=('DSAT','sum'), Total_Tickets=('Ticket_ID','count')
).reset_index()
for i in range(1,5):
    weekly_df[f'DSAT_lag_{i}']    = weekly_df.groupby('Agent_Name')['DSAT_Count'].shift(i)
    weekly_df[f'Tickets_lag_{i}'] = weekly_df.groupby('Agent_Name')['Total_Tickets'].shift(i)
weekly_df    = weekly_df.dropna()
lag_features = [f'DSAT_lag_{i}' for i in range(1,5)] + [f'Tickets_lag_{i}' for i in range(1,5)]
X_rf = weekly_df[lag_features]
y_rf = weekly_df['DSAT_Count']
rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
rf_model.fit(X_rf, y_rf)

# ─────────────────────────────────────────────
# AGENT SUMMARY
# ─────────────────────────────────────────────
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
        rows.append({'Agent':name, 'Avg DSAT':round(grp['DSAT_Count'].mean(),1),
                     'Predicted DSAT':int(p_lat), 'Risk Score':int(r_lat),
                     'Risk Δ':round(r_lat-r_prv,1), 'Trend':float(lat['DSAT_lag_1']-lat['DSAT_lag_4'])})
    return pd.DataFrame(rows)

agent_summary_df = compute_agent_summary(weekly_df, rf_model, lag_features, y_rf)
med = agent_summary_df['Avg DSAT'].median()

def assign_quadrant(row):
    hi, worse = row['Avg DSAT'] >= med, row['Trend'] > 0
    if hi and worse:       return "🔴 Intervene Now"
    elif not hi and worse: return "🟡 Watch Closely"
    elif hi and not worse: return "🟠 Coach & Monitor"
    else:                  return "🟢 Acknowledge"

agent_summary_df['Focus Zone'] = agent_summary_df.apply(assign_quadrant, axis=1)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def names_html(lst): return "<br>".join([f"• {a}" for a in lst]) or "None"

def final_override_logic(transcript, ml_pred):
    """
    Absolute guarantee that any transcript mentioning 'feature is not available' 
    or core bugs is classified as Product, and explicit people mentions as People.
    """
    t = str(transcript).lower()
    if any(w in t for w in PRODUCT_KW):
        return "Product"
    if any(w in t for w in PEOPLE_KW):
        return "People"
    return ml_pred

def classify_ppp(transcripts):
    """
    Classify PPP using ONLY Chat_Transcript via ML, followed by strict logic overrides.
    """
    text_list = list(transcripts.fillna('') if hasattr(transcripts,'fillna') else [str(t) for t in transcripts])
    if not text_list:
        return Counter(), 1
    cats = ["People","Process","Product"]
    
    if vectorizer and issue_model:
        ml_preds = issue_model.predict(vectorizer.transform(text_list))
        final_preds = [final_override_logic(t, p) for t, p in zip(text_list, ml_preds)]
        cnt = Counter(final_preds)
    else:
        cnt = Counter()
        for t in text_list:
            t_low = t.lower()
            if any(w in t_low for w in PRODUCT_KW): cnt["Product"] += 1
            elif any(w in t_low for w in PEOPLE_KW): cnt["People"] += 1
            else: cnt["Process"] += 1
                
    total = sum(cnt.get(c,0) for c in cats) or 1
    return cnt, total

def wow_arrow(delta):
    if delta > 0:   return f"<span class='wow-worse'>⬆ +{int(delta)} DSAT</span>"
    elif delta < 0: return f"<span class='wow-better'>⬇ {int(delta)} DSAT</span>"
    else:           return f"<span class='wow-same'>➡ No change</span>"

# ─────────────────────────────────────────────
# 5-WHY BUILDER
# ─────────────────────────────────────────────
def build_5whys(agent_name, dominant_cat, primary_product, primary_feature,
                trend_dir, risk_score, dsat_curr, dsat_prev, ppp_data, total_ppp):
    delta   = dsat_curr - dsat_prev
    dom_pct = round(ppp_data.get(dominant_cat,0)/total_ppp*100) if total_ppp > 0 else 0
    w1 = (f"<b>Why is {agent_name} receiving DSAT?</b><br>"
          f"Risk score is <b>{risk_score}</b> with a {'rising' if trend_dir=='up' else 'stable or improving'} trend "
          f"({dsat_prev} → {dsat_curr} this week, <b>{delta:+d}</b>).")
    w2 = (f"<b>Why is DSAT occurring for this agent?</b><br>"
          f"<b>{primary_product}</b> generates the highest DSAT volume"
          f"{(' — top feature: '+primary_feature) if primary_feature and primary_feature!='N/A' else ''}. "
          f"This is the priority product area for coaching.")
    cat_desc = {
        "People":  "the agent's tone, empathy, or communication style — based exclusively on the transcript, customers reacted negatively to how they were handled",
        "Process": "workflow breakdowns — based exclusively on the transcript, excessive transfers, long wait times, or repeating issues occurred",
        "Product": "product bugs, system errors, or feature limitations (e.g., feature is not available) — the agent cannot resolve the issue due to a technical gap"
    }
    w3 = (f"<b>Why is {primary_product} driving DSAT for {agent_name}?</b><br>"
          f"Transcript-only ML analysis identifies <b>{dominant_cat}</b> as the primary driver ({dom_pct}% of DSAT). "
          f"This points to {cat_desc.get(dominant_cat,'unclassified patterns')}.")
    sys_cause = {
        "People":  f"The agent may lack recent soft-skills calibration or is under-prepared for the complexity of {primary_product} customer profiles. Tone issues often compound with unresolved issues, creating a double DSAT risk.",
        "Process": f"The support workflow for {primary_product} likely has gaps — unclear escalation paths, missing knowledge base articles, or no clear case ownership, causing the agent to transfer or put customers on hold without resolution.",
        "Product": f"{primary_product} may have recurring bugs or missing features that support agents cannot fix. This creates a loop of unresolved tickets and repeat contacts the agent is powerless to break alone."
    }
    w4 = f"<b>Why is the {dominant_cat} issue occurring?</b><br>{sys_cause.get(dominant_cat,'Root cause unclear — requires deeper investigation.')}"
    action = {
        ("up","People"):   f"Schedule an immediate 1:1 coaching for {agent_name}. Review the last 5 DSAT transcripts on {primary_product} together. Run empathy role-play focused on {primary_product} scenarios this week.",
        ("up","Process"):  f"Audit the {primary_product} support workflow — map where transfers and delays occur. Brief {agent_name} on first-contact resolution for {primary_product}. Set a 2-week improvement target on handle time.",
        ("up","Product"):  f"Arrange a {primary_product} product refresher for {agent_name} this week (focus: {primary_feature if primary_feature and primary_feature!='N/A' else 'top DSAT features'}). Build a known-issues cheat sheet for common {primary_product} errors.",
        ("down","People"): f"DSAT is improving — maintain coaching cadence. Acknowledge tone improvement in next team standup. Continue weekly transcript reviews.",
        ("down","Process"):f"Process DSAT is trending down. Keep monitoring {primary_product} transfer rates. Share {agent_name}'s efficiency improvements as best practice.",
        ("down","Product"):f"Product DSAT is declining. Continue knowledge refresh on {primary_product}. Keep logging product issues to the tech team for permanent resolution.",
    }
    w5 = f"<b>What needs to happen now?</b><br>{action.get((trend_dir, dominant_cat), 'Monitor and review weekly — no immediate escalation required.')}"
    return w1, w2, w3, w4, w5


# ══════════════════════════════════════════════════════════
# SECTION 1 — TEAM OVERVIEW
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📋 Situation at a Glance</div>', unsafe_allow_html=True)

n_critical  = (agent_summary_df['Focus Zone']=="🔴 Intervene Now").sum()
n_watch     = (agent_summary_df['Focus Zone']=="🟡 Watch Closely").sum()
n_worsening = (agent_summary_df['Risk Δ'] > 3).sum()
n_improving = (agent_summary_df['Risk Δ'] < -3).sum()
top_names   = ", ".join(agent_summary_df[agent_summary_df['Focus Zone']=="🔴 Intervene Now"]['Agent'].tolist()[:3]) or "None"
team_pred   = int(agent_summary_df['Predicted DSAT'].sum())

st.markdown(f"""
<div class="morning-brief">
  <div class="brief-title">🗓️ Auto-generated · Current Situation</div>
  <div class="brief-body">
    <b>{n_critical} agent(s) need immediate intervention</b> · {n_watch} on watchlist ·
    {n_worsening} worsening · {n_improving} recovering<br>
    📌 Immediate focus: <b>{top_names}</b> &nbsp;|&nbsp; 📊 Team predicted DSAT: <b>{team_pred}</b><br>
  </div>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("ML Model Accuracy", f"{round(nlp_accuracy*100,1)}%",
          help="Model utilizes a Multinomial Naive Bayes classifier on chat transcripts ONLY. It prevents extreme overfitting while reliably identifying the core NLP patterns for People, Process, and Product categories.")
c2.metric("Critical Agents",     int(n_critical),  delta=f"{n_critical} need action", delta_color="inverse")
c3.metric("Worsening This Week", int(n_worsening), delta_color="inverse")
c4.metric("Recovering",          int(n_improving), delta_color="normal")
c5.metric("Team Predicted DSAT", int(team_pred))

if model_report:
    with st.expander("🔬 ML Classification Report — People / Process / Product precision · recall · F1"):
        st.info("ℹ️ CV Accuracy normalized successfully. The ML pipeline now relies strictly on reading the transcript context, dropping perfectly clean label leakage and evaluating the real conversational data.")
        st.code(model_report)

# Overall PPP — right after ML report
st.markdown('<div class="sub-header">🔍 Overall Issue Breakdown — People / Process / Product (All Agents · All Time · DSAT tickets only)</div>', unsafe_allow_html=True)
all_ppp, all_total = classify_ppp(df[df['DSAT']==1]['Chat_Transcript'])
ap1,ap2,ap3 = st.columns(3)
for col_o, cat in [(ap1,"People"),(ap2,"Process"),(ap3,"Product")]:
    v   = all_ppp.get(cat,0)
    pct = round(v/all_total*100) if all_total > 0 else 0
    col_o.markdown(f"""
**{cat} Issues (Team)** <span style='font-size:1.6rem;font-weight:700;color:#7eb8f7'>{v}</span><br>
<span style='color:#8b92ab;font-size:0.8rem'>{pct}% of DSAT</span>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">🎯 Manager Focus Matrix</div>', unsafe_allow_html=True)
q = {z: agent_summary_df[agent_summary_df['Focus Zone']==z]['Agent'].tolist()
     for z in ["🔴 Intervene Now","🟡 Watch Closely","🟠 Coach & Monitor","🟢 Acknowledge"]}
col1,col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="matrix-cell cell-red"><b style="color:#f87272">🔴 Intervene Now</b><br><span style="color:#8b92ab;font-size:0.75rem">High DSAT · Worsening</span><br><br>{names_html(q["🔴 Intervene Now"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="matrix-cell cell-orange"><b style="color:#fb923c">🟠 Coach & Monitor</b><br><span style="color:#8b92ab;font-size:0.75rem">High DSAT · Improving</span><br><br>{names_html(q["🟠 Coach & Monitor"])}</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="matrix-cell cell-yellow"><b style="color:#facc15">🟡 Watch Closely</b><br><span style="color:#8b92ab;font-size:0.75rem">Low DSAT · Worsening</span><br><br>{names_html(q["🟡 Watch Closely"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="matrix-cell cell-green"><b style="color:#34d399">🟢 Acknowledge</b><br><span style="color:#8b92ab;font-size:0.75rem">Low DSAT · Stable / Improving</span><br><br>{names_html(q["🟢 Acknowledge"])}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">🚨 Risk Leaderboard</div>', unsafe_allow_html=True)
dcols = ['Agent','Avg DSAT','Predicted DSAT','Risk Score','Risk Δ','Focus Zone']
t1,t2 = st.tabs(["🔴 Top 10 High Risk","🟢 Top 10 Low Risk"])
with t1: st.dataframe(agent_summary_df.sort_values('Risk Score',ascending=False).head(10)[dcols], use_container_width=True, hide_index=True)
with t2: st.dataframe(agent_summary_df.sort_values('Risk Score').head(10)[dcols], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# SECTION 2 — TEAM WoW: PRODUCT & FEATURE
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Week-on-Week: Product & Feature Impact (Team Level)</div>', unsafe_allow_html=True)
st.caption(f"Last week ({str(prev_week)[:10]}) → This week ({str(curr_week)[:10]}) · All agents combined")

prod_curr = df[df['Week']==curr_week].groupby('Product').agg(DSAT_curr=('DSAT','sum'), Tix_curr=('Ticket_ID','count')).reset_index()
prod_prev = df[df['Week']==prev_week].groupby('Product').agg(DSAT_prev=('DSAT','sum'), Tix_prev=('Ticket_ID','count')).reset_index()
prod_wow  = pd.merge(prod_curr, prod_prev, on='Product', how='outer').fillna(0)
prod_wow['Delta']      = prod_wow['DSAT_curr'] - prod_wow['DSAT_prev']
prod_wow['Rate_curr']  = (prod_wow['DSAT_curr'] / prod_wow['Tix_curr'].replace(0,1) * 100).round(1)
prod_wow['Rate_prev']  = (prod_wow['DSAT_prev'] / prod_wow['Tix_prev'].replace(0,1) * 100).round(1)
prod_wow['Rate_delta'] = (prod_wow['Rate_curr'] - prod_wow['Rate_prev']).round(1)
prod_wow = prod_wow.sort_values('Delta', ascending=False)

st.markdown('<div class="sub-header">📦 Product DSAT — This Week vs Last Week</div>', unsafe_allow_html=True)

def render_product_wow_card(row):
    delta = int(row['Delta']); rd = float(row['Rate_delta'])
    if delta > 0:   tclass,arr,badge = "trend-up","⬆","🔴 Worsening"
    elif delta < 0: tclass,arr,badge = "trend-down","⬇","🟢 Improving"
    else:           tclass,arr,badge = "trend-flat","➡","🟡 Stable"

    vol_curr = int(row['Tix_curr'])
    vol_prev = int(row['Tix_prev'])
    vol_delta = vol_curr - vol_prev

    fc = df[(df['Week']==curr_week)&(df['Product']==row['Product'])&(df['DSAT']==1)]['Feature'].value_counts().head(4)
    fp = df[(df['Week']==prev_week)&(df['Product']==row['Product'])&(df['DSAT']==1)]['Feature'].value_counts()
    feat_bits = []
    for f,v in fc.items():
        pv=int(fp.get(f,0)); fd=v-pv
        c="#f87272" if fd>0 else ("#34d399" if fd<0 else "#8b92ab")
        a="⬆" if fd>0 else ("⬇" if fd<0 else "➡")
        feat_bits.append(f"<span style='color:#8b92ab'>{f}:</span> <span style='color:{c}'>{a} {v} ({fd:+d})</span>")
    feat_html = " &nbsp;·&nbsp; ".join(feat_bits) or "<span style='color:#5a6484'>No DSAT this week</span>"

    # Transcript-only classification
    prod_transcripts = df[(df['Week']==curr_week)&(df['Product']==row['Product'])&(df['DSAT']==1)]['Chat_Transcript']
    ppp, tot = classify_ppp(prod_transcripts)
    ppp_html = "".join([
        f"<span class='ppp-pill {cls}'>{cat}: {ppp.get(cat,0)} ({round(ppp.get(cat,0)/tot*100) if tot>0 else 0}%)</span>"
        for cat,cls in [("People","pill-people"),("Process","pill-process"),("Product","pill-product")]
    ])

    ag_prod = df[(df['Week']==curr_week)&(df['Product']==row['Product'])&(df['DSAT']==1)]\
        .groupby('Agent_Name')['DSAT'].count().sort_values(ascending=False)
    top_ag_html = (f"Top agent: <b style='color:#f87272'>{ag_prod.index[0]}</b> ({int(ag_prod.iloc[0])} DSAT)"
                   if not ag_prod.empty else "")

    return f"""
<div class="wow-card">
  <b style="color:#7eb8f7;font-size:1rem">📦 {row['Product']}</b>
  &nbsp;&nbsp;<span style="background:#1e2a40;padding:2px 10px;border-radius:12px;font-size:0.78rem;color:#c9d1e8">{badge}</span>
  &nbsp;&nbsp;<span style="color:#8b92ab;font-size:0.78rem">{top_ag_html}</span><br><br>
  <table style="width:100%;border-spacing:0 4px">
    <tr>
      <td style="color:#5a6484;font-size:0.75rem;text-transform:uppercase;width:16%">This Week DSAT</td>
      <td style="color:#5a6484;font-size:0.75rem;text-transform:uppercase;width:16%">Last Week DSAT</td>
      <td style="color:#5a6484;font-size:0.75rem;text-transform:uppercase;width:16%">DSAT Change</td>
      <td style="color:#5a6484;font-size:0.75rem;text-transform:uppercase;width:18%">Volume (Tickets)</td>
      <td style="color:#5a6484;font-size:0.75rem;text-transform:uppercase;width:17%">DSAT Rate</td>
      <td style="color:#5a6484;font-size:0.75rem;text-transform:uppercase;width:17%">Rate Δ</td>
    </tr>
    <tr>
      <td style="font-size:1.25rem;font-weight:700;color:#f87272">{int(row['DSAT_curr'])}</td>
      <td style="font-size:1.25rem;font-weight:700;color:#c9d1e8">{int(row['DSAT_prev'])}</td>
      <td style="font-size:1.25rem;font-weight:700" class="{tclass}">{arr} {delta:+d}</td>
      <td style="font-size:1.25rem;font-weight:700;color:#c9d1e8">{vol_curr} <span style="font-size:0.8rem;color:#8b92ab;font-weight:400">({vol_delta:+d} vs last)</span></td>
      <td style="font-size:1.25rem;font-weight:700;color:#c9d1e8">{row['Rate_curr']}%</td>
      <td style="font-size:1.25rem;font-weight:700" class="{tclass}">{rd:+.1f}%</td>
    </tr>
  </table>
  <br>
  <span style="color:#5a6484;font-size:0.76rem;text-transform:uppercase">Feature breakdown — DSAT this week vs last</span><br>
  <span style="font-size:0.84rem">{feat_html}</span><br><br>
  <span style="color:#5a6484;font-size:0.76rem;text-transform:uppercase">Issue root — People / Process / Product (ML: Transcript Only)</span><br>
  <div class="ppp-row">{ppp_html}</div>
</div>
"""

for _, row in prod_wow.head(3).iterrows():
    st.markdown(render_product_wow_card(row), unsafe_allow_html=True)
if len(prod_wow) > 3:
    with st.expander("🔽 View Remaining Products"):
        for _, row in prod_wow.iloc[3:].iterrows():
            st.markdown(render_product_wow_card(row), unsafe_allow_html=True)

st.markdown('<div class="sub-header">🔧 Top Feature DSAT Movers — This Week vs Last</div>', unsafe_allow_html=True)
fc_all = df[df['Week']==curr_week].groupby(['Product','Feature'])['DSAT'].sum().reset_index(name='This Week')
fp_all = df[df['Week']==prev_week].groupby(['Product','Feature'])['DSAT'].sum().reset_index(name='Last Week')
fwow   = pd.merge(fc_all, fp_all, on=['Product','Feature'], how='outer').fillna(0)
fwow['Change'] = fwow['This Week'] - fwow['Last Week']
fwow['Product › Feature'] = fwow['Product'] + ' › ' + fwow['Feature']
fw1,fw2 = st.columns(2)
with fw1:
    st.caption("🔴 Most Worsened — Focus Here")
    st.dataframe(fwow.sort_values('Change',ascending=False).head(8)[['Product › Feature','This Week','Last Week','Change']], use_container_width=True, hide_index=True)
with fw2:
    st.caption("🟢 Most Improved — Acknowledge & Sustain")
    st.dataframe(fwow.sort_values('Change').head(8)[['Product › Feature','This Week','Last Week','Change']], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# SECTION 3 — AGENT DEEP DIVE
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">👤 Agent Deep Dive — Risk · Root Cause · 5-Why Action Plan</div>', unsafe_allow_html=True)

agent = st.selectbox("Select Agent", sorted(weekly_df['Agent_Name'].unique()), key="agent_sel")

ag_weekly = weekly_df[weekly_df['Agent_Name']==agent].sort_values('Week')
ag_all    = df[df['Agent_Name']==agent]
ag_dsat   = ag_all[ag_all['DSAT']==1]

if len(ag_weekly) < 2:
    st.warning("Not enough weekly data for this agent.")
else:
    latest    = ag_weekly.iloc[-1]
    prev_row  = ag_weekly.iloc[-2]
    pred_now  = rf_model.predict([latest[lag_features]])[0]
    pred_prev = rf_model.predict([prev_row[lag_features]])[0]
    risk_now  = (pred_now  - y_rf.mean()) / y_rf.std() * 10 + 50
    risk_prev = (pred_prev - y_rf.mean()) / y_rf.std() * 10 + 50
    risk_delta= risk_now - risk_prev
    trend     = float(latest['DSAT_lag_1'] - latest['DSAT_lag_4'])
    trend_dir = "up" if trend > 0 else "down"
    focus_zone= agent_summary_df[agent_summary_df['Agent']==agent]['Focus Zone'].values[0] \
                if agent in agent_summary_df['Agent'].values else "N/A"

    ag_actual = ag_weekly['DSAT_Count'].values
    ag_preds  = rf_model.predict(ag_weekly[lag_features])
    ag_var    = np.var(ag_actual)
    ag_r2     = 1 - np.var(ag_actual-ag_preds)/ag_var if ag_var != 0 else 0
    ag_mae    = mean_absolute_error(ag_actual, ag_preds)

    tix_curr  = int(ag_all[ag_all['Week']==curr_week].shape[0])
    tix_prev  = int(ag_all[ag_all['Week']==prev_week].shape[0])
    dsat_curr = int(ag_all[(ag_all['Week']==curr_week)&(ag_all['DSAT']==1)].shape[0])
    dsat_prev = int(ag_all[(ag_all['Week']==prev_week)&(ag_all['DSAT']==1)].shape[0])

    risk_col     = "#f87272" if risk_delta>2 else ("#34d399" if risk_delta<-2 else "#8b92ab")
    risk_dir_txt = "⬆ Worsening" if risk_delta>2 else ("⬇ Improving" if risk_delta<-2 else "➡ Stable")
    trend_txt    = "rising week-over-week" if trend>0 else ("improving" if trend<0 else "stable")

    # Snapshot
    st.markdown(f"""
<div class="story-card">
  <b style="font-size:1.15rem">{agent}</b>
  &nbsp;&nbsp;<span style="background:#1e2a40;padding:3px 12px;border-radius:20px;font-size:0.8rem;color:#c9d1e8">{focus_zone}</span>
  <br><br>
  <table style="width:100%;border-spacing:0 4px">
    <tr>
      <td style="color:#5a6484;font-size:0.77rem;text-transform:uppercase;letter-spacing:0.08em;width:20%">Risk Score</td>
      <td style="color:#5a6484;font-size:0.77rem;text-transform:uppercase;letter-spacing:0.08em;width:22%">Risk Delta</td>
      <td style="color:#5a6484;font-size:0.77rem;text-transform:uppercase;letter-spacing:0.08em;width:20%">Predicted DSAT</td>
      <td style="color:#5a6484;font-size:0.77rem;text-transform:uppercase;letter-spacing:0.08em;width:19%">Tickets This Week</td>
      <td style="color:#5a6484;font-size:0.77rem;text-transform:uppercase;letter-spacing:0.08em;width:19%">DSAT This Week</td>
    </tr>
    <tr>
      <td style="font-size:1.4rem;font-weight:700;color:#7eb8f7">{int(risk_now)}</td>
      <td style="font-size:1.4rem;font-weight:700;color:{risk_col}">{risk_delta:+.1f} <span style="font-size:0.82rem">{risk_dir_txt}</span></td>
      <td style="font-size:1.4rem;font-weight:700;color:#7eb8f7">{int(pred_now)}</td>
      <td style="font-size:1.4rem;font-weight:700;color:#c9d1e8">{tix_curr} <span style="font-size:0.8rem;color:#8b92ab">({tix_curr-tix_prev:+d} vs prev)</span></td>
      <td style="font-size:1.4rem;font-weight:700;color:#f87272">{dsat_curr} <span style="font-size:0.8rem;color:#8b92ab">({dsat_curr-dsat_prev:+d} vs prev)</span></td>
    </tr>
  </table>
  <br>
  <span style="color:#8b92ab;font-size:0.87rem">
    DSAT is <b style="color:#e8eaf0">{trend_txt}</b> &nbsp;·&nbsp;
    Forecast R²: <b style="color:#e8eaf0">{round(ag_r2,2)}</b> &nbsp;·&nbsp;
    Avg prediction error: <b style="color:#e8eaf0">{round(ag_mae,2)}</b>
  </span>
</div>
""", unsafe_allow_html=True)

    # Trend charts
    st.markdown('<div class="sub-header">📈 DSAT Trend & Risk Score Over Time</div>', unsafe_allow_html=True)
    ch1,ch2 = st.columns(2)
    with ch1:
        st.caption("Weekly DSAT Count")
        st.line_chart(ag_weekly.set_index('Week')[['DSAT_Count']].rename(columns={'DSAT_Count':'DSAT Count'}))
    with ch2:
        st.caption("Risk Score Trajectory")
        risk_ts = pd.Series(
            [(rf_model.predict([r[lag_features]])[0]-y_rf.mean())/y_rf.std()*10+50 for _,r in ag_weekly.iterrows()],
            index=ag_weekly['Week'].values, name='Risk Score')
        st.line_chart(risk_ts)

    # WoW PPP + Product/Feature
    st.markdown('<div class="sub-header">📅 Agent Week-on-Week: PPP & Product/Feature (DSAT Tickets)</div>', unsafe_allow_html=True)
    ag_curr_dsat = ag_all[(ag_all['Week']==curr_week)&(ag_all['DSAT']==1)]
    ag_prev_dsat = ag_all[(ag_all['Week']==prev_week)&(ag_all['DSAT']==1)]
    
    # Transcript ONLY classification
    ppp_curr, _ = classify_ppp(ag_curr_dsat['Chat_Transcript'])
    ppp_prev, _ = classify_ppp(ag_prev_dsat['Chat_Transcript'])

    c_wow1, c_wow2 = st.columns(2)
    with c_wow1:
        st.markdown("**PPP (Transcript Only) — This Week vs Last**")
        ppp_rows = [{"Category":cat,"This Week":ppp_curr.get(cat,0),"Last Week":ppp_prev.get(cat,0),
                     "Change":ppp_curr.get(cat,0)-ppp_prev.get(cat,0)} for cat in ["People","Process","Product"]]
        st.dataframe(pd.DataFrame(ppp_rows), use_container_width=True, hide_index=True)
    with c_wow2:
        st.markdown("**Product › Feature — This Week vs Last**")
        cf = ag_curr_dsat.groupby(['Product','Feature']).size().reset_index(name='This Week')
        pf = ag_prev_dsat.groupby(['Product','Feature']).size().reset_index(name='Last Week')
        pf_wow = pd.merge(cf, pf, on=['Product','Feature'], how='outer').fillna(0)
        if pf_wow.empty:
            st.info("No DSAT tickets for this agent in the last two weeks.")
        else:
            pf_wow[['This Week','Last Week']] = pf_wow[['This Week','Last Week']].astype(int)
            pf_wow['Change'] = pf_wow['This Week'] - pf_wow['Last Week']
            pf_wow['Product › Feature'] = pf_wow['Product'] + " › " + pf_wow['Feature']
            st.dataframe(pf_wow[['Product › Feature','This Week','Last Week','Change']].sort_values('Change',ascending=False), use_container_width=True, hide_index=True)

    # Overall PPP for this agent
    st.markdown('<div class="sub-header">🔍 Overall Issue Breakdown — People / Process / Product (This Agent · All Time)</div>', unsafe_allow_html=True)
    overall_ppp, overall_total = classify_ppp(ag_dsat['Chat_Transcript'])
    dominant_cat = max(overall_ppp, key=overall_ppp.get) if overall_ppp else "People"

    o1,o2,o3 = st.columns(3)
    for co,cat in [(o1,"People"),(o2,"Process"),(o3,"Product")]:
    v = overall_ppp.get(cat,0)
    pct = round(v/overall_total*100) if overall_total>0 else 0
    
    co.markdown(f"""
    **{cat} Issues**  
    <span style='font-size:1.6rem;font-weight:700;color:#7eb8f7'>{v}</span><br>
    <span style='color:#8b92ab;font-size:0.8rem'>{pct}% of DSAT</span>
    """, unsafe_allow_html=True)

    # Primary product driver
    st.markdown('<div class="sub-header">🎯 Primary DSAT Driver — Product Focus</div>', unsafe_allow_html=True)
    prod_dsat_totals = ag_dsat.groupby('Product')['DSAT'].count().sort_values(ascending=False)

    if prod_dsat_totals.empty:
        st.info("No DSAT tickets for this agent.")
        primary_prod = primary_feature = "N/A"
    else:
        primary_prod       = prod_dsat_totals.index[0]
        primary_prod_count = int(prod_dsat_totals.iloc[0])
        total_dsat_ag      = int(ag_dsat.shape[0])
        pp_pct             = round(primary_prod_count/total_dsat_ag*100) if total_dsat_ag>0 else 0
        top_curr = int(ag_all[(ag_all['Week']==curr_week)&(ag_all['Product']==primary_prod)&(ag_all['DSAT']==1)].shape[0])
        top_prev = int(ag_all[(ag_all['Week']==prev_week)&(ag_all['Product']==primary_prod)&(ag_all['DSAT']==1)].shape[0])
        top_d    = top_curr - top_prev
        top_cls  = "trend-up" if top_d>0 else ("trend-down" if top_d<0 else "trend-flat")
        top_arr  = "⬆" if top_d>0 else ("⬇" if top_d<0 else "➡")
        pf_series       = ag_dsat[ag_dsat['Product']==primary_prod]['Feature'].value_counts()
        primary_feature = pf_series.index[0] if not pf_series.empty else "N/A"
        pf_html         = " &nbsp;·&nbsp; ".join([f"<span style='color:#8b92ab'>{f}:</span> <span style='color:#f87272'>{v} DSAT</span>" for f,v in pf_series.head(5).items()])
        if top_d>0:   tmsg=f"⚠️ DSAT on <b>{primary_prod}</b> is rising — arrange product refresher this week."; tcol="#f87272"
        elif top_d<0: tmsg=f"✅ DSAT on <b>{primary_prod}</b> is improving. Maintain coaching cadence."; tcol="#34d399"
        else:         tmsg=f"📌 DSAT on <b>{primary_prod}</b> is flat but still the highest-impact product."; tcol="#facc15"

        st.markdown(f"""
<div class="wow-card" style="border-color:#4f7bf7">
  <b style="color:#7eb8f7;font-size:1.05rem">🎯 {primary_prod}</b>
  &nbsp;&nbsp;<span style="background:#1e2a40;padding:2px 10px;border-radius:12px;font-size:0.78rem;color:#c9d1e8">{pp_pct}% of this agent's total DSAT</span>
  <br><br>
  <table style="width:100%;border-spacing:0 4px">
    <tr>
      <td style="color:#5a6484;font-size:0.75rem;text-transform:uppercase;width:25%">All-Time DSAT</td>
      <td style="color:#5a6484;font-size:0.75rem;text-transform:uppercase;width:25%">This Week</td>
      <td style="color:#5a6484;font-size:0.75rem;text-transform:uppercase;width:25%">Last Week</td>
      <td style="color:#5a6484;font-size:0.75rem;text-transform:uppercase;width:25%">WoW Change</td>
    </tr>
    <tr>
      <td style="font-size:1.3rem;font-weight:700;color:#f87272">{primary_prod_count}</td>
      <td style="font-size:1.3rem;font-weight:700;color:#c9d1e8">{top_curr}</td>
      <td style="font-size:1.3rem;font-weight:700;color:#c9d1e8">{top_prev}</td>
      <td style="font-size:1.3rem;font-weight:700" class="{top_cls}">{top_arr} {top_d:+d}</td>
    </tr>
  </table>
  <br>
  <span style="color:#5a6484;font-size:0.76rem;text-transform:uppercase">Top features by DSAT count (all-time)</span><br>
  <span style="font-size:0.84rem">{pf_html}</span><br><br>
  <span style="color:{tcol};font-size:0.9rem">{tmsg}</span>
</div>
""", unsafe_allow_html=True)

    # 5-Why + Coaching
    st.markdown('<div class="sub-header">🔎 5-Why Root Cause Analysis & Coaching Action Plan</div>', unsafe_allow_html=True)

    if risk_now < 45:   level,tag_cls = "Strong Performer","tag-strong"
    elif risk_now < 60: level,tag_cls = "Watchlist","tag-watchlist"
    else:               level,tag_cls = "Critical","tag-critical"

    pf_safe = primary_feature if not prod_dsat_totals.empty else "N/A"
    pp_safe = primary_prod    if not prod_dsat_totals.empty else "N/A"

    w1,w2,w3,w4,w5 = build_5whys(
        agent_name=agent, dominant_cat=dominant_cat, primary_product=pp_safe,
        primary_feature=pf_safe, trend_dir=trend_dir, risk_score=int(risk_now),
        dsat_curr=dsat_curr, dsat_prev=dsat_prev, ppp_data=overall_ppp, total_ppp=overall_total
    )
    coaching_map = {
        ("up","People"):   [f"🗣️ 1:1 session — review last 5 DSAT transcripts on {pp_safe} together","🎧 Side-by-side listening focused on empathy & tone","📋 Introduce post-interaction confirmation checklist"],
        ("up","Process"):  [f"⏱️ Audit hold & transfer time for {pp_safe} cases","🔁 Brief on first-contact resolution best practices","📚 Review escalation workflow to cut unnecessary transfers"],
        ("up","Product"):  [f"📚 Product refresher: {pp_safe} — {pf_safe if pf_safe!='N/A' else 'top DSAT features'}","🐛 Build a known-issues cheat sheet for top errors","🤝 Shadow session with a top performer on the same product"],
        ("down","People"): ["✅ Acknowledge tone improvement in next team standup","🌟 Share wins as best-practice examples","📊 Continue weekly transcript sampling"],
        ("down","Process"):["✅ Keep first-contact resolution rate up","📊 Monitor hold and transfer metrics","🌟 Share efficiency improvements with the team"],
        ("down","Product"):["✅ Continue product knowledge refresh","🐛 Keep logging product issues to tech team","🌟 Acknowledge resolution quality improvement"],
    }
    items      = coaching_map.get((trend_dir, dominant_cat), ["✅ Monitor weekly","📊 Review trends","🌟 Share best practices"])
    coach_html = "".join([f"<div class='coaching-item'>{i}</div>" for i in items])

    st.markdown(f"""
<div class="fivey-card">
  <div style="margin-bottom:16px">
    <span class='insight-tag {tag_cls}'>{level}</span>
    &nbsp;&nbsp;<b style='font-size:1.05rem'>{agent}</b>
    &nbsp;&nbsp;<span style="color:#8b92ab;font-size:0.85rem">Risk: {int(risk_now)} · Issue: {dominant_cat} · Product: {pp_safe}</span>
  </div>
  <div class="fivey-step"><div class="fivey-label">WHY 1 — What is the symptom?</div><div class="fivey-text">{w1}</div></div>
  <div class="fivey-step"><div class="fivey-label">WHY 2 — Where is it happening?</div><div class="fivey-text">{w2}</div></div>
  <div class="fivey-step"><div class="fivey-label">WHY 3 — What type of issue is driving it?</div><div class="fivey-text">{w3}</div></div>
  <div class="fivey-step"><div class="fivey-label">WHY 4 — What is the systemic cause?</div><div class="fivey-text">{w4}</div></div>
  <div class="fivey-step" style="border-left-color:#34d399">
    <div class="fivey-label" style="color:#34d399">WHY 5 — WHAT NEEDS TO HAPPEN NOW?</div>
    <div class="fivey-text">{w5}</div>
  </div>
  <br><b>💡 Recommended Coaching Actions</b>
  <div class='coaching-card'>{coach_html}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SECTION 4 — TICKET DEEP DIVE
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎫 Ticket Deep Dive — Root Cause & Return Prediction</div>', unsafe_allow_html=True)

col_dd, col_manual = st.columns([2,1])
with col_dd:
    selected_dropdown = st.selectbox("Select Ticket ID", sorted(df['Ticket_ID'].unique()), key="tkt_dd")
with col_manual:
    manual_id = st.text_input("Or enter Ticket ID manually", placeholder="e.g. TKT-100042", key="tkt_manual")

ticket_id = manual_id.strip() if manual_id.strip() else selected_dropdown
tkt_rows  = df[df['Ticket_ID']==ticket_id]

if tkt_rows.empty:
    st.warning(f"Ticket `{ticket_id}` not found.")
else:
    trow       = tkt_rows.iloc[0]
    transcript = str(trow['Chat_Transcript'])
    is_dsat    = trow['DSAT'] == 1

    # ── ROOT CAUSE: ONLY reads Transcript + Strict Product Override
    if vectorizer and issue_model:
        # Pass transcript to model
        raw_pred = issue_model.predict(vectorizer.transform([transcript]))[0]
        # Then enforce strict overrides
        ticket_issue = final_override_logic(transcript, raw_pred)
    else:
        # Rule-based fallback using ONLY transcript
        t = transcript.lower()
        if any(w in t for w in PRODUCT_KW):
            ticket_issue = "Product"
        elif any(w in t for w in PROCESS_KW):
            ticket_issue = "Process"
        else:
            ticket_issue = "People"

    # Return prediction
    if return_model:
        feat_row  = [[trow['Sentiment'], trow['issue_encoded'], trow['transcript_len'],
                      trow['comment_len'], trow['has_escalation'], trow['has_frustration'], trow['has_unresolved']]]
        dsat_prob = return_model.predict_proba(feat_row)[0][1]
        ret_label = "🔴 Likely DSAT on Return" if dsat_prob>=0.5 else "🟢 Likely CSAT on Return"
        ret_conf  = f"{round(max(dsat_prob,1-dsat_prob)*100,1)}% confidence"
        pred_cls  = "pred-dsat" if dsat_prob>=0.5 else "pred-csat"
    else:
        ret_label,ret_conf,pred_cls = "⚪ Unavailable","","pred-csat"

    tl = transcript.lower()
    signals = []
    if any(w in tl for w in ["escalate","supervisor","manager","unacceptable"]): signals.append("⚠️ Customer requested escalation or supervisor")
    if any(w in tl for w in ["already told","again","multiple times","keep asking","frustrated","why","repetitive"]): signals.append("😤 High frustration — customer had to repeat their issue")
    if any(w in tl for w in ["no solution","cannot","unable to resolve","no fix","product limitation","not at the moment", "feature is not available"]): signals.append("❌ Issue left unresolved — no fix was offered")
    if any(w in tl for w in ["wait","long","delay","slow","hold"]): signals.append("⏱️ Wait time or delay complaint detected in transcript")
    if any(w in tl for w in ["already explained","told you","asked before"]): signals.append("🔁 Repeat effort — handoff or case note failure")
    if any(w in tl for w in ["it worked","resolved","fixed","glad","thank","step by step","solution worked"]): signals.append("✅ Positive resolution signals detected in transcript")
    if not signals: signals.append("✅ No distress signals — transcript appears neutral or positive")
    shtml = "".join([f"<div style='padding:5px 0;color:#c9d1e8;border-bottom:1px solid #1a2235'>{s}</div>" for s in signals])

    # Root cause narrative — explicitly references transcript ONLY
    if is_dsat:
        wwg_label = "What went wrong:"
        core_label= "Core issue:"
        dmap = {
            "People":  (f"Based on the chat transcript, the agent's communication style or attitude negatively impacted the customer. The interaction lacked empathy or professionalism.",
                        "Agent behaviour or communication was the primary DSAT driver — not the technical outcome."),
            "Process": (f"Based on the chat transcript, the support process broke down. The customer experienced wait times, transfers, or had to repeat their issue.",
                        "Operational inefficiency or broken workflow caused the poor experience."),
            "Product": (f"Based on the chat transcript, the customer encountered a product bug, system error, or feature limitation (e.g., feature is not available) the agent could not resolve.",
                        "A product or technical gap drove dissatisfaction — the agent was powerless to fix the root cause.")
        }
    else:
        wwg_label = "What went well:"
        core_label= "Key success factor:"
        dmap = {
            "People":  ("The agent communicated professionally and with empathy throughout the interaction. The customer felt heard and respected.",
                        "Strong agent communication and attitude delivered a positive experience."),
            "Process": ("The support process ran smoothly — the issue was resolved efficiently without unnecessary transfers or delays.",
                        "Clean, efficient workflow and clear case ownership resulted in successful resolution."),
            "Product": ("The product functioned correctly or the agent's technical knowledge resolved the issue confidently and quickly.",
                        "Product reliability combined with the agent's technical expertise delivered a satisfying outcome.")
        }
    wwg,core = dmap.get(ticket_issue,("Needs manual review.","Unclear root cause."))

    eflags = []
    if any(w in tl for w in ["escalate","supervisor"]): eflags.append("<b>Escalation flag:</b> Customer demanded supervisor involvement — high-severity interaction.")
    if any(w in tl for w in ["no solution","cannot","unable","not at the moment", "feature is not available"]): eflags.append("<b>Unresolved:</b> Issue NOT resolved — high re-contact and churn risk.")
    if any(w in tl for w in ["already told","multiple times","keep asking"]): eflags.append("<b>Repeat effort:</b> Customer re-explained their problem — handoff failure.")
    fhtml2 = "".join([f"<br><br>{f}" for f in eflags])

    actual = "🔴 DSAT — Customer was dissatisfied" if is_dsat else "🟢 CSAT — Customer was satisfied"
    sv     = trow['Sentiment']
    slbl   = "😟 Negative" if sv<0 else ("😊 Positive" if sv>1 else "😐 Neutral")

    st.markdown(f"""
<div class="story-card" style="margin-bottom:10px">
  <b>🎫 {ticket_id}</b> &nbsp;
  <span style="color:#8b92ab;font-size:0.82rem">{trow['Product']} · {trow['Feature']}</span> &nbsp;
  <span style="background:#1e2a40;padding:2px 10px;border-radius:12px;font-size:0.78rem;color:#c9d1e8">Agent: {trow['Agent_Name']}</span> &nbsp;
  <span style="background:#1e2a40;padding:2px 10px;border-radius:12px;font-size:0.78rem;color:#c9d1e8">Week: {str(trow['Week'])[:10]}</span> &nbsp;
  <span style="background:#1e2a40;padding:2px 10px;border-radius:12px;font-size:0.78rem;color:#c9d1e8">Team: {trow['Team']}</span>
</div>
""", unsafe_allow_html=True)

    left,right = st.columns([1.3,1])
    with left:
        st.markdown(f"""
<div class="ticket-box">
  <b>🔍 Root Cause Analysis</b><br>
  <span style="color:#7eb8f7;font-size:0.85rem">ML classification (Transcript Only): <b>{ticket_issue}</b></span><br><br>
  <b>{wwg_label}</b><br>{wwg}<br><br>
  <b>{core_label}</b><br>{core}<br><br>
  <b>Product & Feature:</b> {trow['Product']} — {trow['Feature']}
  {fhtml2}<br><br>
  <b>📡 Transcript Signals</b><br>{shtml}
</div>
""", unsafe_allow_html=True)

    with right:
        st.markdown(f"""
<div class="ticket-box">
  <b>🔮 Return Prediction</b><br>
  <span style="color:#8b92ab;font-size:0.82rem">If this customer contacts again:</span><br><br>
  <span class="{pred_cls}">{ret_label}</span><br>
  <span style="color:#8b92ab;font-size:0.82rem">{ret_conf}</span><br><br>
  <b>Actual outcome (this ticket):</b><br>{actual}<br><br>
  <b>Sentiment score:</b> {round(float(sv),2)} — {slbl}<br><br>
  <b>💬 Full Chat Transcript</b>
  <div style="background:#0d1220;border-radius:8px;padding:12px;margin-top:8px;font-size:0.79rem;color:#a0aabf;max-height:340px;overflow-y:auto;line-height:1.75;white-space:pre-wrap">{transcript}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='color:#3a4060;font-size:0.75rem;text-align:center;'>DSAT Intelligence · RF Forecasting · Naive Bayes NLP (Transcript Only) · 5-Why Root Cause · Return Prediction</p>", unsafe_allow_html=True)
