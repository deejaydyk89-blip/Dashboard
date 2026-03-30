import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import Counter

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="DSAT Intelligence", layout="wide", page_icon="🧠")

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: #0f1117;
    color: #e8eaf0;
}
h1 { color: #ffffff; font-size: 2rem; font-weight: 700; }
h2, h3 { color: #c9d1e8; }

[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2e3555;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] { color: #7eb8f7; font-size: 1.7rem; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #8b92ab; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }

.section-header {
    background: linear-gradient(90deg,#1e2540,#131826);
    border-left: 4px solid #4f7bf7;
    padding: 10px 18px;
    border-radius: 6px;
    margin: 28px 0 14px 0;
    font-size: 1.05rem;
    font-weight: 600;
    color: #c9d1e8;
}
.sub-header {
    border-left: 3px solid #3a4870;
    padding: 6px 14px;
    margin: 18px 0 10px 0;
    font-size: 0.95rem;
    font-weight: 600;
    color: #a0adc8;
    background: #13182a;
    border-radius: 0 6px 6px 0;
}
.morning-brief {
    background: linear-gradient(135deg,#1a2744,#1a1f2e);
    border: 1px solid #3a4870;
    border-radius: 14px;
    padding: 20px 26px;
    margin-bottom: 20px;
}
.brief-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.12em; color: #7eb8f7; margin-bottom: 8px; }
.brief-body  { font-size: 1.02rem; color: #dce3f5; line-height: 1.8; }

.story-card {
    background: #141926;
    border: 1px solid #2e3555;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 14px;
    line-height: 1.85;
}
.product-story {
    background: #101520;
    border: 1px solid #232d45;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    line-height: 1.8;
}
.prod-name   { font-size: 1rem; font-weight: 700; color: #7eb8f7; margin-bottom: 6px; }
.wow-better  { color: #34d399; font-weight: 600; }
.wow-worse   { color: #f87272; font-weight: 600; }
.wow-same    { color: #8b92ab; }

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

.insight-box  { background: #141926; border: 1px solid #2e3555; border-radius: 12px; padding: 20px 24px; line-height: 1.85; }
.insight-tag  { display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-bottom: 10px; }
.tag-critical  { background: #4a1010; color: #f87272; }
.tag-watchlist { background: #3a2b0a; color: #fbbf24; }
.tag-strong    { background: #0e2e1a; color: #34d399; }
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
st.markdown("<p style='color:#8b92ab;margin-top:-14px;font-size:0.9rem;'>ML-powered agent performance · Product analytics · Ticket root cause · Return prediction</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
file_path = "updated_bpo_customer_experience_dataset.csv"
if not os.path.exists(file_path):
    st.error("❌ CSV not found. Place 'updated_bpo_customer_experience_dataset.csv' in the same folder.")
    st.stop()

df = None
for enc in ["utf-8","latin1","utf-16"]:
    for sep in [",","|",";"]:
        try:
            t = pd.read_csv(file_path, encoding=enc, sep=sep)
            if t.shape[1] > 1:
                df = t; break
        except: continue
    if df is not None: break

if df is None or df.empty:
    st.error("❌ Failed to load dataset"); st.stop()

df.columns        = df.columns.str.strip()
df['Week']        = pd.to_datetime(df['Week'], errors='coerce')
df                = df.dropna(subset=['Week'])
df['DSAT']        = df['Customer_Effortless'].apply(lambda x: 1 if str(x).strip().lower() == "no" else 0)
df['Combined_Text'] = df['Customer_Comment'].fillna('') + ' ' + df['Chat_Transcript'].fillna('')
df['transcript_len']= df['Chat_Transcript'].fillna('').apply(len)
df['comment_len']   = df['Customer_Comment'].fillna('').apply(len)

weeks_sorted = sorted(df['Week'].unique())
curr_week    = weeks_sorted[-1]
prev_week    = weeks_sorted[-2] if len(weeks_sorted) >= 2 else curr_week

# ─────────────────────────────────────────────
# FIX 1 — IMPROVED ISSUE LABELLING (richer, balanced)
# Uses transcript + comment separately for stronger signal
# ─────────────────────────────────────────────
def label_issue(text):
    c = str(text).lower()
    people_kw  = ["rude","angry","unhelpful","attitude","unprofessional","frustrat",
                  "no empathy","escalate","supervisor","bad agent","not listening",
                  "dismissive","condescending","impatient","arrogant","yelled",
                  "didn't care","poor service","terrible agent","worst agent",
                  "should be fired","incompetent","clueless","very rude","so rude"]
    process_kw = ["delay","wait","slow","long","transfer","hold","multiple",
                  "inefficient","not read","keep asking","repeat","again","already told",
                  "waiting too long","waited","put on hold","transferred multiple",
                  "no follow up","no callback","asked twice","third time",
                  "still waiting","hours","days","never resolved","took forever",
                  "long wait","longer than expected","kept waiting"]
    product_kw = ["error","bug","failed","not working","broken","limitation",
                  "glitch","crash","outage","system","technical","doesn't work",
                  "cannot login","app crash","feature missing","not available",
                  "stopped working","software","platform issue","website down",
                  "login issue","password","reset not working","page not loading",
                  "technical issue","system error","server error","keeps crashing"]

    people_score  = sum(1 for w in people_kw  if w in c)
    process_score = sum(1 for w in process_kw if w in c)
    product_score = sum(1 for w in product_kw if w in c)

    max_score = max(people_score, process_score, product_score)
    if max_score == 0:
        return "Other"
    if people_score == max_score:
        return "People"
    elif process_score == max_score:
        return "Process"
    else:
        return "Product"

df['Issue_Label'] = df['Combined_Text'].apply(label_issue)

# ─────────────────────────────────────────────
# FIX 2 — PROPER ML MODEL (Gradient Boosting + class balancing)
# No artificial noise injection — let the data speak
# ─────────────────────────────────────────────
df_labeled = df[df['Issue_Label'] != "Other"].copy()
nlp_accuracy  = 0.0
vectorizer    = None
issue_model   = None
label_dist    = {}
model_report  = ""

if len(df_labeled) > 30:
    label_dist = df_labeled['Issue_Label'].value_counts().to_dict()

    # TF-IDF on combined text (bigrams + unigrams)
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 3),       # trigrams capture "not working well", "kept me waiting"
        sublinear_tf=True,        # log scaling reduces dominance of frequent terms
        min_df=2
    )

    # Gradient Boosting with class_weight equivalent via sample_weight
    le = LabelEncoder()
    y_enc = le.fit_transform(df_labeled['Issue_Label'])
    classes = np.unique(y_enc)
    cw = compute_class_weight('balanced', classes=classes, y=y_enc)
    sample_weights = np.array([cw[y] for y in y_enc])

    X_tfidf = vectorizer.fit_transform(df_labeled['Combined_Text'])

    tr_idx, te_idx = train_test_split(
        np.arange(len(df_labeled)), test_size=0.25,
        stratify=df_labeled['Issue_Label'], random_state=42
    )

    issue_model = LogisticRegression(
        max_iter=500, C=1.0,
        class_weight='balanced',   # KEY FIX — prevents "People" domination
        solver='lbfgs'
    )
    issue_model.fit(
        X_tfidf[tr_idx],
        df_labeled['Issue_Label'].values[tr_idx]
    )

    preds = issue_model.predict(X_tfidf[te_idx])
    nlp_accuracy = accuracy_score(df_labeled['Issue_Label'].values[te_idx], preds)
    model_report = classification_report(
        df_labeled['Issue_Label'].values[te_idx], preds, output_dict=False
    )

    # Re-predict entire dataset with the fixed model
    df['Issue_Label'] = issue_model.predict(vectorizer.transform(df['Combined_Text']))

# ─────────────────────────────────────────────
# RETURN PREDICTION MODEL
# ─────────────────────────────────────────────
df['issue_encoded'] = df['Issue_Label'].map({"People":0,"Process":1,"Product":2,"Other":3}).fillna(3)
return_model  = None
ret_features  = ['issue_encoded','transcript_len','comment_len']

# Add sentiment via simple polarity (no TextBlob dependency issue)
def simple_sentiment(text):
    pos = ["great","excellent","good","happy","satisfied","thank","thanks","love","helpful","resolved","fix","fixed","perfect","awesome","brilliant","quick","fast","easy"]
    neg = ["bad","terrible","awful","horrible","poor","worst","hate","angry","frustrated","useless","broken","failed","slow","rude","unhelpful","annoyed","disgusting","pathetic","waste","never","again"]
    t = str(text).lower()
    score = sum(1 for w in pos if w in t) - sum(1 for w in neg if w in t)
    return score

df['Sentiment'] = df['Combined_Text'].apply(simple_sentiment)
ret_features = ['Sentiment','issue_encoded','transcript_len','comment_len']

if df['DSAT'].sum() > 10:
    return_model = RandomForestClassifier(
        n_estimators=200, random_state=42,
        class_weight='balanced', max_depth=6
    )
    return_model.fit(df[ret_features].fillna(0), df['DSAT'])

# ─────────────────────────────────────────────
# DSAT FORECASTING MODEL (unchanged logic, kept)
# ─────────────────────────────────────────────
weekly_df = df.groupby(['Agent_Name','Week']).agg(
    DSAT_Count=('DSAT','sum'),
    Total_Tickets=('Ticket_ID','count')
).reset_index()

for i in range(1,5):
    weekly_df[f'DSAT_lag_{i}']    = weekly_df.groupby('Agent_Name')['DSAT_Count'].shift(i)
    weekly_df[f'Tickets_lag_{i}'] = weekly_df.groupby('Agent_Name')['Total_Tickets'].shift(i)

weekly_df   = weekly_df.dropna()
lag_features= [f'DSAT_lag_{i}' for i in range(1,5)] + [f'Tickets_lag_{i}' for i in range(1,5)]
X = weekly_df[lag_features]
y = weekly_df['DSAT_Count']
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# ─────────────────────────────────────────────
# AGENT SUMMARY
# ─────────────────────────────────────────────
def compute_agent_summary(wdf, model, feats, y_ref):
    rows = []
    for name, grp in wdf.groupby('Agent_Name'):
        grp = grp.sort_values('Week')
        if len(grp) < 2: continue
        lat, prv = grp.iloc[-1], grp.iloc[-2]
        p_lat = model.predict([lat[feats]])[0]
        p_prv = model.predict([prv[feats]])[0]
        r_lat = (p_lat - y_ref.mean()) / y_ref.std() * 10 + 50
        r_prv = (p_prv - y_ref.mean()) / y_ref.std() * 10 + 50
        rows.append({
            'Agent': name,
            'Avg DSAT': round(grp['DSAT_Count'].mean(),1),
            'Predicted DSAT': int(p_lat),
            'Risk Score': int(r_lat),
            'Risk Δ': round(r_lat - r_prv, 1),
            'Trend': float(lat['DSAT_lag_1'] - lat['DSAT_lag_4'])
        })
    return pd.DataFrame(rows)

agent_summary_df = compute_agent_summary(weekly_df, rf_model, lag_features, y)
med = agent_summary_df['Avg DSAT'].median()

def assign_quadrant(row):
    hi    = row['Avg DSAT'] >= med
    worse = row['Trend'] > 0
    if hi and worse:       return "🔴 Intervene Now"
    elif not hi and worse: return "🟡 Watch Closely"
    elif hi and not worse: return "🟠 Coach & Monitor"
    else:                  return "🟢 Acknowledge"

agent_summary_df['Focus Zone'] = agent_summary_df.apply(assign_quadrant, axis=1)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def names_html(lst): return "<br>".join([f"• {a}" for a in lst]) or "None"

def ppp_counts(texts):
    cats = ["People","Process","Product"]
    text_list = list(texts)
    if len(text_list) > 0 and vectorizer and issue_model:
        preds = issue_model.predict(vectorizer.transform(text_list))
        cnt   = Counter(preds)
    else:
        cnt = Counter([label_issue(t) for t in text_list])
    total = sum(cnt.get(c,0) for c in cats) or 1
    return cnt, total

def wow_arrow(delta):
    if delta > 0:   return f"<span class='wow-worse'>⬆ +{int(delta)} DSAT</span>"
    elif delta < 0: return f"<span class='wow-better'>⬇ {int(delta)} DSAT</span>"
    else:           return f"<span class='wow-same'>➡ No change</span>"

kw_map = {
    "People":  ["rude","unhelpful","attitude","angry","unprofessional","escalate","supervisor","frustrat","not listening","dismissive"],
    "Process": ["delay","wait","slow","transfer","hold","multiple","inefficient","keep asking","repeat","already told","again"],
    "Product": ["error","bug","failed","not working","broken","limitation","glitch","crash","outage","technical","doesn't work"]
}

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

# Label distribution debug info
label_dist_str = " · ".join([f"{k}: {v}" for k, v in label_dist.items()]) if label_dist else "N/A"

st.markdown(f"""
<div class="morning-brief">
  <div class="brief-title">🗓️ Auto-generated · Current Situation</div>
  <div class="brief-body">
    <b>{n_critical} agent(s) need immediate intervention</b> · {n_watch} on watchlist · {n_worsening} worsening this week · {n_improving} recovering<br>
    📌 Immediate focus: <b>{top_names}</b> &nbsp;|&nbsp; 📊 Team predicted DSAT: <b>{team_pred}</b><br>
    <span style="color:#5a6484;font-size:0.8rem">ML Label Distribution: {label_dist_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("ML Model Accuracy",   f"{round(nlp_accuracy*100,1)}%")
c2.metric("Critical Agents",     int(n_critical),  delta=f"{n_critical} need action", delta_color="inverse")
c3.metric("Worsening This Week", int(n_worsening), delta_color="inverse")
c4.metric("Recovering",          int(n_improving), delta_color="normal")
c5.metric("Team Predicted DSAT", int(team_pred))

# Model report expander
if model_report:
    with st.expander("🔬 ML Model Classification Report"):
        st.code(model_report)

st.markdown('<div class="section-header">🎯 Manager Focus Matrix</div>', unsafe_allow_html=True)
q   = {z: agent_summary_df[agent_summary_df['Focus Zone']==z]['Agent'].tolist()
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
# NEW SECTION — TEAM WoW PRODUCT & FEATURE SUMMARY
# What is impacting us across all agents, week on week
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Week-on-Week Team Summary — Product & Feature Impact</div>', unsafe_allow_html=True)
st.caption(f"Comparing {str(prev_week)[:10]} → {str(curr_week)[:10]} across all agents · Focus: where to direct training and attention")

# ── Product level WoW
prod_curr = df[df['Week']==curr_week].groupby('Product').agg(
    DSAT_curr=('DSAT','sum'), Tickets_curr=('Ticket_ID','count')
).reset_index()
prod_prev = df[df['Week']==prev_week].groupby('Product').agg(
    DSAT_prev=('DSAT','sum'), Tickets_prev=('Ticket_ID','count')
).reset_index()
prod_wow = pd.merge(prod_curr, prod_prev, on='Product', how='outer').fillna(0)
prod_wow['DSAT_delta']    = prod_wow['DSAT_curr'] - prod_wow['DSAT_prev']
prod_wow['DSAT_rate_curr']= (prod_wow['DSAT_curr'] / prod_wow['Tickets_curr'].replace(0,1) * 100).round(1)
prod_wow['DSAT_rate_prev']= (prod_wow['DSAT_prev'] / prod_wow['Tickets_prev'].replace(0,1) * 100).round(1)
prod_wow['Rate_delta']    = (prod_wow['DSAT_rate_curr'] - prod_wow['DSAT_rate_prev']).round(1)
prod_wow = prod_wow.sort_values('DSAT_delta', ascending=False)

st.markdown('<div class="sub-header">📦 Product — DSAT Movement This Week</div>', unsafe_allow_html=True)

for _, row in prod_wow.iterrows():
    delta     = int(row['DSAT_delta'])
    rate_delta= float(row['Rate_delta'])
    if delta > 0:
        trend_cls = "trend-up";   arr = "⬆"; urgency = "🔴 Needs Attention"
    elif delta < 0:
        trend_cls = "trend-down"; arr = "⬇"; urgency = "🟢 Improving"
    else:
        trend_cls = "trend-flat"; arr = "➡"; urgency = "🟡 Stable"

    # Feature breakdown for this product this week
    feat_curr = df[(df['Week']==curr_week)&(df['Product']==row['Product'])&(df['DSAT']==1)]['Feature'].value_counts().head(4)
    feat_html = " &nbsp;·&nbsp; ".join([f"<span style='color:#f87272'>{f}</span> ({v})" for f, v in feat_curr.items()]) or "<span style='color:#5a6484'>No DSAT this week</span>"

    # PPP breakdown for this product (current week)
    prod_texts = df[(df['Week']==curr_week)&(df['Product']==row['Product'])&(df['DSAT']==1)]['Combined_Text']
    ppp, total_p = ppp_counts(prod_texts)
    ppp_html = ""
    for cat, cls in [("People","pill-people"),("Process","pill-process"),("Product","pill-product")]:
        v   = ppp.get(cat,0)
        pct = round(v/total_p*100) if total_p > 0 else 0
        ppp_html += f"<span class='ppp-pill {cls}'>{cat}: {v} ({pct}%)</span>"

    st.markdown(f"""
<div class="wow-card">
  <b style="color:#7eb8f7;font-size:1rem">📦 {row['Product']}</b>
  &nbsp;&nbsp;<span style="background:#1e2a40;padding:2px 10px;border-radius:12px;font-size:0.78rem;color:#c9d1e8">{urgency}</span>
  <br><br>
  <table style="width:100%;border-spacing:0 4px">
    <tr>
      <td style="color:#5a6484;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.07em;width:20%">DSAT This Week</td>
      <td style="color:#5a6484;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.07em;width:20%">DSAT Change</td>
      <td style="color:#5a6484;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.07em;width:20%">DSAT Rate (curr)</td>
      <td style="color:#5a6484;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.07em;width:20%">Rate Change</td>
      <td style="color:#5a6484;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.07em;width:20%">Tickets This Week</td>
    </tr>
    <tr>
      <td style="font-size:1.25rem;font-weight:700;color:#f87272">{int(row['DSAT_curr'])}</td>
      <td style="font-size:1.25rem;font-weight:700" class="{trend_cls}">{arr} {delta:+d}</td>
      <td style="font-size:1.25rem;font-weight:700;color:#c9d1e8">{row['DSAT_rate_curr']}%</td>
      <td style="font-size:1.25rem;font-weight:700" class="{trend_cls}">{rate_delta:+.1f}%</td>
      <td style="font-size:1.25rem;font-weight:700;color:#c9d1e8">{int(row['Tickets_curr'])}</td>
    </tr>
  </table>
  <br>
  <span style="color:#5a6484;font-size:0.77rem;text-transform:uppercase;letter-spacing:0.07em">Top DSAT features this week</span><br>
  <span style="font-size:0.84rem">{feat_html}</span>
  <br><br>
  <div class="ppp-row">{ppp_html}</div>
</div>
""", unsafe_allow_html=True)

# ── Feature level WoW — top movers
st.markdown('<div class="sub-header">🔧 Feature — Top DSAT Movers This Week (All Products)</div>', unsafe_allow_html=True)

feat_curr_all = df[df['Week']==curr_week].groupby(['Product','Feature'])['DSAT'].sum().reset_index(name='DSAT_curr')
feat_prev_all = df[df['Week']==prev_week].groupby(['Product','Feature'])['DSAT'].sum().reset_index(name='DSAT_prev')
feat_wow_all  = pd.merge(feat_curr_all, feat_prev_all, on=['Product','Feature'], how='outer').fillna(0)
feat_wow_all['Delta'] = feat_wow_all['DSAT_curr'] - feat_wow_all['DSAT_prev']
feat_wow_all['Label'] = feat_wow_all['Product'] + ' › ' + feat_wow_all['Feature']

top_worsening = feat_wow_all.sort_values('Delta', ascending=False).head(8)
top_improving = feat_wow_all.sort_values('Delta').head(8)

fw1, fw2 = st.columns(2)
with fw1:
    st.caption("🔴 Most Worsened Features")
    display_w = top_worsening[['Label','DSAT_curr','DSAT_prev','Delta']].rename(
        columns={'Label':'Product › Feature','DSAT_curr':'This Week','DSAT_prev':'Last Week','Delta':'Change'})
    st.dataframe(display_w, use_container_width=True, hide_index=True)
with fw2:
    st.caption("🟢 Most Improved Features")
    display_i = top_improving[['Label','DSAT_curr','DSAT_prev','Delta']].rename(
        columns={'Label':'Product › Feature','DSAT_curr':'This Week','DSAT_prev':'Last Week','Delta':'Change'})
    st.dataframe(display_i, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# SECTION 2 — AGENT STORY
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">👤 Agent Deep Dive</div>', unsafe_allow_html=True)

agent = st.selectbox("Select Agent", sorted(weekly_df['Agent_Name'].unique()), key="agent_sel")

ag_weekly = weekly_df[weekly_df['Agent_Name']==agent].sort_values('Week')
ag_all    = df[df['Agent_Name']==agent]
ag_dsat   = ag_all[ag_all['DSAT']==1]

if len(ag_weekly) < 2:
    st.warning("Not enough weekly data for this agent.")
else:
    latest = ag_weekly.iloc[-1]
    prev   = ag_weekly.iloc[-2]

    pred_now   = rf_model.predict([latest[lag_features]])[0]
    pred_prev  = rf_model.predict([prev[lag_features]])[0]
    risk_now   = (pred_now  - y.mean()) / y.std() * 10 + 50
    risk_prev  = (pred_prev - y.mean()) / y.std() * 10 + 50
    risk_delta = risk_now - risk_prev
    trend      = float(latest['DSAT_lag_1'] - latest['DSAT_lag_4'])
    focus_zone = agent_summary_df[agent_summary_df['Agent']==agent]['Focus Zone'].values[0] \
                 if agent in agent_summary_df['Agent'].values else "N/A"

    ag_actual  = ag_weekly['DSAT_Count'].values
    ag_preds   = rf_model.predict(ag_weekly[lag_features])
    ag_var     = np.var(ag_actual)
    ag_r2      = 1 - np.var(ag_actual - ag_preds)/ag_var if ag_var != 0 else 0
    ag_mae     = mean_absolute_error(ag_actual, ag_preds)

    tix_curr  = int(ag_all[ag_all['Week']==curr_week].shape[0])
    tix_prev  = int(ag_all[ag_all['Week']==prev_week].shape[0])
    dsat_curr = int(ag_all[(ag_all['Week']==curr_week)&(ag_all['DSAT']==1)].shape[0])
    dsat_prev = int(ag_all[(ag_all['Week']==prev_week)&(ag_all['DSAT']==1)].shape[0])

    risk_col = "#f87272" if risk_delta > 2 else ("#34d399" if risk_delta < -2 else "#8b92ab")
    risk_dir = "⬆ Worsening" if risk_delta > 2 else ("⬇ Improving" if risk_delta < -2 else "➡ Stable")
    trend_txt= "rising week-over-week" if trend > 0 else ("improving" if trend < 0 else "stable")

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
      <td style="font-size:1.4rem;font-weight:700;color:{risk_col}">{risk_delta:+.1f} <span style="font-size:0.82rem">{risk_dir}</span></td>
      <td style="font-size:1.4rem;font-weight:700;color:#7eb8f7">{int(pred_now)}</td>
      <td style="font-size:1.4rem;font-weight:700;color:#c9d1e8">{tix_curr} <span style="font-size:0.8rem;color:#8b92ab">({tix_curr-tix_prev:+d} vs prev)</span></td>
      <td style="font-size:1.4rem;font-weight:700;color:#f87272">{dsat_curr} <span style="font-size:0.8rem;color:#8b92ab">({dsat_curr-dsat_prev:+d} vs prev)</span></td>
    </tr>
  </table>
  <br>
  <span style="color:#8b92ab;font-size:0.87rem">
    DSAT trend is <b style="color:#e8eaf0">{trend_txt}</b> &nbsp;·&nbsp;
    Prediction R²: <b style="color:#e8eaf0">{round(ag_r2,2)}</b> &nbsp;·&nbsp;
    Avg prediction error: <b style="color:#e8eaf0">{round(ag_mae,2)}</b>
  </span>
</div>
""", unsafe_allow_html=True)

    # ── Trend charts
    st.markdown('<div class="sub-header">📈 Historical DSAT Trend & Risk Score</div>', unsafe_allow_html=True)
    ch1,ch2 = st.columns(2)
    with ch1:
        st.caption("Weekly DSAT Count")
        st.line_chart(ag_weekly.set_index('Week')[['DSAT_Count']].rename(columns={'DSAT_Count':'DSAT Count'}))
    with ch2:
        st.caption("Risk Score over Time")
        risk_ts = pd.Series(
            [(rf_model.predict([r[lag_features]])[0] - y.mean())/y.std()*10+50 for _,r in ag_weekly.iterrows()],
            index=ag_weekly['Week'].values, name='Risk Score'
        )
        st.line_chart(risk_ts)

    # ══════════════════════════════════════════════════════════
    # FIX 3 — AGENT: PRIMARY DSAT PRODUCT FIRST, THEN WoW TREND
    # Shows the #1 problem product prominently + week trend chart
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="sub-header">🎯 Primary DSAT Driver — Which Product Is Causing the Most Issues?</div>', unsafe_allow_html=True)

    # Compute all-time DSAT per product for this agent
    prod_dsat_totals = ag_dsat.groupby('Product')['DSAT'].count().sort_values(ascending=False)

    if prod_dsat_totals.empty:
        st.info("No DSAT tickets found for this agent.")
    else:
        top_prod = prod_dsat_totals.index[0]
        top_prod_count = int(prod_dsat_totals.iloc[0])
        total_dsat_ag  = int(ag_dsat.shape[0])
        top_prod_pct   = round(top_prod_count/total_dsat_ag*100) if total_dsat_ag > 0 else 0

        # WoW trend for top product
        top_prod_weekly = ag_all[ag_all['Product']==top_prod].groupby('Week').agg(
            DSAT_count=('DSAT','sum'), Ticket_count=('Ticket_ID','count')
        ).reset_index().sort_values('Week')

        top_curr_dsat = int(ag_all[(ag_all['Week']==curr_week)&(ag_all['Product']==top_prod)&(ag_all['DSAT']==1)].shape[0])
        top_prev_dsat = int(ag_all[(ag_all['Week']==prev_week)&(ag_all['Product']==top_prod)&(ag_all['DSAT']==1)].shape[0])
        top_delta     = top_curr_dsat - top_prev_dsat
        top_trend_cls = "trend-up" if top_delta > 0 else ("trend-down" if top_delta < 0 else "trend-flat")
        top_arr       = "⬆" if top_delta > 0 else ("⬇" if top_delta < 0 else "➡")

        # Top features for this product
        top_feats = ag_dsat[ag_dsat['Product']==top_prod]['Feature'].value_counts().head(5)

        # Training recommendation
        if top_delta > 0:
            train_rec = f"⚠️ DSAT on <b>{top_prod}</b> is actively rising. A product refresher training is <b>recommended immediately</b>."
            train_col = "#f87272"
        elif top_delta < 0:
            train_rec = f"✅ DSAT on <b>{top_prod}</b> is improving week-on-week. Monitor and maintain coaching cadence."
            train_col = "#34d399"
        else:
            train_rec = f"📌 DSAT on <b>{top_prod}</b> is flat. Watch closely — volume still warrants attention."
            train_col = "#facc15"

        feat_rows_html = "".join([
            f"<span style='color:#8b92ab'>{f}:</span> <span style='color:#f87272'>{v} DSAT</span> &nbsp;·&nbsp; "
            for f, v in top_feats.items()
        ]).rstrip(" &nbsp;·&nbsp; ")

        st.markdown(f"""
<div class="wow-card" style="border-color:#4f7bf7">
  <b style="color:#7eb8f7;font-size:1.1rem">🎯 Primary Problem: {top_prod}</b>
  &nbsp;&nbsp;<span style="background:#1e2a40;padding:2px 10px;border-radius:12px;font-size:0.78rem;color:#c9d1e8">{top_prod_pct}% of this agent's total DSAT</span>
  <br><br>
  <table style="width:100%;border-spacing:0 4px">
    <tr>
      <td style="color:#5a6484;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.07em;width:25%">All-Time DSAT</td>
      <td style="color:#5a6484;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.07em;width:25%">This Week DSAT</td>
      <td style="color:#5a6484;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.07em;width:25%">Last Week DSAT</td>
      <td style="color:#5a6484;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.07em;width:25%">WoW Change</td>
    </tr>
    <tr>
      <td style="font-size:1.3rem;font-weight:700;color:#f87272">{top_prod_count}</td>
      <td style="font-size:1.3rem;font-weight:700;color:#c9d1e8">{top_curr_dsat}</td>
      <td style="font-size:1.3rem;font-weight:700;color:#c9d1e8">{top_prev_dsat}</td>
      <td style="font-size:1.3rem;font-weight:700" class="{top_trend_cls}">{top_arr} {top_delta:+d}</td>
    </tr>
  </table>
  <br>
  <span style="color:#5a6484;font-size:0.77rem;text-transform:uppercase;letter-spacing:0.07em">Top features causing DSAT (all-time)</span><br>
  <span style="font-size:0.84rem">{feat_rows_html}</span>
  <br><br>
  <span style="color:{train_col};font-size:0.9rem">{train_rec}</span>
</div>
""", unsafe_allow_html=True)

        # WoW Trend chart for the primary product
        if not top_prod_weekly.empty:
            st.caption(f"📈 Weekly DSAT trend — {top_prod} (for {agent})")
            st.line_chart(top_prod_weekly.set_index('Week')[['DSAT_count']].rename(columns={'DSAT_count':f'{top_prod} DSAT'}))

        # All other products ranked
        st.markdown('<div class="sub-header">📦 All Products Ranked by DSAT (This Agent)</div>', unsafe_allow_html=True)

        for prod, total_d in prod_dsat_totals.items():
            curr_d = int(ag_all[(ag_all['Week']==curr_week)&(ag_all['Product']==prod)&(ag_all['DSAT']==1)].shape[0])
            prev_d = int(ag_all[(ag_all['Week']==prev_week)&(ag_all['Product']==prod)&(ag_all['DSAT']==1)].shape[0])
            delta  = curr_d - prev_d
            curr_t = int(ag_all[(ag_all['Week']==curr_week)&(ag_all['Product']==prod)].shape[0])
            prev_t = int(ag_all[(ag_all['Week']==prev_week)&(ag_all['Product']==prod)].shape[0])

            f_curr = ag_all[(ag_all['Week']==curr_week)&(ag_all['Product']==prod)&(ag_all['DSAT']==1)]['Feature'].value_counts()
            f_prev = ag_all[(ag_all['Week']==prev_week)&(ag_all['Product']==prod)&(ag_all['DSAT']==1)]['Feature'].value_counts()
            all_f  = set(f_curr.index) | set(f_prev.index)

            feat_rows = []
            for f in sorted(all_f):
                fc = int(f_curr.get(f,0))
                fp = int(f_prev.get(f,0))
                fd = fc - fp
                col = "#f87272" if fd > 0 else ("#34d399" if fd < 0 else "#8b92ab")
                arr = "⬆" if fd > 0 else ("⬇" if fd < 0 else "➡")
                feat_rows.append(f"<span style='color:#8b92ab'>{f}:</span> <span style='color:{col}'>{arr} {fc} DSAT ({fd:+d})</span>")

            feat_html = " &nbsp;·&nbsp; ".join(feat_rows) if feat_rows else "<span style='color:#5a6484'>No DSAT tickets this week</span>"

            prod_texts = ag_all[(ag_all['Product']==prod)&(ag_all['DSAT']==1)]['Combined_Text']
            ppp, total_p = ppp_counts(prod_texts)
            ppp_pills = ""
            for cat, cls in [("People","pill-people"),("Process","pill-process"),("Product","pill-product")]:
                v   = ppp.get(cat,0)
                pct = round(v/total_p*100) if total_p > 0 else 0
                ppp_pills += f"<span class='ppp-pill {cls}'>{cat}: {v} ({pct}%)</span>"

            is_primary = "⭐ Primary Problem" if prod == top_prod else ""
            border_col = "#4f7bf7" if prod == top_prod else "#232d45"

            st.markdown(f"""
<div class="product-story" style="border-color:{border_col}">
  <div class="prod-name">📦 {prod} &nbsp;<span style="color:#facc15;font-size:0.78rem">{is_primary}</span>
    &nbsp;<span style="color:#8b92ab;font-size:0.78rem">All-time DSAT: {int(total_d)}</span>
  </div>
  <span style="color:#8b92ab;font-size:0.8rem">DSAT change this week:</span> {wow_arrow(delta)}
  &nbsp;&nbsp;
  <span style="color:#8b92ab;font-size:0.8rem">Tickets:</span>
  <span style="color:#c9d1e8;font-size:0.85rem">{curr_t} ({curr_t-prev_t:+d} vs prev week)</span>
  <br><br>
  <span style="color:#5a6484;font-size:0.77rem;text-transform:uppercase;letter-spacing:0.08em">Feature breakdown (DSAT — this week vs previous)</span><br>
  <span style="font-size:0.85rem">{feat_html}</span>
  <br><br>
  <span style="color:#5a6484;font-size:0.77rem;text-transform:uppercase;letter-spacing:0.08em">Issue root cause (People / Process / Product)</span><br>
  <div class="ppp-row">{ppp_pills}</div>
</div>
""", unsafe_allow_html=True)

    # ── Overall PPP
    st.markdown('<div class="sub-header">🔍 Overall Issue Breakdown — People / Process / Product</div>', unsafe_allow_html=True)
    overall_ppp, overall_total = ppp_counts(ag_dsat['Combined_Text'])

    o1,o2,o3 = st.columns(3)
    for col_obj, cat, clr in [(o1,"People","#fb923c"),(o2,"Process","#facc15"),(o3,"Product","#34d399")]:
        v   = overall_ppp.get(cat,0)
        pct = round(v/overall_total*100) if overall_total > 0 else 0
        col_obj.metric(f"{cat} Issues", v, delta=f"{pct}% of DSAT")

    dominant_cat = max(overall_ppp, key=overall_ppp.get) if overall_ppp else "People"

    matched = []
    for txt in ag_dsat['Combined_Text']:
        for kw in kw_map.get(dominant_cat,[]):
            if kw in str(txt).lower():
                matched.append(kw); break
    if matched:
        kw_df = pd.DataFrame(Counter(matched).most_common(8), columns=["Keyword","Occurrences"])
        st.caption(f"Top keywords driving **{dominant_cat}** complaints for {agent}")
        st.dataframe(kw_df, use_container_width=True, hide_index=True)

    # ── AI Insight + Coaching
    st.markdown('<div class="sub-header">🤖 AI Insight & Coaching Plan</div>', unsafe_allow_html=True)

    if risk_now < 45:   level, tag_cls = "Strong Performer", "tag-strong"
    elif risk_now < 60: level, tag_cls = "Watchlist",        "tag-watchlist"
    else:               level, tag_cls = "Critical",          "tag-critical"

    if trend > 0:
        trend_msg = "DSAT is rising week-over-week — this agent is on a declining performance path."
        sent_msg  = "Customer sentiment is turning more negative in line with the DSAT increase."
    elif trend < 0:
        trend_msg = "DSAT is decreasing — clear signs of performance recovery are visible."
        sent_msg  = "Customer sentiment is improving alongside the recovery."
    else:
        trend_msg = "DSAT is stable with no significant movement recently."
        sent_msg  = "Customer sentiment is relatively flat."

    coaching_map = {
        ("up","People"):   ["🗣️ Run empathy & active listening workshops","🎧 Review recent chat recordings for tone","📋 Introduce post-interaction confirmation checklist"],
        ("up","Process"):  ["⏱️ Audit average hold & transfer time","🔁 Train on first-contact resolution techniques","📞 Reduce unnecessary escalations via product knowledge"],
        ("up","Product"):  ["📚 Refresh product & system knowledge training","🐛 Create escalation protocol for recurring issues","🤝 Pair with a top performer for knowledge shadowing"],
        ("down","People"): ["✅ Maintain empathy standards","🌟 Share communication best practices with team","📊 Keep monitoring sentiment weekly"],
        ("down","Process"):["✅ Keep first-contact resolution rate up","📊 Monitor transfer rates","🌟 Recognise and share efficiency improvements"],
        ("down","Product"):["✅ Continue product knowledge refresh","🐛 Keep logging product issues for the tech team","🌟 Acknowledge improvement in resolution quality"],
    }
    direction  = "up" if trend > 0 else "down"
    items      = coaching_map.get((direction, dominant_cat), ["✅ Maintain current performance","📊 Monitor trends weekly","🌟 Share best practices"])
    coach_html = "".join([f"<div class='coaching-item'>{i}</div>" for i in items])
    dom_pct    = round(overall_ppp.get(dominant_cat,0)/overall_total*100) if overall_total > 0 else 0

    # Training recommendation based on primary product
    primary_prod_name = prod_dsat_totals.index[0] if not prod_dsat_totals.empty else "N/A"
    training_note = f"<br><br><b>🏋️ Training Priority:</b> Focus refresher training on <b style='color:#7eb8f7'>{primary_prod_name}</b> — this product accounts for the highest share of DSAT for this agent."

    st.markdown(f"""
<div class="insight-box">
  <div><span class='insight-tag {tag_cls}'>{level}</span></div>
  <b style='font-size:1.05rem'>{agent}</b><br><br>
  <b>📊 Summary</b><br>
  {trend_msg}<br>{sent_msg}<br><br>
  <b>🔍 Primary Root Cause: {dominant_cat}</b><br>
  {dominant_cat} issues are the dominant driver of DSAT for this agent, accounting for <b>{dom_pct}%</b> of all flagged complaints.
  {"Immediate intervention is recommended." if trend > 0 else "Performance improvements are visible — maintain the momentum."}
  {training_note}<br><br>
  <b>💡 Recommended Coaching Actions</b>
  <div class='coaching-card'>{coach_html}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SECTION 3 — TICKET DEEP DIVE
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎫 Ticket Deep Dive — Root Cause & Return Prediction</div>', unsafe_allow_html=True)
st.caption("Select from dropdown or type a Ticket ID manually to investigate root cause, transcript signals and return likelihood.")

col_dd, col_manual = st.columns([2,1])
with col_dd:
    selected_dropdown = st.selectbox("Select Ticket ID", sorted(df['Ticket_ID'].unique()), key="tkt_dd")
with col_manual:
    manual_id = st.text_input("Or enter Ticket ID manually", placeholder="e.g. TKT-100042", key="tkt_manual")

ticket_id = manual_id.strip() if manual_id.strip() else selected_dropdown
tkt_rows  = df[df['Ticket_ID']==ticket_id]

if tkt_rows.empty:
    st.warning(f"Ticket `{ticket_id}` not found in dataset.")
else:
    trow       = tkt_rows.iloc[0]
    combined   = str(trow['Customer_Comment']) + ' ' + str(trow['Chat_Transcript'])
    transcript = str(trow['Chat_Transcript'])
    comment    = str(trow['Customer_Comment'])

    if vectorizer and issue_model:
        ticket_issue = issue_model.predict(vectorizer.transform([combined]))[0]
    else:
        ticket_issue = label_issue(combined)

    if return_model:
        feat_row  = [[trow['Sentiment'], trow['issue_encoded'], trow['transcript_len'], trow['comment_len']]]
        dsat_prob = return_model.predict_proba(feat_row)[0][1]
        ret_label = "🔴 Likely DSAT on Return" if dsat_prob >= 0.5 else "🟢 Likely CSAT on Return"
        ret_conf  = f"{round(max(dsat_prob,1-dsat_prob)*100,1)}% confidence"
        pred_cls  = "pred-dsat" if dsat_prob >= 0.5 else "pred-csat"
    else:
        ret_label, ret_conf, pred_cls = "⚪ Unavailable","","pred-csat"

    signals = []
    tl = transcript.lower()
    if any(w in tl for w in ["escalate","supervisor","manager","unacceptable","not acceptable"]):
        signals.append("⚠️ Customer requested escalation or supervisor")
    if any(w in tl for w in ["why","already told","again","not reading","multiple times","keep asking","frustrated"]):
        signals.append("😤 High frustration — customer had to repeat their issue")
    if any(w in tl for w in ["no solution","not at the moment","product limitation","cannot","unable to resolve","no fix"]):
        signals.append("❌ Issue left unresolved — no fix was offered")
    if any(w in tl for w in ["wait","long","delay","slow","hold"]):
        signals.append("⏱️ Wait time or delay complaint present")
    if any(w in tl for w in ["already explained","told you","asked before","multiple times"]):
        signals.append("🔁 Repeat effort detected — handoff or process failure")
    if not signals:
        signals.append("✅ No major distress signals detected")

    signals_html = "".join([f"<div style='padding:5px 0;color:#c9d1e8;border-bottom:1px solid #1a2235'>{s}</div>" for s in signals])

    if ticket_issue == "People":
        what_went_wrong = "The agent's communication style and tone negatively impacted the customer. The interaction lacked empathy or professionalism."
        core_issue = "Agent behaviour or attitude was the primary driver of dissatisfaction."
    elif ticket_issue == "Process":
        what_went_wrong = "The support process broke down — excessive wait times, unnecessary transfers, or agent failed to own the issue end-to-end."
        core_issue = "Operational inefficiency or broken workflow caused the poor experience."
    else:
        what_went_wrong = "The customer encountered a product bug, system error, or feature limitation that could not be resolved. A structural product gap."
        core_issue = "A product or technical gap drove the dissatisfaction, not agent behaviour alone."

    extra_flags = []
    if any(w in tl for w in ["escalate","supervisor"]):
        extra_flags.append("<b>Escalation flag:</b> Customer demanded supervisor — high-severity signal.")
    if any(w in tl for w in ["no solution","cannot","unable","not at the moment"]):
        extra_flags.append("<b>Resolution status:</b> Issue was NOT resolved. High probability of re-contact.")
    if any(w in tl for w in ["already told","multiple times","keep asking"]):
        extra_flags.append("<b>Repeat effort:</b> Customer had to re-explain — handoff or case note failure.")

    flags_html = "".join([f"<br><br>{f}" for f in extra_flags])
    actual_outcome = "🔴 DSAT — Customer was dissatisfied" if trow['DSAT']==1 else "🟢 CSAT — Customer was satisfied"
    sent_val = trow['Sentiment']
    sent_lbl = "😟 Negative" if sent_val < 0 else ("😊 Positive" if sent_val > 1 else "😐 Neutral")

    st.markdown(f"""
<div class="story-card" style="margin-bottom:10px">
  <b style="font-size:1rem">🎫 {ticket_id}</b>
  &nbsp;&nbsp;<span style="color:#8b92ab;font-size:0.82rem">{trow['Product']} · {trow['Feature']}</span>
  &nbsp;&nbsp;<span style="background:#1e2a40;padding:2px 10px;border-radius:12px;font-size:0.78rem;color:#c9d1e8">Agent: {trow['Agent_Name']}</span>
  &nbsp;&nbsp;<span style="background:#1e2a40;padding:2px 10px;border-radius:12px;font-size:0.78rem;color:#c9d1e8">Week: {str(trow['Week'])[:10]}</span>
  &nbsp;&nbsp;<span style="background:#1e2a40;padding:2px 10px;border-radius:12px;font-size:0.78rem;color:#c9d1e8">Team: {trow['Team']}</span>
</div>
""", unsafe_allow_html=True)

    left, right = st.columns([1.3,1])
    with left:
        st.markdown(f"""
<div class="ticket-box">
  <b>🔍 Root Cause Analysis</b><br>
  <span style="color:#7eb8f7;font-size:0.85rem">Issue classified as: <b>{ticket_issue}</b></span><br><br>
  <b>What went wrong:</b><br>{what_went_wrong}<br><br>
  <b>Core issue:</b><br>{core_issue}<br><br>
  <b>Customer's own words:</b><br>
  <span style="color:#aab4cc;font-style:italic">"{comment[:300]}{'...' if len(comment)>300 else ''}"</span><br><br>
  <b>Product & Feature:</b> {trow['Product']} — {trow['Feature']}
  {flags_html}
  <br><br>
  <b>📡 Transcript Signals</b><br>{signals_html}
</div>
""", unsafe_allow_html=True)

    with right:
        st.markdown(f"""
<div class="ticket-box">
  <b>🔮 Return Prediction</b><br>
  <span style="color:#8b92ab;font-size:0.82rem">If this customer contacts again, will it be DSAT or CSAT?</span><br><br>
  <span class="{pred_cls}">{ret_label}</span><br>
  <span style="color:#8b92ab;font-size:0.82rem">{ret_conf}</span><br><br>
  <b>Actual outcome (this ticket):</b><br>{actual_outcome}<br><br>
  <b>Sentiment score:</b> {round(float(sent_val),2)} — {sent_lbl}<br><br>
  <b>💬 Full Chat Transcript</b>
  <div style="background:#0d1220;border-radius:8px;padding:12px;margin-top:8px;font-size:0.79rem;color:#a0aabf;max-height:340px;overflow-y:auto;line-height:1.75;white-space:pre-wrap">{transcript}</div>
</div>
""", unsafe_allow_html=True)

# ── Footer
st.markdown("---")
st.markdown("<p style='color:#3a4060;font-size:0.75rem;text-align:center;'>DSAT Intelligence · Random Forest + TF-IDF (Balanced Multinomial LR) · People / Process / Product · Transcript Analysis · Return Prediction</p>", unsafe_allow_html=True)
