# ================================
# DSAT INTELLIGENCE — FINAL PRO
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import Counter

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

import google.generativeai as genai

# ─────────────────────────────
# API SETUP
# ─────────────────────────────
API_KEY = os.getenv(AIzaSyACy5Mq1U-NNgRs3KcDPW-8dDcB0ORGZ_8)
if API_KEY:
    genai.configure(api_key=API_KEY)
    gemini = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini = None

# ─────────────────────────────
# PAGE
# ─────────────────────────────
st.set_page_config(page_title="DSAT Intelligence PRO", layout="wide")
st.title("🧠 DSAT Intelligence PRO")

# ─────────────────────────────
# LOAD DATA
# ─────────────────────────────
df = pd.read_csv("updated_bpo_customer_experience_dataset.csv")

df['Week'] = pd.to_datetime(df['Week'], errors='coerce')
df = df.dropna(subset=['Week'])

df['DSAT'] = df['Customer_Effortless'].apply(lambda x: 1 if str(x).lower()=="no" else 0)
df['Combined_Text'] = df['Customer_Comment'].fillna('') + " " + df['Chat_Transcript'].fillna('')

# ─────────────────────────────
# ISSUE LABELING
# ─────────────────────────────
def label_issue(text):
    t = text.lower()
    if any(w in t for w in ["rude","attitude","not listening"]): return "People"
    if any(w in t for w in ["wait","delay","transfer"]): return "Process"
    if any(w in t for w in ["error","bug","not working","failed","limitation"]): return "Product"
    return "Other"

df['Issue_Label'] = df['Combined_Text'].apply(label_issue)

# ─────────────────────────────
# ML MODEL + CV
# ─────────────────────────────
df_labeled = df[(df['Issue_Label']!="Other") & (df['DSAT']==1)]

vectorizer = TfidfVectorizer(max_features=4000)
X = vectorizer.fit_transform(df_labeled['Combined_Text'])
y = df_labeled['Issue_Label']

skf = StratifiedKFold(n_splits=5, shuffle=True)

y_true, y_pred = [], []

for tr, te in skf.split(X, y):
    m = LogisticRegression(max_iter=300)
    m.fit(X[tr], y.iloc[tr])
    preds = m.predict(X[te])
    y_true.extend(y.iloc[te])
    y_pred.extend(preds)

accuracy = accuracy_score(y_true, y_pred)

st.metric("ML Accuracy (CV)", f"{round(accuracy*100,1)}%")

st.expander("Model Report").code(classification_report(y_true, y_pred))

# FINAL MODEL
model = LogisticRegression(max_iter=300)
model.fit(X, y)

df['Issue_Label'] = model.predict(vectorizer.transform(df['Combined_Text']))

# ─────────────────────────────
# PPP DISTRIBUTION
# ─────────────────────────────
ppp = df[df['DSAT']==1]['Issue_Label'].value_counts()
total = ppp.sum()

st.header("📊 PPP Breakdown")

for k,v in ppp.items():
    st.write(f"{k}: {v} ({round(v/total*100)}%)")

dominant = ppp.idxmax()

# ─────────────────────────────
# FORECAST + RISK
# ─────────────────────────────
weekly = df.groupby(['Agent_Name','Week']).agg(DSAT=('DSAT','sum')).reset_index()

weekly['lag1'] = weekly.groupby('Agent_Name')['DSAT'].shift(1)
weekly = weekly.dropna()

rf = RandomForestRegressor()
rf.fit(weekly[['lag1']], weekly['DSAT'])

agent_summary = []

for agent, grp in weekly.groupby('Agent_Name'):
    grp = grp.sort_values('Week')
    if len(grp) < 2: continue

    last = grp.iloc[-1]
    prev = grp.iloc[-2]

    p1 = rf.predict([[last['lag1']]])[0]
    p2 = rf.predict([[prev['lag1']]])[0]

    risk = (p1 - weekly['DSAT'].mean())/weekly['DSAT'].std()*10 + 50
    delta = risk - ((p2 - weekly['DSAT'].mean())/weekly['DSAT'].std()*10 + 50)

    agent_summary.append([agent, int(p1), int(risk), round(delta,1)])

agent_df = pd.DataFrame(agent_summary, columns=["Agent","Pred DSAT","Risk","Δ"])

st.header("🚨 Agent Risk")
st.dataframe(agent_df)

# ─────────────────────────────
# 5 WHY ENGINE
# ─────────────────────────────
st.header("🔎 5 WHY")

agent = st.selectbox("Agent", df['Agent_Name'].unique())

a_df = df[df['Agent_Name']==agent]
prod = a_df[a_df['DSAT']==1]['Product'].value_counts().idxmax()

st.write(f"""
WHY 1: DSAT increasing  
WHY 2: Product: {prod}  
WHY 3: Issue: {dominant}  
WHY 4: Root cause: {'Product limitation' if dominant=='Product' else 'Agent/process gap'}  
WHY 5: Action: {'Escalate product fix' if dominant=='Product' else 'Coaching'}  
""")

# ─────────────────────────────
# TICKET INTELLIGENCE
# ─────────────────────────────
st.header("🎫 Ticket Deep Dive")

t_id = st.selectbox("Ticket", df['Ticket_ID'])

t = df[df['Ticket_ID']==t_id].iloc[0]

txt = t['Combined_Text'].lower()

signals = []
if "escalate" in txt: signals.append("Escalation")
if "again" in txt: signals.append("Repeat effort")
if "not working" in txt: signals.append("Product failure")

st.write("Signals:", signals)

# ─────────────────────────────
# RETURN MODEL
# ─────────────────────────────
df['issue_encoded'] = df['Issue_Label'].map({"People":0,"Process":1,"Product":2}).fillna(3)

rf_ret = RandomForestClassifier()
rf_ret.fit(df[['issue_encoded']], df['DSAT'])

prob = rf_ret.predict_proba([[t['issue_encoded']]])[0][1]

st.write("Return DSAT Probability:", round(prob,2))

# ─────────────────────────────
# GEMINI (OPTIONAL)
# ─────────────────────────────
if gemini:
    if st.button("AI Insight"):
        res = gemini.generate_content(f"Analyze DSAT for {agent} on {prod}")
        st.write(res.text)
