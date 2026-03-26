import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

from textblob import TextBlob

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="DSAT Intelligence", layout="wide")
st.title("🚀 DSAT Intelligence Dashboard")

# -------------------------------
# SAFE CSV LOADER
# -------------------------------
file_path = "bpo_customer_experience_dataset.csv"

if not os.path.exists(file_path):
    st.error("❌ CSV file not found")
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
st.success(f"✅ Loaded {df.shape[0]} rows")

# -------------------------------
# CLEANING
# -------------------------------
df['Week'] = pd.to_datetime(df['Week'], errors='coerce')
df = df.dropna(subset=['Week'])

# -------------------------------
# 🎛️ FILTER (SAFE ADD)
# -------------------------------
st.sidebar.header("Filters")
selected_week = st.sidebar.selectbox(
    "Select Week",
    sorted(df['Week'].dropna().unique())
)
df = df[df['Week'] <= selected_week]

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df['DSAT'] = df['Customer_Effortless'].apply(lambda x: 1 if str(x).lower() == "no" else 0)
df['Sentiment'] = df['Customer_Comment'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# -------------------------------
# LABEL FUNCTION
# -------------------------------
def label_issue(comment):
    c = str(comment).lower()
    if any(w in c for w in ["rude","bad","angry","unhelpful"]):
        return "Communication"
    elif any(w in c for w in ["delay","wait","slow","long"]):
        return "Process"
    elif any(w in c for w in ["error","bug","failed","not working"]):
        return "Product"
    else:
        return "Other"

df['Issue_Label'] = df['Customer_Comment'].apply(label_issue)

# -------------------------------
# NLP MODEL
# -------------------------------
df_clean = df[df['Issue_Label'] != "Other"].copy()

if len(df_clean) > 50:
    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.3,
        stratify=df_clean['Issue_Label'],
        random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)

    X_train = vectorizer.fit_transform(train_df['Customer_Comment'])
    y_train = train_df['Issue_Label']

    X_test = vectorizer.transform(test_df['Customer_Comment'])
    y_test = test_df['Issue_Label']

    issue_model = LogisticRegression(max_iter=200)
    issue_model.fit(X_train, y_train)

    y_pred = issue_model.predict(X_test)
    nlp_accuracy = accuracy_score(y_test, y_pred)
else:
    nlp_accuracy = 0.0

# -------------------------------
# DSAT MODEL
# -------------------------------
weekly_df = df.groupby(['Agent_Name','Week']).agg({
    'DSAT': 'sum',
    'Ticket_ID': 'count'
}).reset_index()

weekly_df.rename(columns={'DSAT':'DSAT_Count','Ticket_ID':'Total_Tickets'}, inplace=True)

for i in range(1,5):
    weekly_df[f'DSAT_lag_{i}'] = weekly_df.groupby('Agent_Name')['DSAT_Count'].shift(i)
    weekly_df[f'Tickets_lag_{i}'] = weekly_df.groupby('Agent_Name')['Total_Tickets'].shift(i)

weekly_df = weekly_df.dropna()

features = [f'DSAT_lag_{i}' for i in range(1,5)] + [f'Tickets_lag_{i}' for i in range(1,5)]

model = RandomForestRegressor()
model.fit(weekly_df[features], weekly_df['DSAT_Count'])

# -------------------------------
# SELECT AGENT
# -------------------------------
agent = st.selectbox("Select Agent", weekly_df['Agent_Name'].unique())

agent_data = weekly_df[weekly_df['Agent_Name']==agent].sort_values('Week')
latest = agent_data.iloc[-1]

prediction = model.predict([latest[features]])[0]
risk = (prediction - weekly_df['DSAT_Count'].mean())/weekly_df['DSAT_Count'].std()*10 + 50

# -------------------------------
# METRICS
# -------------------------------
agent_actual = agent_data['DSAT_Count'].values
agent_pred = model.predict(agent_data[features])

agent_mae = mean_absolute_error(agent_actual, agent_pred)

agent_variance = np.var(agent_actual)
agent_error_var = np.var(agent_actual - agent_pred)

agent_r2 = 1 - (agent_error_var / agent_variance) if agent_variance != 0 else 0

st.subheader("📊 System Accuracy Overview")

c1,c2,c3 = st.columns(3)
c1.metric("Issue Detection Accuracy", f"{round(nlp_accuracy*100,1)}%")
c2.metric("Prediction Reliability", round(agent_r2,2))
c3.metric("Avg Prediction Error", round(agent_mae,2))

# -------------------------------
# ALERT SYSTEM
# -------------------------------
st.subheader("🚨 Alerts & Risk Monitoring")

agent_summary = weekly_df.groupby('Agent_Name')['DSAT_Count'].mean().reset_index()
agent_summary['Risk'] = (agent_summary['DSAT_Count'] - weekly_df['DSAT_Count'].mean())/weekly_df['DSAT_Count'].std()*10 + 50

st.dataframe(agent_summary.sort_values("Risk", ascending=False).head(10))

st.subheader("🟢 Low Risk Agents")
st.dataframe(agent_summary.sort_values("Risk").head(10))

# -------------------------------
# 🔥 TOP 5 AGENTS (NEW - CORRECT PLACE)
# -------------------------------
st.subheader("🔥 Top 5 Agents to Act On")

top5 = agent_summary.sort_values("Risk", ascending=False).head(5)

for _, row in top5.iterrows():
    st.markdown(f"**{row['Agent_Name']}** — Risk: **{int(row['Risk'])}**")

# -------------------------------
# KPI
# -------------------------------
st.metric("Predicted DSAT", int(prediction))
st.metric("Risk Score", int(risk))

st.line_chart(agent_data.set_index('Week')['DSAT_Count'])

# -------------------------------
# ISSUE BREAKDOWN
# -------------------------------
agent_comments = df[df['Agent_Name']==agent]

pred_issues = issue_model.predict(vectorizer.transform(agent_comments['Customer_Comment']))

issue_df = pd.DataFrame(pred_issues, columns=["Issue"])
issue_df = issue_df.value_counts().reset_index(name="Count")

st.subheader("📊 Issue Breakdown")
st.dataframe(issue_df)
st.bar_chart(issue_df.set_index("Issue")["Count"])

# -------------------------------
# AI INSIGHT (UNCHANGED LOGIC)
# -------------------------------
def generate_ai_insight(agent, pred, risk, issue_df, sentiment, trend):
    top_issue = issue_df.sort_values("Count", ascending=False).iloc[0]["Issue"]

    if trend > 0:
        trend_msg = "DSAT is rising week-over-week, indicating performance decline."
    elif trend < 0:
        trend_msg = "DSAT is improving, showing recovery."
    else:
        trend_msg = "DSAT is stable."

    return f"""
### 🤖 AI Performance Insight

Agent **{agent}**

- Predicted DSAT: **{int(pred)}**
- Risk Score: **{int(risk)}**
- {trend_msg}

Primary issue: **{top_issue}**
"""

trend = latest['DSAT_lag_1'] - latest['DSAT_lag_4']
sentiment = agent_comments['Sentiment'].mean()

st.subheader("🤖 AI Insight")
st.markdown(generate_ai_insight(agent, prediction, risk, issue_df, sentiment, trend))

# -------------------------------
# 🎯 SMART COACHING PLAN (NEW)
# -------------------------------
st.subheader("🎯 Smart Coaching Plan (Top 5)")

sorted_issues = issue_df.sort_values("Count", ascending=False).head(5)

plans = []

for _, row in sorted_issues.iterrows():
    issue = row["Issue"]

    if issue == "Communication":
        action = "Improve empathy, avoid interruptions, acknowledge concerns"
    elif issue == "Process":
        action = "Reduce wait time, avoid transfers, provide timelines"
    elif issue == "Product":
        action = "Improve product knowledge, escalate faster"
    else:
        action = "Monitor performance"

    plans.append({
        "Issue": issue,
        "Cases": int(row["Count"]),
        "Action": action
    })

st.dataframe(pd.DataFrame(plans))
