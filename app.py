# ===============================
# LOTTERY AI SUPER SYSTEM (PRO VERSION)
# ===============================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import json, os, random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

st.set_page_config(page_title="Lottery AI Super System PRO", layout="wide")

DRAW_FILE = "draws.json"
RESULT_FILE = "results.json"

# -------------------------------
# STORAGE
# -------------------------------

def save_draw(draw):
    data = []
    if os.path.exists(DRAW_FILE):
        with open(DRAW_FILE, "r") as f:
            data = json.load(f)

    data.append({"numbers": draw, "date": str(datetime.now())})

    with open(DRAW_FILE, "w") as f:
        json.dump(data, f)


def load_draws():
    if not os.path.exists(DRAW_FILE):
        return []

    with open(DRAW_FILE, "r") as f:
        data = json.load(f)

    draws = []
    for d in data:
        draws.append(d.get("numbers", []))

    return draws


def reset_data():
    if os.path.exists(DRAW_FILE):
        os.remove(DRAW_FILE)
    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)

# -------------------------------
# ANALYSIS
# -------------------------------

def analyze(draws):
    numbers = list(range(1,25))
    all_nums = [n for d in draws for n in d]
    freq = Counter(all_nums)

    df = pd.DataFrame({
        "Number": numbers,
        "Frequency": [freq.get(n,0) for n in numbers]
    })

    df["Freq_norm"] = df["Frequency"] / max(df["Frequency"].max(),1)

    most = df.sort_values("Frequency", ascending=False).head(5)
    least = df.sort_values("Frequency", ascending=True).head(5)

    return df, most, least

# -------------------------------
# AI MODEL
# -------------------------------

def train_model(draws):
    if len(draws) < 5:
        return None

    nums = list(range(1,25))
    X, y = [], []

    for i in range(1,len(draws)):
        prev = draws[i-1]
        curr = draws[i]

        X.append([1 if n in prev else 0 for n in nums])
        y.append([1 if n in curr else 0 for n in nums])

    model = RandomForestClassifier()
    model.fit(X,y)
    return model


def ai_scores(draws, model):
    nums = list(range(1,25))
    last = draws[-1]
    X = [[1 if n in last else 0 for n in nums]]
    probs = model.predict_proba(X)

    scores = {}
    for i,n in enumerate(nums):
        scores[n] = probs[i][0][1] if len(probs[i][0])>1 else 0

    return scores

# -------------------------------
# PROBABILITY + RISK ENGINE
# -------------------------------

def calculate_probabilities(df):
    total = df["Hybrid"].sum()
    df["Prob"] = df["Hybrid"] / total if total > 0 else 0
    return df


def ticket_probability(ticket, df):
    probs = df.set_index("Number")["Prob"].to_dict()
    p = 1
    for n in ticket:
        p *= probs.get(n, 0.0001)
    return p


def expected_value(prob, payout):
    return prob * payout


def ticket_risk(ticket, df):
    scores = df.set_index("Number").loc[ticket]["Hybrid"]
    return np.std(scores)

# -------------------------------
# SMART NUMBER GENERATOR
# -------------------------------

def generate_ticket(df, count):
    df = df.sort_values("Hybrid", ascending=False)
    hot = df.head(12)["Number"].tolist()
    return sorted(random.sample(hot, count))

# -------------------------------
# BET CARD COLOR FUNCTION
# -------------------------------

def card_color(is_best, risk, min_risk, max_risk):
    if is_best:
        return "#4CAF50"  # Green for best EV
    elif risk == min_risk:
        return "#2196F3"  # Blue for safest
    elif risk == max_risk:
        return "#F44336"  # Red for riskiest
    else:
        return "#FFC107"  # Amber for regular

# -------------------------------
# UI
# -------------------------------

st.title("🚀 Lottery AI Super System PRO")

col1, col2 = st.columns(2)

with col1:
    st.subheader("➕ Add Draw")
    inp = st.text_input("Enter 12 numbers (comma separated)")

    if st.button("Save Draw"):
        try:
            nums = list(map(int, inp.split(",")))
            if len(nums)==12:
                save_draw(nums)
                st.success("Saved")
            else:
                st.error("Enter exactly 12 numbers")
        except:
            st.error("Invalid input")

with col2:
    st.subheader("⚙️ Controls")
    if st.button("Reset Database"):
        reset_data()
        st.warning("Database cleared")

# -------------------------------
# LOAD DATA
# -------------------------------

draws = load_draws()

if len(draws)==0:
    st.warning("No data yet")
    st.stop()

# -------------------------------
# ANALYSIS
# -------------------------------

df, most, least = analyze(draws)

# -------------------------------
# AI
# -------------------------------

model = train_model(draws)

if model:
    ai = ai_scores(draws, model)
    df["AI"] = df["Number"].map(ai)
    df["AI_norm"] = df["AI"] / max(df["AI"].max(),1)
else:
    df["AI_norm"] = 0

# -------------------------------
# HYBRID SCORE
# -------------------------------

df["Hybrid"] = (df["Freq_norm"]*0.5) + (df["AI_norm"]*0.5)
df = calculate_probabilities(df)

# -------------------------------
# BET TYPES
# -------------------------------

bet_data = [
    ("1 Ball", 1.40, 1),
    ("2 Balls", 3.00, 2),
    ("3 Balls", 8.00, 3),
    ("4 Balls", 17.00, 4),
    ("5 Balls", 41.00, 5),
    ("6 Balls", 111.00, 6),
    ("7 Balls", 351.00, 7),
    ("8 Balls", 1201.00, 8),
]

results = []

for label, payout, count in bet_data:
    ticket = generate_ticket(df, count)
    prob = ticket_probability(ticket, df)
    ev = expected_value(prob, payout)
    risk = ticket_risk(ticket, df)

    results.append({
        "Bet Type": label,
        "Numbers": ticket,
        "Payout": payout,
        "Probability": prob,
        "EV": ev,
        "Risk": risk
    })

bet_df = pd.DataFrame(results)

# Rankings
bet_df["Safety Rank"] = bet_df["Risk"].rank(method="min")
bet_df = bet_df.sort_values("Risk")
best_idx = bet_df["EV"].idxmax()
min_risk = bet_df["Risk"].min()
max_risk = bet_df["Risk"].max()

# -------------------------------
# DISPLAY BETS IN PROFESSIONAL CARDS
# -------------------------------

st.subheader("🎲 Smart Bet Recommendations (Card View)")

cols = st.columns(3)
for idx, row in bet_df.iterrows():
    is_best = row.name == best_idx
    col_idx = idx % 3
    with cols[col_idx]:
        color = card_color(is_best, row["Risk"], min_risk, max_risk)
        st.markdown(
            f"""
            <div style="
                background-color: {color};
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 15px;
                color: white;
                box-shadow: 3px 3px 10px rgba(0,0,0,0.3);
            ">
                <h3 style="text-align:center">{row['Bet Type']} {"🟢 BEST BET" if is_best else ""}</h3>
                <p><strong>🎟 Numbers:</strong> {row['Numbers']}</p>
                <p><strong>💰 Payout:</strong> {row['Payout']}x</p>
                <p><strong>📊 Probability:</strong> {row['Probability']:.8f}</p>
                <p><strong>📈 Expected Value:</strong> {row['EV']:.8f}</p>
                <p><strong>⚠️ Risk Score:</strong> {row['Risk']:.5f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------
# RANKING TABLE
# -------------------------------

st.subheader("📊 Bet Ranking (Safest → Riskiest)")
st.dataframe(bet_df.sort_values("Risk"))

# -------------------------------
# INSIGHTS
# -------------------------------

st.subheader("🔥 Most Frequent")
st.success(most["Number"].tolist())

st.subheader("❄️ Least Frequent")
st.info(least["Number"].tolist())

# -------------------------------
# VISUALS
# -------------------------------

col1, col2 = st.columns(2)

with col1:
    fig1 = plt.figure()
    plt.bar(df["Number"], df["Frequency"])
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.title("Number Frequency")
    st.pyplot(fig1)

with col2:
    fig2 = plt.figure()
    plt.bar(df["Number"], df["Hybrid"])
    plt.xlabel("Number")
    plt.ylabel("Hybrid Score")
    plt.title("Hybrid Score per Number")
    st.pyplot(fig2)

# -------------------------------
# DATA TABLE
# -------------------------------

st.subheader("📊 Full Analysis Table")
st.dataframe(df.sort_values("Hybrid", ascending=False))

# -------------------------------
# HISTORY
# -------------------------------

st.subheader("📜 Draw History")
st.write(draws)