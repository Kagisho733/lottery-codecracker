# ===============================
# LOTTERY AI PRO – FOUR SECTION SAAS (1 Set per Ball Count)
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import os, json
from datetime import datetime
from collections import Counter
import plotly.express as px

st.set_page_config(page_title="🎰 Lottery AI PRO - 4 Sections", layout="wide")

DRAW_FILE = "draws.json"
RL_FILE = "rl_model.json"
NUMBERS = list(range(1, 25))

# -------------------------------
# STORAGE FUNCTIONS
# -------------------------------
def save_draw(draw):
    data = []
    if os.path.exists(DRAW_FILE):
        data = json.load(open(DRAW_FILE))
    data.append({"numbers": draw, "date": str(datetime.now())})
    json.dump(data, open(DRAW_FILE, "w"))

def load_draws():
    if not os.path.exists(DRAW_FILE):
        return []
    return json.load(open(DRAW_FILE))

# -------------------------------
# REINFORCEMENT LEARNING MODEL
# -------------------------------
def init_rl():
    return {str(n): {"a":1,"b":1} for n in NUMBERS}

def load_rl():
    if not os.path.exists(RL_FILE):
        return init_rl()
    data = json.load(open(RL_FILE))
    return {str(k):v for k,v in data.items()}

def save_rl(m):
    json.dump(m, open(RL_FILE,"w"))

def update_rl(m, draw):
    s = set(draw)
    for k in m:
        n = int(k)
        if n in s:
            m[k]["a"] += 2
        else:
            m[k]["b"] += 1
    return m

def rl_probs(m):
    p = {int(n): np.random.beta(v["a"],v["b"]) for n,v in m.items()}
    t = sum(p.values())
    return {n:v/t for n,v in p.items()}

# -------------------------------
# PROBABILITY ENGINE
# -------------------------------
def build_model(draws):
    only_draws = [d["numbers"] for d in draws]

    freq = Counter(n for d in only_draws for n in d)
    total = len(only_draws)*12
    freq_p = {n:freq[n]/total for n in NUMBERS}

    rec = {n:0 for n in NUMBERS}
    for i,d in enumerate(reversed(only_draws)):
        w = 0.9**i
        for n in d: rec[n] += w
    rec_p = {n: rec[n]/sum(rec.values()) for n in NUMBERS}

    trans = {n:Counter() for n in NUMBERS}
    for i in range(1,len(only_draws)):
        for p in only_draws[i-1]:
            for c in only_draws[i]:
                trans[p][c] += 1

    trans_p = {}
    for n in NUMBERS:
        t = sum(trans[n].values()) or 1
        trans_p[n] = {k:v/t for k,v in trans[n].items()}

    return only_draws, freq, freq_p, rec, rec_p, trans_p

def final_probs(draws,freq_p,rec_p,trans_p):
    last = draws[-1] if draws else []
    scores = {}
    for n in NUMBERS:
        base = 0.4*freq_p[n] + 0.3*rec_p[n]
        t = np.mean([trans_p.get(p,{}).get(n,0) for p in last]) if last else 0
        scores[n] = 0.7*base + 0.3*t
    total = sum(scores.values())
    return {n: v/total for n,v in scores.items()}

def combine(base,rl):
    c = {n:0.6*base[n] + 0.4*rl[n] for n in NUMBERS}
    t = sum(c.values())
    return {n:v/t for n,v in c.items()}

# -------------------------------
# TICKET GENERATOR
# -------------------------------
def gen_ticket(p,c):
    nums = list(p.keys())
    w = np.array([p[n] for n in nums])
    w /= w.sum()
    return sorted(np.random.choice(nums,c,False,p=w))

# -------------------------------
# TICKET EXPLANATION
# -------------------------------
def explain_ticket(ticket, freq, rec, trans_p, rl_model, last_draw):
    explanation = []
    for n in ticket:
        reasons = []
        if freq[n] > np.mean(list(freq.values())):
            reasons.append("hot")
        if rec[n] > np.mean(list(rec.values())):
            reasons.append("recent")
        if last_draw:
            trans_score = np.mean([trans_p.get(p, {}).get(n,0) for p in last_draw])
            if trans_score > 0.05:
                reasons.append("follows pattern")
        if rl_model.get(str(n), {"a":1,"b":1})["a"] > rl_model.get(str(n), {"a":1,"b":1})["b"]:
            reasons.append("AI learned")
        if reasons:
            explanation.append(f"{n}: {', '.join(reasons)}")
    return explanation

# -------------------------------
# UI STYLE
# -------------------------------
st.markdown("""
<style>
.card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 15px;
    color: white;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.3s;
}
.card:hover {
    transform: scale(1.03);
    box-shadow: 0px 10px 25px rgba(0,255,255,0.6);
}
.numbers {
    font-size: 28px;
    font-weight: bold;
    color: #0ff;
    text-align:center;
}
.explain {
    font-size: 12px;
    color: #ccc;
    margin-top: 10px;
}
.heading {
    font-size: 20px;
    color: #0ff;
    font-weight: bold;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

def card(ticket, explanation, heading):
    nums = " - ".join(map(str, ticket))
    exp_html = "<br>".join(explanation)
    st.markdown(f"""
    <div class="card">
        <div class="heading">{heading}</div>
        <div class="numbers">{nums}</div>
        <div class="explain">{exp_html}</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# NAVIGATION
# -------------------------------
page = st.sidebar.radio("Menu", ["Dashboard","Add Draw","History","Reset"])
reload_tickets = st.sidebar.button("🔄 Generate New Ticket Sets")

# -------------------------------
# ADD DRAW
# -------------------------------
if page=="Add Draw":
    inp = st.text_input("Enter 12 numbers (comma separated)")
    if st.button("Save"):
        try:
            nums = list(map(int, inp.split(",")))
            if len(nums) == 12:
                save_draw(nums)
                rl_model = update_rl(load_rl(), nums)
                save_rl(rl_model)
                st.success("✅ Draw Saved")
            else:
                st.error("❌ Enter exactly 12 numbers")
        except:
            st.error("❌ Invalid input")

# -------------------------------
# DASHBOARD
# -------------------------------
elif page=="Dashboard":
    data = load_draws()
    if not data:
        st.warning("Add draws first")
        st.stop()

    draws, freq, freq_p, rec, rec_p, trans_p = build_model(data)
    base = final_probs(draws, freq_p, rec_p, trans_p)
    rl_model = load_rl()
    rl = rl_probs(rl_model)
    final = combine(base, rl)

    st.subheader("📊 Probability Distribution")
    df_plot = pd.DataFrame({"Number": list(final.keys()), "Probability": list(final.values())})
    st.plotly_chart(px.bar(df_plot, x="Number", y="Probability", title="Number Probabilities"))

    # -------------------------------
    # FOUR SECTIONS
    # -------------------------------
    for section in range(1,5):
        st.markdown(f"### 🔹 Section {section} - Suggested Tickets")
        for balls in range(1,9):
            t = gen_ticket(final, balls)
            heading = f"{balls} Ball Selection"
            explanation = explain_ticket(t, freq, rec, trans_p, rl_model, draws[-1])
            card(t, explanation, heading)

# -------------------------------
# HISTORY
# -------------------------------
elif page=="History":
    data = load_draws()
    if data:
        st.subheader("📜 Draw History Table")
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No draw history")

# -------------------------------
# RESET
# -------------------------------
elif page=="Reset":
    if st.button("Reset All"):
        if os.path.exists(DRAW_FILE): os.remove(DRAW_FILE)
        if os.path.exists(RL_FILE): os.remove(RL_FILE)
        st.warning("All data reset complete")