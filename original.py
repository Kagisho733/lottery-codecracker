# LOTTERY AI PRO v4 - LIVE INSIGHTS + UNIFORM CARDS
import streamlit as st
import pandas as pd
import numpy as np
import os, json
from datetime import datetime
from collections import Counter
import plotly.express as px

st.set_page_config(page_title="Lottery AI PRO v4", layout="wide")

DRAW_FILE = "draws.json"
RL_FILE = "rl_model.json"
FIN_FILE = "finance.json"
NUMBERS = list(range(1, 25))

# ---------------- STORAGE ----------------
def load_json(file, default):
    if not os.path.exists(file): return default
    try: return json.load(open(file))
    except: return default

def save_json(file, data): json.dump(data, open(file, "w"), indent=2)

def load_draws(): return load_json(DRAW_FILE, [])
def load_finance(): return load_json(FIN_FILE, [])
def load_rl(): return load_json(RL_FILE, {str(n): {"a":1,"b":1} for n in NUMBERS})

def save_draw(draw):
    data = load_draws()
    data.append({"numbers": draw, "date": str(datetime.now())})
    save_json(DRAW_FILE, data)

def save_finance(entry):
    data = load_finance()
    data.append(entry)
    save_json(FIN_FILE, data)

# ---------------- RL ----------------
def update_rl(model, draw):
    for k in model:
        if int(k) in draw: model[k]["a"] += 2
        else: model[k]["b"] += 1
    return model

def rl_probs(model):
    p = {int(n): np.random.beta(v["a"], v["b"]) for n, v in model.items()}
    t = sum(p.values()) or 1
    return {n: v/t for n,v in p.items()}

# ---------------- MODEL ----------------
def build_model(draw_data):
    draws = [x["numbers"] for x in draw_data]
    freq = Counter(n for row in draws for n in row)

    rec = {n:0 for n in NUMBERS}
    for i,row in enumerate(reversed(draws)):
        w = 0.9**i
        for n in row: rec[n]+=w

    total = max(len(draws)*12,1)
    freq_p = {n:freq[n]/total for n in NUMBERS}

    rec_sum = sum(rec.values()) or 1
    rec_p = {n:rec[n]/rec_sum for n in NUMBERS}

    trans = {n:Counter() for n in NUMBERS}
    for i in range(1,len(draws)):
        for p in draws[i-1]:
            for c in draws[i]: trans[p][c]+=1

    trans_p = {}
    for n in NUMBERS:
        t = sum(trans[n].values()) or 1
        trans_p[n] = {k:v/t for k,v in trans[n].items()}

    return draws, freq, freq_p, rec, rec_p, trans_p

# ---------------- INSIGHTS ----------------
def live_updates(draws, freq, rec):
    msgs = []
    avg_freq = np.mean(list(freq.values())) if freq else 0
    avg_rec = np.mean(list(rec.values())) if rec else 0

    hot = [n for n in NUMBERS if freq[n] > avg_freq]
    rising = [n for n in NUMBERS if rec[n] > avg_rec]

    if hot: msgs.append(f"🔥 Hot numbers picking up: {hot[:5]}")
    if rising: msgs.append(f"📈 Rising trend numbers: {rising[:5]}")

    overlap = list(set(hot).intersection(set(rising)))
    if overlap: msgs.append(f"🚀 Strong signals forming: {overlap[:5]}")

    return msgs

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(2,6,23,0.88), rgba(2,6,23,0.92)), url('https://images.unsplash.com/photo-1518546305927-5a555bb7020d?q=80&w=1600&auto=format&fit=crop');
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}
.card {
    background: rgba(17,24,39,0.92);
    backdrop-filter: blur(8px);
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 16px;
    border-top: 4px solid #06b6d4;
    min-height: 280px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    gap: 10px;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(6,182,212,0.25);
}
.title {
    font-size: 18px;
    font-weight: 700;
    color: #67e8f9;
    line-height: 1.2;
}
.grid {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 8px;
}
.num {
    background: #020617;
    padding: 8px;
    text-align: center;
    border-radius: 8px;
    color: #a5f3fc;
    border: 1px solid #1e293b;
    font-size: 13px;
    font-weight: 700;
}
.num-strong {
    background: #083344;
    border: 1px solid #22d3ee;
    color: white;
}
.explain {
    font-size: 11px;
    color: #cbd5e1;
    line-height: 1.45;
    max-height: 90px;
    overflow-y: auto;
    word-wrap: break-word;
}
.copy-note {
    font-size: 10px;
    color: #94a3b8;
    margin-top: auto;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.card {
    animation: fadeUp 0.45s ease;
}
</style>
""", unsafe_allow_html=True)

# ---------------- NAV ----------------
page = st.sidebar.radio("Menu", ["Dashboard","Add Draw","History","Finance","Reset"])

# ---------------- ADD DRAW ----------------
if page=="Add Draw":
    st.subheader("➕ Add Draw")
    inp = st.text_input("Enter 12 numbers")

    if st.button("Save Draw"):
        try:
            nums = list(map(int, inp.split(",")))

            if len(nums)!=12:
                st.error("Need 12 numbers")
            elif len(set(nums))!=12:
                st.error("No duplicates")
            elif any(n<1 or n>24 for n in nums):
                st.error("Range 1-24")
            else:
                save_draw(nums)
                rl = update_rl(load_rl(), nums)
                save_json(RL_FILE, rl)
                st.success("Draw saved")

                # 🔥 LIVE UPDATE FEEDBACK
                draws,freq,freq_p,rec,rec_p,trans_p = build_model(load_draws())
                updates = live_updates(draws,freq,rec)
                for u in updates:
                    st.info(u)

        except:
            st.error("Invalid input")

# ---------------- DASHBOARD ----------------
elif page=="Dashboard":
    data = load_draws()
    if not data:
        st.warning("Add draws first")
        st.stop()

    draws,freq,freq_p,rec,rec_p,trans_p = build_model(data)
    rl = rl_probs(load_rl())

    final = {n:((0.6*freq_p[n]+0.4*rec_p[n])*0.6)+rl[n]*0.4 for n in NUMBERS}

    st.subheader("📊 Probability")
    df = pd.DataFrame({"Number": list(final.keys()), "Prob": list(final.values())})

    # animated-feel probability chart
    fig_bar = px.bar(
        df,
        x="Number",
        y="Prob",
        title="Animated Probability Strength",
        text_auto='.3f'
    )
    fig_bar.update_traces(marker_line_width=0)
    fig_bar.update_layout(
        transition={"duration": 800},
        xaxis_title="Lottery Number",
        yaxis_title="Probability Score"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # proper heatmap colors
    st.subheader("🔥 Draw Heatmap")
    heat_df = pd.DataFrame(draws)
    fig_heat = px.imshow(
        heat_df,
        aspect="auto",
        color_continuous_scale="Turbo",
        title="Historical Draw Heat Intensity"
    )
    fig_heat.update_layout(
        xaxis_title="Ball Position",
        yaxis_title="Draw Index",
        transition={"duration": 800}
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("🎯 Ticket Sections")
    last_draw = draws[-1] if draws else []

    for section in range(1,5):
        st.markdown(f"### Section {section}")
        cols = st.columns(2)

        for i,balls in enumerate(range(1,9)):
            nums = list(final.keys())
            weights = np.array(list(final.values()))
            weights /= weights.sum()
            ticket = sorted(np.random.choice(nums, balls, replace=False, p=weights))

            explanations=[]
            for n in ticket:
                if freq[n] > np.mean(list(freq.values())):
                    explanations.append(f"{n}: hot")
                if rec[n] > np.mean(list(rec.values())):
                    explanations.append(f"{n}: recent")
                if last_draw:
                    score = np.mean([trans_p.get(p,{}).get(n,0) for p in last_draw])
                    if score>0.05:
                        explanations.append(f"{n}: follows pattern")

            exp_html = "<br>".join(explanations) if explanations else "Balanced"
            strongest = max(ticket, key=lambda x: final.get(x, 0)) if ticket else None
            grid = "".join([
                f"<div class='num {'num-strong' if n == strongest else ''}'>{n}</div>" for n in ticket
            ])

            with cols[i%2]:
                st.markdown(f"""
                <div class='card'>
                    <div class='title'>{balls} Ball</div>
                    <div class='grid'>{grid}</div>
                    <div class='explain'>{exp_html}</div>
                    <div class='copy-note'>⭐ Highlighted = strongest number in this ticket</div>
                </div>
                """, unsafe_allow_html=True)

# ---------------- HISTORY ----------------
elif page=="History":
    df = pd.DataFrame(load_draws())
    if not df.empty:
        edited = st.data_editor(df, use_container_width=True)
        if st.button("Save Changes"):
            save_json(DRAW_FILE, edited.to_dict("records"))
            st.success("Updated")

# ---------------- FINANCE ----------------
elif page=="Finance":
    st.subheader("💰 Finance")
    stake = st.number_input("Stake",0.0)
    payout = st.number_input("Payout",0.0)

    if st.button("Add Entry"):
        profit = payout-stake
        save_finance({"stake":stake,"payout":payout,"profit":profit})

    df = pd.DataFrame(load_finance())
    if not df.empty:
        edited = st.data_editor(df, use_container_width=True)

        total_spent = edited["stake"].sum()
        total_profit = edited["profit"].sum()

        roi = (total_profit/total_spent*100) if total_spent>0 else 0
        win_rate = len(edited[edited["profit"]>0])/len(edited)*100

        st.metric("ROI %", round(roi,2))
        st.metric("Win Rate %", round(win_rate,2))

        if st.button("Save Finance"):
            save_json(FIN_FILE, edited.to_dict("records"))

        if st.button("Delete Finance"):
            save_json(FIN_FILE, [])

# ---------------- RESET ----------------
elif page=="Reset":
    if st.button("Reset All"):
        for f in [DRAW_FILE,RL_FILE,FIN_FILE]:
            if os.path.exists(f): os.remove(f)
        st.warning("Reset done")
