import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(page_title="Lottery AI PRO v8 SaaS", layout="wide")

# =========================
# FILES & CONSTANTS
# =========================
DRAW_FILE = "draws.json"
RL_FILE = "rl_model.json"
FIN_FILE = "finance.json"
TREND_FILE = "trend.json"
PAIR_FILE = "pairs.json"
COMMENTARY_FILE = "commentary.json"
NUMBERS = list(range(1, 25))

# =========================
# STORAGE FUNCTIONS
# =========================
def load_json(file, default):
    if not os.path.exists(file):
        return default
    try:
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_draws():
    return load_json(DRAW_FILE, [])

def load_finance():
    return load_json(FIN_FILE, [])

def load_pairs():
    return load_json(PAIR_FILE, {})

def load_trend():
    return load_json(TREND_FILE, [])

@st.cache_data(show_spinner=False)
def load_commentary():
    return load_json(COMMENTARY_FILE, [])

def load_rl():
    return load_json(RL_FILE, {str(n): {"a": 1, "b": 1} for n in NUMBERS})

def save_draw(draw, comment=""):
    data = load_draws()
    entry = {"numbers": draw, "date": str(datetime.now()), "comment": comment}
    data.append(entry)
    save_json(DRAW_FILE, data)

    trend = load_trend()
    trend.append({"date": entry["date"], "numbers": draw})
    save_json(TREND_FILE, trend)

    pairs = load_pairs()
    for i in range(len(draw)):
        for j in range(i + 1, len(draw)):
            a, b = sorted([draw[i], draw[j]])
            key = f"{a}-{b}"
            pairs[key] = pairs.get(key, 0) + 1
    save_json(PAIR_FILE, pairs)

def save_finance(entry):
    data = load_finance()
    data.append(entry)
    save_json(FIN_FILE, data)

# =========================
# RL + ANALYTICS
# =========================
def update_rl(model, draw):
    for k in model:
        if int(k) in draw:
            model[k]["a"] += 2
        else:
            model[k]["b"] += 1
    return model

def rl_probs(model):
    probs = {int(k): np.random.beta(v["a"], v["b"]) for k, v in model.items()}
    total = sum(probs.values()) or 1
    return {k: v / total for k, v in probs.items()}

@st.cache_data(show_spinner=False)
def build_model(draw_data):
    draws = [x["numbers"] for x in draw_data]
    freq = Counter(n for row in draws for n in row)

    rec = {n: 0 for n in NUMBERS}
    for i, row in enumerate(reversed(draws)):
        w = 0.9 ** i
        for n in row:
            rec[n] += w

    total = max(len(draws) * 12, 1)
    freq_p = {n: freq[n] / total for n in NUMBERS}
    rec_sum = sum(rec.values()) or 1
    rec_p = {n: rec[n] / rec_sum for n in NUMBERS}

    trans = {n: Counter() for n in NUMBERS}
    for i in range(1, len(draws)):
        for prev in draws[i - 1]:
            for curr in draws[i]:
                trans[prev][curr] += 1

    trans_p = {}
    for n in NUMBERS:
        t = sum(trans[n].values()) or 1
        trans_p[n] = {k: v / t for k, v in trans[n].items()}

    return draws, freq, freq_p, rec, rec_p, trans_p

def optimize_best_picks(final_probs, min_size=4, max_size=8):
    ranked = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
    optimized = {}
    for size in range(min_size, max_size + 1):
        optimized[size] = sorted([n for n, _ in ranked[:size]])
    return optimized

def generate_updates(freq, rec):
    msgs = []
    avg_freq = np.mean(list(freq.values())) if freq else 0
    avg_rec = np.mean(list(rec.values())) if rec else 0
    hot = [n for n in NUMBERS if freq[n] > avg_freq][:6]
    rising = [n for n in NUMBERS if rec[n] > avg_rec][:6]
    if hot:
        msgs.append(f"🔥 Hot numbers picking up: {hot}")
    if rising:
        msgs.append(f"📈 Rising trend numbers: {rising}")
    overlap = list(set(hot).intersection(set(rising)))
    if overlap:
        msgs.append(f"🚀 Strong signals forming: {overlap[:6]}")
    return msgs

@st.cache_data(show_spinner=False)
def plot_heatmap(draws):
    if not draws:
        return None
    fig = px.imshow(
        pd.DataFrame(draws),
        aspect="auto",
        color_continuous_scale="Turbo",
        template="plotly_dark",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
    )
    return fig

@st.cache_data(show_spinner=False)
def plot_pair_network(pairs, top_n=15):
    if not pairs:
        return None

    items = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    G = nx.Graph()

    for key, value in items:
        a, b = map(int, key.split("-"))
        G.add_edge(a, b, weight=value)

    pos = nx.spring_layout(G, seed=42, iterations=20)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, labels = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(str(node))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1)))
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=20),
        )
    )
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=600,
    )
    return fig

# =========================
# LOTTERY THEME CSS
# =========================
st.markdown(f"""
<style>
/* Main app background: dark space + lottery sparkles */
.stApp {{
    background:
        radial-gradient(circle at 20% 20%, rgba(255,215,0,0.2), rgba(0,0,20,0.95) 70%),
        url('https://images.unsplash.com/photo-1614761356785-c7f3fda25c62?auto=format&fit=crop&w=1600&q=80');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}}

/* Container padding */
.block-container {{
    max-width: 1600px;
    padding-top: 1rem;
}}

/* Sidebar with lottery theme */
[data-testid="stSidebar"] {{
    background:
        linear-gradient(rgba(25,25,112,0.9), rgba(0,0,50,0.95)),
        url('https://images.unsplash.com/photo-1614761356785-c7f3fda25c62?auto=format&fit=crop&w=800&q=80');
    background-size: cover;
    background-position: center;
    color: white;
}}

/* Card styling for lottery tickets */
.ticket-card {{
    background: rgba(20,30,50,0.85);
    border: 1px solid rgba(255,255,255,0.1);
    border-top: 4px solid #fbbf24;
    border-radius: 20px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.3);
    min-height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.2s ease-in-out;
}}
.ticket-card:hover {{
    transform: scale(1.02);
}}

/* Number balls grid */
.number-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(50px, 1fr));
    gap: 12px;
    margin-top: 14px;
    justify-items: center;
}}
.ball {{
   background: radial-gradient(circle at 30% 30%, #f59e0b, #ef4444);
    color: white;
    text-align: center;
    padding: 12px 0;
    border-radius: 50%;
    font-weight: 900;
    font-size: 16px;
    min-width: 50px;
    min-height: 50px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    transition: transform 0.2s ease-in-out;
}}
.ball:hover {{
    transform: scale(1.2);
    box-shadow: 0 6px 18px rgba(255,200,0,0.6);
}}

/* Commentary box */
.commentary-box {{
    background: rgba(30,41,59,0.8);
    border-left: 4px solid #22c55e;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 10px;
    animation: glow 1.5s infinite alternate;
}}

/* Glow animation for live updates */
@keyframes glow {{
    from {{ box-shadow: 0 0 5px #22c55e; }}
    to {{ box-shadow: 0 0 20px #22c55e; }}
}}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "advanced_graphs" not in st.session_state:
    st.session_state.advanced_graphs = False

page = st.sidebar.radio("Navigation", ["Dashboard", "Add Draw", "History", "Finance", "Reset"])
st.sidebar.toggle("Advanced Graphs", key="advanced_graphs")

# =========================
# ADD DRAW
# =========================
if page == "Add Draw":
    st.subheader("➕ Add Draw")
    with st.form("draw_form"):
        inp = st.text_input("Enter 12 numbers comma separated")
        comment = st.text_input("Commentary")
        submitted = st.form_submit_button("Save Draw")

    if submitted:
        try:
            nums = [int(x.strip()) for x in inp.split(",") if x.strip()]
            if len(nums) != 12 or len(set(nums)) != 12:
                st.error("Enter exactly 12 unique numbers")
            else:
                save_draw(nums, comment)
                rl = update_rl(load_rl(), nums)
                save_json(RL_FILE, rl)

                draws, freq, _, rec, _, _ = build_model(load_draws())
                updates = generate_updates(freq, rec)
                existing = load_commentary()
                existing.insert(0, {"date": str(datetime.now()), "messages": updates})
                save_json(COMMENTARY_FILE, existing[:20])
                st.cache_data.clear()

                st.success("Draw saved + analysis updated")
                for msg in updates:
                    st.info(msg)
        except Exception:
            st.error("Invalid input")

# =========================
# DASHBOARD
# =========================
elif page == "Dashboard":
    st.title("🎰 Lottery AI PRO Dashboard")
    st.markdown("""
    <div class='ticket-card'>
        <h3>🎉 Welcome back</h3>
        <p>Your smart lottery SaaS workspace is live with predictive analytics, finance, and ticket intelligence.</p>
    </div>
    """, unsafe_allow_html=True)

    data = load_draws()
    if not data:
        st.warning("Add draws first")
        st.stop()

    draws, freq, freq_p, rec, rec_p, trans_p = build_model(data)
    rl = rl_probs(load_rl())
    final = {n: ((0.6 * freq_p[n] + 0.4 * rec_p[n]) * 0.6) + rl[n] * 0.4 for n in NUMBERS}

    fin = pd.DataFrame(load_finance())
    c1, c2, c3 = st.columns(3)
    spent = fin["stake"].sum() if not fin.empty else 0
    profit = fin["profit"].sum() if not fin.empty else 0
    roi = (profit / spent * 100) if spent else 0
    c1.metric("Total Expense", f"{spent:.2f}")
    c2.metric("Profit", f"{profit:.2f}")
    c3.metric("ROI", f"{roi:.2f}%")

    row1, row2 = st.columns([2, 1])
    with row1:
        prob_df = pd.DataFrame({"Number": list(final.keys()), "Prob": list(final.values())})
        fig_prob = px.bar(prob_df, x="Number", y="Prob", title="🎯 Probability Strength", template="plotly_dark")
        fig_prob.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_prob, use_container_width=True)

    with row2:
        freq_df = pd.DataFrame({"Number": list(freq.keys()), "Frequency": list(freq.values())})
        fig_freq = px.bar(freq_df, x="Number", y="Frequency", title="📈 Frequency", template="plotly_dark")
        fig_freq.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_freq, use_container_width=True)

    heatmap = plot_heatmap(draws)
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("📝 Live Update Commentary")
    for item in load_commentary()[:5]:
        st.markdown(f"<div class='commentary-box'><b>{item['date']}</b></div>", unsafe_allow_html=True)
        for m in item["messages"]:
            st.markdown(f"<div class='commentary-box'>{m}</div>", unsafe_allow_html=True)

    st.subheader("🎯 Best 4–8 Picks Optimizer")
    optimized_sets = optimize_best_picks(final)

    tabs = st.tabs(["4 Picks", "5 Picks", "6 Picks", "7 Picks", "8 Picks"])
    for idx, size in enumerate(range(4, 9)):
        with tabs[idx]:
            ticket = optimized_sets[size]
            grid = "".join([f"<div class='ball'>{n}</div>" for n in ticket])
            st.markdown(
                f"""
                <div class='ticket-card'>
                    <h4>🎟️ Optimized {size}-Number Ticket</h4>
                    <p>Best statistically ranked combination based on frequency, recency, and RL confidence.</p>
                    <div class='number-grid'>{grid}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("🎯 Smart Ticket Sections")
    for sec in range(1, 5):
        st.markdown(f"### Section {sec}")
        cols = st.columns(4)
        for i, balls in enumerate(range(1, 9)):
            with cols[i % 4]:
                weights = np.array(list(final.values()))
                weights /= weights.sum()
                ticket = sorted(np.random.choice(NUMBERS, balls, replace=False, p=weights))
                grid = "".join([f"<div class='ball'>{n}</div>" for n in ticket])
                st.markdown(
                    f"<div class='ticket-card'><b>{balls} Ball</b><div class='number-grid'>{grid}</div></div>",
                    unsafe_allow_html=True,
                )

    st.subheader("📚 Recent History")
    hist = pd.DataFrame(data[-100:])
    hist.insert(0, "No", range(1, len(hist) + 1))
    st.dataframe(hist.tail(10), use_container_width=True)

    if st.session_state.advanced_graphs:
        st.subheader("🕸️ Advanced Pair Graph")
        fig = plot_pair_network(load_pairs())
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# =========================
# HISTORY
# =========================
elif page == "History":
    df = pd.DataFrame(load_draws())
    if not df.empty:
        df.insert(0, "No", range(1, len(df) + 1))
        st.subheader("📚 History Manager")
        edited = st.data_editor(df, use_container_width=True, num_rows="dynamic")

        if st.button("Remove Duplicate Rows"):
            cleaned = edited.drop_duplicates(subset=["numbers", "date"]).reset_index(drop=True)
            cleaned = cleaned.drop(columns=["No"], errors="ignore")
            save_json(DRAW_FILE, cleaned.to_dict("records"))
            st.success("Duplicates removed")

        if st.button("Save History"):
            cleaned = edited.drop(columns=["No"], errors="ignore")
            save_json(DRAW_FILE, cleaned.to_dict("records"))
            st.success("History saved")

# =========================
# FINANCE
# =========================
elif page == "Finance":
    st.subheader("💰 Finance Tracker")
    
    # Add new entry
    with st.form("finance_form"):
        stake = st.number_input("Stake Amount", min_value=0.0, step=1.0)
        profit = st.number_input("Profit Amount", step=1.0)
        submitted = st.form_submit_button("Add Entry")
    if submitted:
        save_finance({"stake": stake, "profit": profit, "date": str(datetime.now())})
        st.success("Finance entry added")

    # Reset Finance button
    st.markdown("---")
    st.markdown("<b>Reset Finance Data</b>", unsafe_allow_html=True)
    if st.button("🗑️ Reset Finance"):
        if os.path.exists(FIN_FILE):
            os.remove(FIN_FILE)
        st.cache_data.clear()  # Clear cached data
        st.success("✅ Finance data has been reset!")
        # Mark a session state flag to refresh the table
        st.session_state["finance_reset"] = True

    # Load finance data for display
    df = pd.DataFrame(load_finance())
    
    # If reset was just clicked, clear the table
    if "finance_reset" in st.session_state:
        df = pd.DataFrame()  # empty table
        del st.session_state["finance_reset"]
    
    # Display last 10 entries
    if not df.empty:
        st.dataframe(df.tail(10), use_container_width=True)

# =========================
# RESET
# =========================
elif page == "Reset":
    st.title("⚠️ Reset All Data")
    st.markdown("""
    <p style="color: #fbbf24;">
        This action will permanently delete all saved draws, finance data, trends, pairs, commentary, and RL models.
        <b>Use with caution!</b>
    </p>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<br>", unsafe_allow_html=True)
        reset_col1, reset_col2, reset_col3 = st.columns([1,2,1])
        with reset_col2:
            if st.button("🗑️ Reset All Data", key="reset_button", help="Click to erase everything"):
                for file in [DRAW_FILE, FIN_FILE, TREND_FILE, PAIR_FILE, COMMENTARY_FILE, RL_FILE]:
                    if os.path.exists(file):
                        os.remove(file)
                st.success("✅ All data reset! Please refresh the page.")