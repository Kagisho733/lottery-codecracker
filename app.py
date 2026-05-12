# =====================================================
# LOTTERY AI PRO FINAL v24.0
# =====================================================
# FULL ADMIN VERSION
#
# FEATURES:
# - Firebase Lottery Dashboard
# - Draw Management
# - Finance Tracking
# - Markov Chain Analytics
# - Monte Carlo Simulations
# - Smart Ticket Generator
# - Pair Relationship Engine
# - Heatmaps
# - Trend Detection
# - Advanced Graph Analytics
# - Auto Commentary System
# - Reset Tools
#
# ADMIN ONLY VERSION
# - All user authentication removed
# - All approval systems removed
# - Full unrestricted admin access
#
# =====================================================

# =====================================================
# IMPORTS
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import base64
import random

from datetime import datetime, timedelta
from collections import Counter, defaultdict

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

import firebase_admin

from firebase_admin import credentials, firestore
from google.api_core.exceptions import ResourceExhausted

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Lottery AI PRO FINAL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CONSTANTS
# =====================================================

NUMBERS = list(range(1, 25))

COLLECTIONS = {
    "draws": "draws",
    "finance": "finance",
    "commentary": "commentary",
    "pairs": "pairs",
    "trend": "trend",
}

# =====================================================
# FIREBASE INITIALIZATION
# =====================================================

@st.cache_resource
def init_firebase():

    try:

        if firebase_admin._apps:
            return firestore.client()

        config = dict(st.secrets["FIREBASE"])

        config["private_key"] = (
            config["private_key"]
            .replace("\\n", "\n")
            .strip()
        )

        cred = credentials.Certificate(config)

        firebase_admin.initialize_app(cred)

        return firestore.client()

    except Exception as e:

        st.error(f"🔥 Firebase Initialization Failed: {e}")

        return None


db = init_firebase()

# =====================================================
# IMAGE LOADER
# =====================================================

def get_base64(file_path):

    try:

        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    except:
        return ""


dashboard_bg = get_base64("assets/dashboard_bg.jpg.webp")
sidebar_bg = get_base64("assets/sidebar_bg.jpg.webp")

# =====================================================
# GLOBAL STYLING
# =====================================================

st.markdown(f"""
<style>

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

header {{
    visibility: hidden;
}}

[data-testid="stToolbar"] {{
    display:none;
}}

.stApp {{
    background:
        linear-gradient(rgba(5,10,30,.88),
        rgba(5,10,30,.95)),
        url("data:image/webp;base64,{dashboard_bg}");

    background-size: cover;
    background-attachment: fixed;
    color:white;
}}

[data-testid="stSidebar"] {{
    background:
        linear-gradient(rgba(10,20,60,.88),
        rgba(5,10,30,.95)),
        url("data:image/webp;base64,{sidebar_bg}");

    background-size: cover;
}}

.ticket-card {{

    background: rgba(15,23,42,.72);

    border-top: 4px solid #fbbf24;

    border-radius: 24px;

    padding: 20px;

    margin-bottom: 18px;

    box-shadow: 0 0 30px rgba(251,191,36,.25);
}}

.ball {{

    background:
        radial-gradient(circle at 30% 30%,
        #fbbf24,
        #ef4444);

    width: 52px;

    height: 52px;

    border-radius: 50%;

    display:flex;

    align-items:center;

    justify-content:center;

    font-weight:900;

    color:white;
}}

.number-grid {{

    display:flex;

    flex-wrap:wrap;

    gap:10px;
}}

.commentary-box {{

    background: rgba(30,41,59,.75);

    border-left:4px solid #22c55e;

    border-radius:14px;

    padding:12px;

    margin-bottom:10px;
}}

</style>
""", unsafe_allow_html=True)

# =====================================================
# FIREBASE HELPERS
# =====================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_collection_docs(name, limit=200):

    if db is None:
        return []

    try:

        docs = (
            db.collection(COLLECTIONS[name])
            .limit(limit)
            .get()
        )

        return [
            {**doc.to_dict(), "_id": doc.id}
            for doc in docs
        ]

    except ResourceExhausted:

        st.warning("⚠️ Firebase Daily Quota Exceeded")

        return []

    except Exception as e:

        st.error(f"Firebase Read Error: {e}")

        return []


def add_doc(name, data):

    try:

        db.collection(COLLECTIONS[name]).add(data)

        st.cache_data.clear()

    except Exception as e:

        st.error(f"Add Document Error: {e}")


def delete_doc(name, doc_id):

    try:

        db.collection(COLLECTIONS[name]).document(doc_id).delete()

        st.cache_data.clear()

    except Exception as e:

        st.error(f"Delete Error: {e}")


def reset_collection(name):

    try:

        docs = (
            db.collection(COLLECTIONS[name])
            .limit(500)
            .get()
        )

        for doc in docs:
            doc.reference.delete()

        st.cache_data.clear()

    except Exception as e:

        st.error(f"Reset Error: {e}")

# =====================================================
# PAIR ENGINE
# =====================================================

def upsert_pair(pair_key):

    try:

        ref = db.collection("pairs").document(pair_key)

        ref.set({
            "pair": pair_key,
            "count": firestore.Increment(1),
            "updated": datetime.now().isoformat()
        }, merge=True)

    except Exception as e:

        st.error(f"Pair Update Error: {e}")

# =====================================================
# ANALYTICS MODEL
# =====================================================

@st.cache_data(show_spinner=False)
def build_model(draws_data):

    cleaned_draws = []

    for row in draws_data:

        nums = row.get("numbers", [])

        if not isinstance(nums, list):
            continue

        try:

            nums = sorted(list(set([
                int(n)
                for n in nums
                if 1 <= int(n) <= 24
            ])))

        except:
            continue

        if len(nums) != 12:
            continue

        cleaned_draws.append(nums)

    draws = cleaned_draws

    if not draws:

        return [], Counter(), {}, {}, {}

    # =========================================
    # FREQUENCY
    # =========================================

    freq = Counter()

    for row in draws:
        for n in row:
            freq[n] += 1

    # =========================================
    # RECENCY
    # =========================================

    rec = {n: 0 for n in NUMBERS}

    recent_draws = draws[-100:]

    for i, row in enumerate(reversed(recent_draws)):

        weight = 0.9 ** i

        for n in row:
            rec[n] += weight

    # =========================================
    # PROBABILITIES
    # =========================================

    total = max(len(draws) * 12, 1)

    freq_p = {
        n: freq.get(n, 0) / total
        for n in NUMBERS
    }

    rec_sum = sum(rec.values()) or 1

    rec_p = {
        n: rec[n] / rec_sum
        for n in NUMBERS
    }

    return draws, freq, freq_p, rec, rec_p

# =====================================================
# MARKOV CHAIN ENGINE
# =====================================================

def build_markov_chain(draws):

    transitions = defaultdict(Counter)

    for i in range(len(draws) - 1):

        current_draw = draws[i]
        next_draw = draws[i + 1]

        for n in current_draw:
            for nxt in next_draw:
                transitions[n][nxt] += 1

    return transitions


def markov_prediction(transitions):

    scores = Counter()

    for current_num, next_nums in transitions.items():

        total = sum(next_nums.values())

        if total == 0:
            continue

        for nxt, count in next_nums.items():

            scores[nxt] += count / total

    return scores

# =====================================================
# MONTE CARLO ENGINE
# =====================================================

def monte_carlo_simulation(final_probs, simulations=5000):

    simulation_counts = Counter()

    numbers = list(final_probs.keys())

    weights = np.array(list(final_probs.values()))

    weights = weights / weights.sum()

    for _ in range(simulations):

        simulated_draw = np.random.choice(
            numbers,
            size=12,
            replace=False,
            p=weights
        )

        for n in simulated_draw:
            simulation_counts[n] += 1

    return simulation_counts

# =====================================================
# OPTIMIZER
# =====================================================

def optimize_best_picks(final_probs):

    ranked = sorted(
        final_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        size: sorted([n for n, _ in ranked[:size]])
        for size in range(4, 9)
    }

# =====================================================
# COMMENTARY ENGINE
# =====================================================

def generate_updates(freq, rec):

    msgs = []

    avg_freq = np.mean(list(freq.values())) if freq else 0
    avg_rec = np.mean(list(rec.values())) if rec else 0

    hot = [
        n for n in NUMBERS
        if freq[n] > avg_freq
    ][:6]

    rising = [
        n for n in NUMBERS
        if rec[n] > avg_rec
    ][:6]

    overlap = list(set(hot).intersection(set(rising)))[:6]

    if hot:
        msgs.append(f"🔥 Hot Numbers: {hot}")

    if rising:
        msgs.append(f"📈 Rising Numbers: {rising}")

    if overlap:
        msgs.append(f"🚀 Strong Momentum: {overlap}")

    return msgs

# =====================================================
# COMMENTARY CLEANUP
# =====================================================

def cleanup_old_commentary():

    rows = get_collection_docs("commentary", 100)

    now = datetime.now()

    for row in rows:

        try:

            dt = datetime.fromisoformat(row["date"])

            if now - dt > timedelta(hours=24):
                delete_doc("commentary", row["_id"])

        except:
            pass

# =====================================================
# SAVE DRAW
# =====================================================

def save_draw(nums, comment):

    now = datetime.now().isoformat()

    cleanup_old_commentary()

    # =========================================
    # SAVE DRAW
    # =========================================

    add_doc("draws", {
        "numbers": nums,
        "comment": comment,
        "date": now
    })

    # =========================================
    # SAVE TREND
    # =========================================

    add_doc("trend", {
        "numbers": nums,
        "date": now
    })

    # =========================================
    # UPDATE PAIRS
    # =========================================

    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):

            pair = f"{min(nums[i], nums[j])}-{max(nums[i], nums[j])}"

            upsert_pair(pair)

    # =========================================
    # GENERATE COMMENTARY
    # =========================================

    draws_data = get_collection_docs("draws", 150)

    draws, freq, freq_p, rec, rec_p = build_model(draws_data)

    updates = generate_updates(freq, rec)

    add_doc("commentary", {
        "date": now,
        "messages": (
            [f"✅ New Draw Added: {nums}"] + updates
        )
    })

    return updates

# =====================================================
# CHART HELPERS
# =====================================================

def transparent_chart(fig, height=420):

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def plot_heatmap(draws):

    if not draws:
        return None

    fig = px.imshow(
        pd.DataFrame(draws[-100:]),
        aspect="auto",
        color_continuous_scale="Turbo",
        template="plotly_dark"
    )

    return transparent_chart(fig)


def plot_pair_network(pairs_docs):

    if not pairs_docs:
        return None

    G = nx.Graph()

    for row in pairs_docs[:40]:

        pair = row.get("pair")

        if not pair:
            continue

        a, b = map(int, pair.split("-"))

        G.add_edge(
            a,
            b,
            weight=row.get("count", 1)
        )

    pos = nx.spring_layout(G, seed=42)

    fig = go.Figure()

    for edge in G.edges():

        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                showlegend=False
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[pos[n][0] for n in G.nodes()],
            y=[pos[n][1] for n in G.nodes()],
            text=[str(n) for n in G.nodes()],
            mode="markers+text",
            textposition="top center",
            marker=dict(size=20)
        )
    )

    fig.update_layout(
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🎰 Lottery AI PRO")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Add Draw",
        "History",
        "Finance",
        "Reset"
    ]
)

advanced_graphs = st.sidebar.toggle("Advanced Graphs")

# =====================================================
# ADD DRAW PAGE
# =====================================================

if page == "Add Draw":

    st.title("➕ Add Draw")

    with st.form("draw_form"):

        inp = st.text_input(
            "Enter 12 Unique Numbers"
        )

        comment = st.text_input("Commentary")

        submitted = st.form_submit_button("Save Draw")

    if submitted:

        try:

            nums = sorted(list(set([
                int(x.strip())
                for x in inp.split(",")
                if x.strip()
            ])))

            if len(nums) != 12:

                st.error("Enter Exactly 12 Unique Numbers")

            else:

                updates = save_draw(nums, comment)

                st.success("✅ Draw Saved")

                for msg in updates:
                    st.info(msg)

        except:

            st.error("Invalid Input")

# =====================================================
# DASHBOARD
# =====================================================

elif page == "Dashboard":

    st.title("🎰 Lottery AI PRO Dashboard")

    draws_data = get_collection_docs("draws", 100)
    finance_data = get_collection_docs("finance", 300)
    commentary_data = get_collection_docs("commentary", 10)
    pairs_data = get_collection_docs("pairs", 50)

    if not draws_data:

        st.warning("No Draws Available")

        st.stop()

    # =========================================
    # MODEL
    # =========================================

    draws, freq, freq_p, rec, rec_p = build_model(draws_data)

    base_probs = {
        n: (
            (0.6 * freq_p[n]) +
            (0.4 * rec_p[n])
        )
        for n in NUMBERS
    }

    transitions = build_markov_chain(draws)

    markov_scores = markov_prediction(transitions)

    combined_probs = {}

    for n in NUMBERS:

        combined_probs[n] = (
            (base_probs.get(n, 0) * 0.7) +
            (markov_scores.get(n, 0) * 0.3)
        )

    monte_results = monte_carlo_simulation(combined_probs)

    final_probs = {}

    for n in NUMBERS:

        final_probs[n] = (
            combined_probs.get(n, 0)
            +
            (monte_results.get(n, 0) / 100000)
        )

    # =========================================
    # FINANCE
    # =========================================

    fin_df = pd.DataFrame(finance_data)

    spent = (
        fin_df["stake"].sum()
        if not fin_df.empty else 0
    )

    profit = (
        fin_df["profit"].sum()
        if not fin_df.empty else 0
    )

    roi = (
        (profit / spent) * 100
        if spent else 0
    )

    # =========================================
    # CARDS
    # =========================================

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(f"""
        <div class='ticket-card'>
            <h4>💸 Expense</h4>
            <h2>R {spent:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class='ticket-card'>
            <h4>💰 Profit</h4>
            <h2>R {profit:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown(f"""
        <div class='ticket-card'>
            <h4>📈 ROI</h4>
            <h2>{roi:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    # =========================================
    # CHARTS
    # =========================================

    col1, col2 = st.columns(2)

    with col1:

        prob_df = pd.DataFrame({
            "Number": list(final_probs.keys()),
            "Probability": list(final_probs.values())
        })

        st.plotly_chart(
            transparent_chart(
                px.bar(
                    prob_df,
                    x="Number",
                    y="Probability",
                    template="plotly_dark"
                )
            ),
            use_container_width=True
        )

    with col2:

        freq_df = pd.DataFrame({
            "Number": list(freq.keys()),
            "Frequency": list(freq.values())
        })

        st.plotly_chart(
            transparent_chart(
                px.bar(
                    freq_df,
                    x="Number",
                    y="Frequency",
                    template="plotly_dark"
                )
            ),
            use_container_width=True
        )

    # =========================================
    # HEATMAP
    # =========================================

    heatmap = plot_heatmap(draws)

    if heatmap:

        st.plotly_chart(
            heatmap,
            use_container_width=True
        )

    # =========================================
    # COMMENTARY
    # =========================================

    st.subheader("📝 Live Commentary")

    for row in commentary_data:

        for msg in row.get("messages", []):

            st.markdown(
                f"<div class='commentary-box'>{msg}</div>",
                unsafe_allow_html=True
            )

    # =========================================
    # MARKOV DATA
    # =========================================

    with st.expander("🧠 Markov Scores"):

        st.dataframe(
            pd.DataFrame({
                "Number": list(markov_scores.keys()),
                "Score": list(markov_scores.values())
            }).sort_values(
                by="Score",
                ascending=False
            ),
            use_container_width=True
        )

    # =========================================
    # MONTE CARLO DATA
    # =========================================

    with st.expander("🎲 Monte Carlo Results"):

        st.dataframe(
            pd.DataFrame({
                "Number": list(monte_results.keys()),
                "Hits": list(monte_results.values())
            }).sort_values(
                by="Hits",
                ascending=False
            ),
            use_container_width=True
        )

    # =========================================
    # BEST PICKS
    # =========================================

    st.subheader("🎯 Best 4–8 Picks Optimizer")

    best_sets = optimize_best_picks(final_probs)

    tabs = st.tabs(["4", "5", "6", "7", "8"])

    for i, size in enumerate(range(4, 9)):

        with tabs[i]:

            balls = "".join([
                f"<div class='ball'>{n}</div>"
                for n in best_sets[size]
            ])

            st.markdown(f"""
            <div class='ticket-card'>
                <div class='number-grid'>
                    {balls}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # =========================================
    # SMART TICKET SECTIONS
    # =========================================

    st.subheader("🎟️ Smart Ticket Sections")

    for sec in range(1, 5):

        st.markdown(f"## Section {sec}")

        cols = st.columns(4)

        for i in range(8):

            weights = np.array(
                list(final_probs.values())
            )

            weights /= weights.sum()

            ticket = sorted(
                np.random.choice(
                    NUMBERS,
                    i + 1,
                    replace=False,
                    p=weights
                )
            )

            balls = "".join([
                f"<div class='ball'>{n}</div>"
                for n in ticket
            ])

            with cols[i % 4]:

                st.markdown(f"""
                <div class='ticket-card'>
                    <b>{i+1} Balls</b>
                    <div class='number-grid'>
                        {balls}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # =========================================
    # HISTORY
    # =========================================

    st.subheader("📚 Recent Draw History")

    hist_df = pd.DataFrame(draws_data[-10:])

    st.dataframe(
        hist_df[
            ["numbers", "comment", "date"]
        ],
        use_container_width=True
    )

    # =========================================
    # ADVANCED GRAPH
    # =========================================

    if advanced_graphs:

        st.subheader("🕸️ Pair Network")

        fig = plot_pair_network(pairs_data)

        if fig:

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# =====================================================
# HISTORY PAGE
# =====================================================

elif page == "History":

    st.title("📚 Draw History")

    df = pd.DataFrame(
        get_collection_docs("draws", 300)
    )

    if not df.empty:

        st.dataframe(
            df[
                [
                    "numbers",
                    "comment",
                    "date",
                    "_id"
                ]
            ],
            use_container_width=True
        )

        if st.button("🗑️ Delete Latest Draw"):

            delete_doc(
                "draws",
                df.iloc[-1]["_id"]
            )

            st.success("Latest Draw Deleted")

            st.rerun()

# =====================================================
# FINANCE PAGE
# =====================================================

elif page == "Finance":

    st.title("💵 Finance Tracker")

    with st.form("finance_form"):

        stake = st.number_input(
            "Stake",
            min_value=0.0
        )

        profit = st.number_input("Profit")

        submitted = st.form_submit_button(
            "Save Finance"
        )

    if submitted:

        add_doc("finance", {
            "stake": stake,
            "profit": profit,
            "date": datetime.now().isoformat()
        })

        st.success("Finance Saved")

    finance_df = pd.DataFrame(
        get_collection_docs("finance", 500)
    )

    if finance_df.empty:

        st.warning("No Finance Records")

    else:

        spent = finance_df["stake"].sum()

        total_profit = finance_df["profit"].sum()

        roi = (
            (total_profit / spent) * 100
            if spent else 0
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric("Total Stake", f"R {spent:,.2f}")

        with c2:

            st.metric("Total Profit", f"R {total_profit:,.2f}")

        with c3:

            st.metric("ROI", f"{roi:.2f}%")

        st.dataframe(
            finance_df[
                [
                    "stake",
                    "profit",
                    "date"
                ]
            ],
            use_container_width=True
        )

        if st.button("🗑️ Reset Finance Data"):

            reset_collection("finance")

            st.success("Finance Reset Complete")

# =====================================================
# RESET PAGE
# =====================================================

elif page == "Reset":

    st.title("⚠️ Reset System")

    if st.button("🗑️ Reset Everything"):

        for name in COLLECTIONS:
            reset_collection(name)

        st.success("""
        ✅ ALL FIREBASE COLLECTIONS CLEARED
        """)