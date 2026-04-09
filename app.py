# =====================================================
# LOTTERY AI PRO FINAL — PURPOSE & FEATURES
# =====================================================
# PURPOSE:
# - Firebase-powered lottery analytics SaaS dashboard
# - Track draw history, finance, live commentary, trend signals
# - Generate smart tickets (1–8 balls)
# - Optimize best 4–8 picks from probability model
# - Visualize frequency, probability, heatmap, pair graph
# - Manage draw history and reset collections safely
#
# MAIN FUNCTIONS:
# 1) Dashboard
#    - Welcome card
#    - Finance tracker (expense / profit / ROI)
#    - Probability strength chart
#    - Frequency chart
#    - Heatmap
#    - Live update commentary (24h auto-clear)
#    - Best 4–8 optimizer
#    - Smart ticket sections (4 sections)
#    - Recent history table
#    - Advanced pair graph
#
# 2) Add Draw
#    - Save 12 unique numbers
#    - Auto trend + pair aggregation update
#    - Auto commentary updates
#
# 3) History
#    - Full history table
#    - Delete latest row
#
# 4) Finance
#    - Save stake/profit
#    - Reset finance only
#
# 5) Reset
#    - Reset all Firebase collections
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import base64
from datetime import datetime, timedelta
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import firebase_admin
from firebase_admin import credentials, firestore

st.set_page_config(page_title="Lottery AI PRO FINAL", layout="wide")
NUMBERS = list(range(1, 25))

# =====================================================
# IMAGE LOADER
# =====================================================
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


dashboard_bg = get_base64("assets/dashboard_bg.jpg.webp")
sidebar_bg = get_base64("assets/sidebar_bg.jpg.webp")

# =====================================================
# FIREBASE INIT
# =====================================================
@st.cache_resource
def init_firebase():
    try:
        if firebase_admin._apps:
            return firestore.client()

        config = dict(st.secrets["FIREBASE"])
        config["private_key"] = config["private_key"].replace("\\n", "\n").strip()
        cred = credentials.Certificate(config)
        firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"🔥 Firebase auth failed: {e}")
        return None


db = init_firebase()

COLLECTIONS = {
    "draws": "draws",
    "finance": "finance",
    "commentary": "commentary",
    "pairs": "pairs",  # aggregated counts only
    "trend": "trend",
}

# =====================================================
# FIREBASE HELPERS
# =====================================================
@st.cache_data(ttl=30, show_spinner=False)
def get_collection_docs(name, limit=300):
    if db is None:
        return []
    docs = db.collection(COLLECTIONS[name]).limit(limit).get()
    return [{**doc.to_dict(), "_id": doc.id} for doc in docs]


def add_doc(name, data):
    db.collection(COLLECTIONS[name]).add(data)
    st.cache_data.clear()


def upsert_pair(pair_key):
    ref = db.collection("pairs").document(pair_key)
    snap = ref.get()
    if snap.exists:
        current = snap.to_dict().get("count", 0)
        ref.update({"count": current + 1, "updated": datetime.now().isoformat()})
    else:
        ref.set({"pair": pair_key, "count": 1, "updated": datetime.now().isoformat()})
    st.cache_data.clear()


def delete_doc(name, doc_id):
    db.collection(COLLECTIONS[name]).document(doc_id).delete()
    st.cache_data.clear()


def reset_collection(name):
    docs = db.collection(COLLECTIONS[name]).limit(1000).get()
    for doc in docs:
        doc.reference.delete()
    st.cache_data.clear()

# =====================================================
# ANALYTICS
# =====================================================
@st.cache_data(show_spinner=False)
def build_model(draws_data):
    draws = [x["numbers"] for x in draws_data if "numbers" in x]
    if not draws:
        return [], Counter(), {}, {}, {}

    freq = Counter(n for row in draws for n in row)

    rec = {n: 0 for n in NUMBERS}
    for i, row in enumerate(reversed(draws[-100:])):
        w = 0.9 ** i
        for n in row:
            rec[n] += w

    total = max(len(draws) * 12, 1)
    freq_p = {n: freq[n] / total for n in NUMBERS}
    rec_sum = sum(rec.values()) or 1
    rec_p = {n: rec[n] / rec_sum for n in NUMBERS}

    return draws, freq, freq_p, rec, rec_p


def optimize_best_picks(final_probs):
    ranked = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
    return {size: sorted([n for n, _ in ranked[:size]]) for size in range(4, 9)}


def generate_updates(freq, rec):
    msgs = []
    avg_freq = np.mean(list(freq.values())) if freq else 0
    avg_rec = np.mean(list(rec.values())) if rec else 0

    hot = [n for n in NUMBERS if freq[n] > avg_freq][:6]
    rising = [n for n in NUMBERS if rec[n] > avg_rec][:6]
    overlap = list(set(hot).intersection(set(rising)))[:6]

    if hot:
        msgs.append(f"🔥 Hot numbers picking up: {hot}")
    if rising:
        msgs.append(f"📈 Rising trend numbers: {rising}")
    if overlap:
        msgs.append(f"🚀 Strong signals forming: {overlap}")
    return msgs

# =====================================================
# 24H COMMENTARY AUTO CLEAR
# =====================================================
def cleanup_old_commentary():
    rows = get_collection_docs("commentary", 100)
    now = datetime.now()
    for row in rows:
        dt = datetime.fromisoformat(row["date"])
        if now - dt > timedelta(hours=24):
            delete_doc("commentary", row["_id"])

# =====================================================
# SAVE DRAW
# =====================================================
def save_draw_to_firebase(nums, comment):
    now = datetime.now().isoformat()
    cleanup_old_commentary()

    add_doc("draws", {
        "numbers": nums,
        "comment": comment,
        "date": now
    })

    add_doc("trend", {
        "numbers": nums,
        "date": now
    })

    # aggregated pair storage (saves quota)
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            pair = f"{min(nums[i], nums[j])}-{max(nums[i], nums[j])}"
            upsert_pair(pair)

    draws_data = get_collection_docs("draws", 150)
    _, freq, _, rec, _ = build_model(draws_data)
    updates = generate_updates(freq, rec)

    add_doc("commentary", {
        "date": now,
        "messages": [f"✅ New draw inserted: {nums}"] + updates
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
    fig = px.imshow(pd.DataFrame(draws[-100:]), aspect="auto", color_continuous_scale="Turbo")
    return transparent_chart(fig)


def plot_pair_network(pairs_docs):
    if not pairs_docs:
        return None

    G = nx.Graph()
    for row in pairs_docs[:30]:
        pair = row.get("pair")
        if not pair:
            continue
        a, b = map(int, pair.split("-"))
        G.add_edge(a, b, weight=row.get("count", 1))

    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", showlegend=False))

    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        text=[str(n) for n in G.nodes()],
        mode="markers+text",
        textposition="top center",
        marker=dict(size=22)
    ))

    fig.update_layout(height=550, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# =====================================================
# CSS
# =====================================================
st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(rgba(5,10,30,.82), rgba(5,10,30,.92)),
    url('data:image/webp;base64,{dashboard_bg}');
    background-size: cover;
    color: white;
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(rgba(10,20,60,.88), rgba(5,10,30,.95)),
    url('data:image/webp;base64,{sidebar_bg}');
    background-size: cover;
}}
.ticket-card {{
    background: rgba(15,23,42,.72);
    border-top: 4px solid #fbbf24;
    border-radius: 24px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 0 30px rgba(251,191,36,.35);
}}
.ball {{
    background: radial-gradient(circle at 30% 30%, #fbbf24, #ef4444);
    color: white; border-radius: 50%; min-width: 50px;
    text-align:center; padding:12px 0; font-weight:900;
}}
.number-grid {{
    display:grid; grid-template-columns:repeat(auto-fit,minmax(50px,1fr)); gap:12px;
}}
.commentary-box {{
    background: rgba(30,41,59,.75);
    border-left:4px solid #22c55e;
    border-radius:14px; padding:12px; margin-bottom:10px;
}}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
page = st.sidebar.radio("Navigation", ["Dashboard", "Add Draw", "History", "Finance", "Reset"])
advanced_graphs = st.sidebar.toggle("Advanced Graphs")

# =====================================================
# ADD DRAW
# =====================================================
if page == "Add Draw":
    st.subheader("➕ Add Draw")
    with st.form("draw_form"):
        inp = st.text_input("Enter 12 unique numbers comma separated")
        comment = st.text_input("Commentary")
        submitted = st.form_submit_button("Save Draw")

    if submitted:
        nums = [int(x.strip()) for x in inp.split(",") if x.strip()]
        if len(nums) == 12 and len(set(nums)) == 12:
            updates = save_draw_to_firebase(nums, comment)
            st.success("✅ Draw saved successfully")
            for msg in updates:
                st.info(msg)
        else:
            st.error("Enter exactly 12 unique numbers.")

# =====================================================
# DASHBOARD
# =====================================================
elif page == "Dashboard":
    cleanup_old_commentary()
    st.title("🎰 Lottery AI PRO Dashboard")

    draws_data = get_collection_docs("draws", 150)
    finance_data = get_collection_docs("finance", 200)
    commentary_data = get_collection_docs("commentary", 10)
    pairs_data = get_collection_docs("pairs", 300)

    st.markdown("""
    <div class='ticket-card'>
        <h3>🎉 Welcome Back</h3>
        <p>Everything is synced: Firebase, analytics, finance, live commentary, optimizer, smart tickets and history.</p>
    </div>
    """, unsafe_allow_html=True)

    if not draws_data:
        st.warning("No draws yet")
        st.stop()

    draws, freq, freq_p, rec, rec_p = build_model(draws_data)
    final_probs = {n: ((0.6 * freq_p[n] + 0.4 * rec_p[n]) * 0.6) + 0.4 * np.random.rand() for n in NUMBERS}

    fin_df = pd.DataFrame(finance_data)
    spent = fin_df["stake"].sum() if not fin_df.empty else 0
    profit = fin_df["profit"].sum() if not fin_df.empty else 0
    roi = (profit / spent * 100) if spent else 0

    profit_color = "#22c55e" if profit >= 0 else "#ef4444"
    roi_color = "#22c55e" if roi >= 0 else "#ef4444"

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
            <h2 style='color:{profit_color};'>R {profit:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='ticket-card'>
            <h4>📈 ROI</h4>
            <h2 style='color:{roi_color};'>{roi:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        prob_df = pd.DataFrame({"Number": list(final_probs.keys()), "Probability": list(final_probs.values())})
        st.plotly_chart(transparent_chart(px.bar(prob_df, x="Number", y="Probability")), use_container_width=True)
    with col2:
        freq_df = pd.DataFrame({"Number": list(freq.keys()), "Frequency": list(freq.values())})
        st.plotly_chart(transparent_chart(px.bar(freq_df, x="Number", y="Frequency")), use_container_width=True)

    heatmap = plot_heatmap(draws)
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("📝 Live Update Commentary")
    for row in commentary_data:
        for msg in row.get("messages", []):
            st.markdown(f"<div class='commentary-box'>{msg}</div>", unsafe_allow_html=True)

    st.subheader("🎯 Best 4–8 Picks Optimizer")
    best_sets = optimize_best_picks(final_probs)
    tabs = st.tabs(["4", "5", "6", "7", "8"])
    for i, size in enumerate(range(4, 9)):
        with tabs[i]:
            balls = "".join([f"<div class='ball'>{n}</div>" for n in best_sets[size]])
            st.markdown(f"<div class='ticket-card'><div class='number-grid'>{balls}</div></div>", unsafe_allow_html=True)

    st.subheader("🎟️ Smart Ticket Sections")
    for sec in range(1, 5):
        st.markdown(f"### Section {sec}")
        cols = st.columns(4)
        for i in range(8):
            weights = np.array(list(final_probs.values()))
            weights /= weights.sum()
            ticket = sorted(np.random.choice(NUMBERS, i + 1, replace=False, p=weights))
            balls = "".join([f"<div class='ball'>{n}</div>" for n in ticket])
            with cols[i % 4]:
                st.markdown(f"<div class='ticket-card'><b>{i+1} Balls</b><div class='number-grid'>{balls}</div></div>", unsafe_allow_html=True)

    st.subheader("📚 Recent History")
    hist_df = pd.DataFrame(draws_data[-10:])
    st.dataframe(hist_df[["numbers", "comment", "date"]], use_container_width=True)

    if advanced_graphs:
        fig = plot_pair_network(pairs_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# HISTORY
# =====================================================
elif page == "History":
    st.subheader("📚 History Manager")
    df = pd.DataFrame(get_collection_docs("draws", 300))
    st.dataframe(df[["numbers", "comment", "date", "_id"]], use_container_width=True)
    if not df.empty and st.button("🗑️ Delete Latest Row"):
        delete_doc("draws", df.iloc[-1]["_id"])
        st.success("✅ Latest row deleted successfully")
        st.rerun()

# =====================================================
# FINANCE
# =====================================================
elif page == "Finance":
    st.subheader("💵 Finance Tracker")
    with st.form("finance_form"):
        stake = st.number_input("Stake", min_value=0.0)
        profit = st.number_input("Profit")
        submitted = st.form_submit_button("Save")

    if submitted:
        add_doc("finance", {"stake": stake, "profit": profit, "date": datetime.now().isoformat()})
        st.success("✅ Finance saved")

    if st.button("🗑️ Reset Finance Data"):
        reset_collection("finance")
        st.success("✅ Finance data has been reset")

# =====================================================
# RESET
# =====================================================
elif page == "Reset":
    st.title("⚠️ Reset All Firebase Data")
    if st.button("🗑️ Reset Everything"):
        for name in COLLECTIONS:
            reset_collection(name)
        st.success("✅ Reset completed successfully. All collections were cleared.")
