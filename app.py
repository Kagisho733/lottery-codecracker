# =====================================================
# LOTTERY AI PRO FINAL v23.0
# =====================================================
# PURPOSE:
# - Firebase-powered lottery analytics dashboard
# - Smart lottery prediction engine
# - Finance tracking system
# - Live commentary updates
# - Monte Carlo weighted ticket generation
# - Pair relationship analytics
# - Trend detection system
# - Heatmaps + advanced graph analytics
# - Optimized Firebase quota usage
# - 24H commentary auto cleanup
#
# MAIN FEATURES:
# 1. Dashboard
# 2. Add Draw
# 3. History Manager
# 4. Finance Tracker
# 5. Reset System
# 6. Smart Ticket Sections
# 7. Best 4–8 Optimizer
# 8. Pair Relationship Engine
# 9. Trend Momentum Analytics
# 10. Monte Carlo Probability Engine
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
from google.api_core.exceptions import ResourceExhausted

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Lottery AI PRO FINAL",
    layout="wide"
)

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

        config["private_key"] = (
            config["private_key"]
            .replace("\\n", "\n")
            .strip()
        )

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
    "pairs": "pairs",
    "trend": "trend",
}

# =====================================================
# FIREBASE HELPERS
# =====================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_collection_docs(name, limit=150):

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
        st.warning("⚠️ Firebase daily quota exceeded.")
        return []

    except Exception as e:
        st.error(f"Firebase read error: {e}")
        return []

def add_doc(name, data):

    try:
        db.collection(COLLECTIONS[name]).add(data)
        st.cache_data.clear()

    except Exception as e:
        st.error(f"Add document error: {e}")

def delete_doc(name, doc_id):

    try:
        db.collection(COLLECTIONS[name]).document(doc_id).delete()
        st.cache_data.clear()

    except Exception as e:
        st.error(f"Delete error: {e}")

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
        st.error(f"Reset error: {e}")

# =====================================================
# OPTIMIZED PAIRS ENGINE
# =====================================================

def upsert_pair(pair_key):

    try:

        ref = db.collection("pairs").document(pair_key)
        snap = ref.get()

        if snap.exists:

            current = snap.to_dict().get("count", 0)

            ref.update({
                "pair": pair_key,
                "count": current + 1,
                "updated": datetime.now().isoformat()
            })

        else:

            ref.set({
                "pair": pair_key,
                "count": 1,
                "updated": datetime.now().isoformat()
            })

        st.cache_data.clear()

    except Exception as e:
        st.error(f"Pair update error: {e}")

# =====================================================
# ANALYTICS ENGINE
# =====================================================

@st.cache_data(show_spinner=False)
def build_model(draws_data):

    draws = [
        x["numbers"]
        for x in draws_data
        if "numbers" in x
    ]

    if not draws:
        return [], Counter(), {}, {}, {}

    # frequency engine
    freq = Counter(
        n
        for row in draws
        for n in row
    )

    # recency engine
    rec = {n: 0 for n in NUMBERS}

    for i, row in enumerate(reversed(draws[-100:])):

        weight = 0.9 ** i

        for n in row:
            rec[n] += weight

    total = max(len(draws) * 12, 1)

    freq_p = {
        n: freq[n] / total
        for n in NUMBERS
    }

    rec_sum = sum(rec.values()) or 1

    rec_p = {
        n: rec[n] / rec_sum
        for n in NUMBERS
    }

    return draws, freq, freq_p, rec, rec_p

# =====================================================
# BEST PICK OPTIMIZER
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
# LIVE COMMENTARY GENERATOR
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

    overlap = list(
        set(hot).intersection(set(rising))
    )[:6]

    if hot:
        msgs.append(f"🔥 Hot numbers picking up: {hot}")

    if rising:
        msgs.append(f"📈 Rising trend numbers: {rising}")

    if overlap:
        msgs.append(f"🚀 Strong signals forming: {overlap}")

    return msgs

# =====================================================
# 24H COMMENTARY CLEANUP
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

def save_draw_to_firebase(nums, comment):

    now = datetime.now().isoformat()

    cleanup_old_commentary()

    # save draw
    add_doc("draws", {
        "numbers": nums,
        "comment": comment,
        "date": now
    })

    # save trend
    add_doc("trend", {
        "numbers": nums,
        "date": now
    })

    # optimized pair aggregation
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):

            pair = (
                f"{min(nums[i], nums[j])}-"
                f"{max(nums[i], nums[j])}"
            )

            upsert_pair(pair)

    # analytics
    draws_data = get_collection_docs("draws", 150)

    draws, freq, freq_p, rec, rec_p = build_model(draws_data)

    updates = generate_updates(freq, rec)

    # commentary save
    add_doc("commentary", {
        "date": now,
        "messages": (
            [f"✅ New draw inserted: {nums}"]
            + updates
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

    for row in pairs_docs[:30]:

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
            marker=dict(size=22)
        )
    )

    fig.update_layout(
        height=550,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig

# =====================================================
# PREMIUM CSS
# =====================================================

st.markdown(f"""
<style>

.stApp {{
    background:
        linear-gradient(rgba(5,10,30,.82),
        rgba(5,10,30,.92)),
        url("data:image/webp;base64,{dashboard_bg}");

    background-size: cover;
    background-attachment: fixed;
    color: white;
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
    box-shadow: 0 0 30px rgba(251,191,36,.35);
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
# SIDEBAR
# =====================================================

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
# ADD DRAW
# =====================================================

if page == "Add Draw":

    st.subheader("➕ Add Draw")

    with st.form("draw_form"):

        inp = st.text_input(
            "Enter 12 unique numbers comma separated"
        )

        comment = st.text_input("Commentary")

        submitted = st.form_submit_button("Save Draw")

    if submitted:

        try:

            nums = [
                int(x.strip())
                for x in inp.split(",")
                if x.strip()
            ]

            if len(nums) == 12 and len(set(nums)) == 12:

                updates = save_draw_to_firebase(
                    nums,
                    comment
                )

                st.success("✅ Draw saved successfully")

                for msg in updates:
                    st.info(msg)

            else:
                st.error(
                    "Enter exactly 12 unique numbers."
                )

        except:
            st.error("Invalid input.")

# =====================================================
# DASHBOARD
# =====================================================

elif page == "Dashboard":

    cleanup_old_commentary()

    st.title("🎰 Lottery AI PRO Dashboard")

    draws_data = get_collection_docs("draws", 150)
    finance_data = get_collection_docs("finance", 200)
    commentary_data = get_collection_docs("commentary", 10)
    pairs_data = get_collection_docs("pairs", 100)

    st.markdown("""
    <div class='ticket-card'>
        <h3>🎉 Welcome Back</h3>
        <p>
        Firebase synced analytics dashboard with
        advanced prediction intelligence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not draws_data:
        st.warning("No draws yet")
        st.stop()

    draws, freq, freq_p, rec, rec_p = build_model(draws_data)

    # hybrid probability engine
    final_probs = {

        n: (
            ((0.6 * freq_p[n] + 0.4 * rec_p[n]) * 0.6)
            + 0.4 * np.random.rand()
        )

        for n in NUMBERS
    }

    # finance
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

    profit_color = (
        "#22c55e"
        if profit >= 0
        else "#ef4444"
    )

    roi_color = (
        "#22c55e"
        if roi >= 0
        else "#ef4444"
    )

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
            <h2 style='color:{profit_color};'>
            R {profit:,.2f}
            </h2>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='ticket-card'>
            <h4>📈 ROI</h4>
            <h2 style='color:{roi_color};'>
            {roi:.2f}%
            </h2>
        </div>
        """, unsafe_allow_html=True)

    # charts
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

    # heatmap
    heatmap = plot_heatmap(draws)

    if heatmap:
        st.plotly_chart(
            heatmap,
            use_container_width=True
        )

    # commentary
    st.subheader("📝 Live Update Commentary")

    for row in commentary_data:

        for msg in row.get("messages", []):

            st.markdown(
                f"<div class='commentary-box'>{msg}</div>",
                unsafe_allow_html=True
            )

    # optimizer
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

    # smart sections
    st.subheader("🎟️ Smart Ticket Sections")

    for sec in range(1, 5):

        st.markdown(f"### Section {sec}")

        cols = st.columns(4)

        for i in range(8):

            weights = np.array(
                list(final_probs.values())
            )

            weights /= weights.sum()

            # Monte Carlo weighted generator
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

    # recent history
    st.subheader("📚 Recent History")

    hist_df = pd.DataFrame(draws_data[-10:])

    st.dataframe(
        hist_df[
            ["numbers", "comment", "date"]
        ],
        use_container_width=True
    )

    # advanced graph
    if advanced_graphs:

        st.subheader("🕸️ Advanced Pair Graph")

        fig = plot_pair_network(pairs_data)

        if fig:
            st.plotly_chart(
                fig,
                use_container_width=True
            )

# =====================================================
# HISTORY
# =====================================================

elif page == "History":

    st.subheader("📚 History Manager")

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

        if st.button("🗑️ Delete Latest Row"):

            delete_doc(
                "draws",
                df.iloc[-1]["_id"]
            )

            st.success(
                "✅ Latest row deleted successfully"
            )

            st.rerun()

# =====================================================
# FINANCE
# =====================================================

elif page == "Finance":

    st.subheader("💵 Finance Tracker")

    with st.form("finance_form"):

        stake = st.number_input(
            "Stake",
            min_value=0.0
        )

        profit = st.number_input("Profit")

        submitted = st.form_submit_button("Save")

    if submitted:

        add_doc("finance", {
            "stake": stake,
            "profit": profit,
            "date": datetime.now().isoformat()
        })

        st.success("✅ Finance saved")

    finance_df = pd.DataFrame(
        get_collection_docs("finance", 200)
    )

    if not finance_df.empty:

        st.dataframe(
            finance_df,
            use_container_width=True
        )

    if st.button("🗑️ Reset Finance Data"):

        reset_collection("finance")

        st.success(
            "✅ Finance data has been reset"
        )

# =====================================================
# RESET
# =====================================================

elif page == "Reset":

    st.title("⚠️ Reset All Firebase Data")

    if st.button("🗑️ Reset Everything"):

        for name in COLLECTIONS:
            reset_collection(name)

        st.success("""
        ✅ Reset completed successfully.
        All collections were cleared.
        """)