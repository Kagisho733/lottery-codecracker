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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Lottery AI PRO FINAL", layout="centered")
NUMBERS = list(range(1, 25))

# =====================================================
# LOCAL IMAGE LOADER
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
    "pairs": "pairs",
    "rl_model": "rl_model",
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

def delete_doc(name, doc_id):
    db.collection(COLLECTIONS[name]).document(doc_id).delete()
    st.cache_data.clear()

def reset_collection(name):
    docs = db.collection(COLLECTIONS[name]).limit(500).get()
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
# SAVE DRAW
# =====================================================
def save_draw_to_firebase(nums, comment):
    now = datetime.now().isoformat()

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

    # save pairs
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            pair = f"{min(nums[i], nums[j])}-{max(nums[i], nums[j])}"
            add_doc("pairs", {
                "pair": pair,
                "date": now
            })

    # build live update messages
    draws_data = get_collection_docs("draws", 150)
    draws, freq, freq_p, rec, rec_p = build_model(draws_data)
    updates = generate_updates(freq, rec)

    messages = [f"✅ New draw inserted: {nums}"]
    messages.extend(updates)

    # commentary save
    add_doc("commentary", {
        "date": now,
        "messages": messages
    })

    return messages

# =====================================================
# PLOTS
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

    pair_counts = Counter([x["pair"] for x in pairs_docs if "pair" in x])
    G = nx.Graph()

    for pair, count in pair_counts.most_common(20):
        a, b = map(int, pair.split("-"))
        G.add_edge(a, b, weight=count)

    pos = nx.spring_layout(G, seed=42)

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
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=22)
    ))
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=550
    )
    return fig

# =====================================================
# BEAUTIFUL PREMIUM CSS
# =====================================================
st.markdown(f"""
<style>
.stApp {{
    background:
        linear-gradient(rgba(5,10,30,0.82), rgba(5,10,30,0.92)),
        url("data:image/webp;base64,{dashboard_bg}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}}

[data-testid="stSidebar"] {{
    background:
        linear-gradient(rgba(10,20,60,0.88), rgba(5,10,30,0.95)),
        url("data:image/webp;base64,{sidebar_bg}");
    background-size: cover;
}}

.ticket-card {{
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 4px solid #fbbf24;
    border-radius: 24px;
    padding: 20px;
    margin-bottom: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 30px rgba(251,191,36,0.35);
    transition: 0.3s ease;
}}
.ticket-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 0 40px rgba(251,191,36,0.6);
}}

.number-grid {{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(50px,1fr));
    gap:12px;
    margin-top:14px;
    justify-items:center;
}}

.ball {{
    background: radial-gradient(circle at 30% 30%, #fbbf24, #ef4444);
    color:white;
    border-radius:50%;
    font-weight:900;
    padding:12px 0;
    text-align:center;
    min-width:50px;
    box-shadow: 0 0 16px rgba(251,191,36,0.5);
}}

.commentary-box {{
    background: rgba(30,41,59,0.75);
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
    st.title("🎰 Lottery AI PRO Dashboard")

    st.markdown("""
    <div class='ticket-card'>
        <h3>🎉 Welcome Back</h3>
        <p>Your predictive lottery SaaS workspace is fully synced with Firebase, advanced charts, smart ticket generation, finance tracking, commentary intelligence, and real-time trend analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    draws_data = get_collection_docs("draws", 150)
    finance_data = get_collection_docs("finance", 200)
    commentary_data = get_collection_docs("commentary", 10)
    pairs_data = get_collection_docs("pairs", 300)

    if not draws_data:
        st.warning("No draws yet")
        st.stop()

    draws, freq, freq_p, rec, rec_p = build_model(draws_data)

    final_probs = {
        n: ((0.6 * freq_p[n] + 0.4 * rec_p[n]) * 0.6) + 0.4 * np.random.rand()
        for n in NUMBERS
    }

    fin_df = pd.DataFrame(finance_data)
    spent = fin_df["stake"].sum() if not fin_df.empty else 0
    profit = fin_df["profit"].sum() if not fin_df.empty else 0
    roi = (profit / spent * 100) if spent else 0

    profit_color = "green" if profit >= 0 else "red"
    roi_color = "green" if roi >= 0 else "red"

    c1, c2, c3 = st.columns(3)

    c1.metric("💸 Expense", f"R {spent:,.2f}")

    c2.markdown(
        f"""
        <div class='ticket-card'>
            <h4>💰 Profit</h4>
            <h2 style='color:{profit_color};'>R {profit:,.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    c3.markdown(
        f"""
        <div class='ticket-card'>
            <h4>📈 ROI</h4>
            <h2 style='color:{roi_color};'>{roi:.2f}%</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    prob_df = pd.DataFrame({"Number": list(final_probs.keys()), "Probability": list(final_probs.values())})
    freq_df = pd.DataFrame({"Number": list(freq.keys()), "Frequency": list(freq.values())})

    with col1:
        st.plotly_chart(
            transparent_chart(px.bar(prob_df, x="Number", y="Probability", template="plotly_dark")),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            transparent_chart(px.bar(freq_df, x="Number", y="Frequency", template="plotly_dark")),
            use_container_width=True
        )

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
                grid = "".join([f"<div class='ball'>{n}</div>" for n in best_sets[size]])
                st.markdown(f"""
                <div class='ticket-card'>
                    <h4>🎟️ Best {size} Picks</h4>
                    <p>Optimized from live probability signals.</p>
                    <div class='number-grid'>{grid}</div>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("🎟️ Smart Ticket Sections")
        for sec in range(1, 5):
            st.markdown(f"### Section {sec}")
            cols = st.columns(4)
            for i in range(8):
                weights = np.array(list(final_probs.values()))
                weights /= weights.sum()
                ticket = sorted(np.random.choice(NUMBERS, i+1, replace=False, p=weights))
                grid = "".join([f"<div class='ball'>{n}</div>" for n in ticket])

                with cols[i % 4]:
                    st.markdown(f"""
                    <div class='ticket-card'>
                        <b>{i+1} Balls</b>
                        <div class='number-grid'>{grid}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.subheader("📚 Recent History")
        hist_df = pd.DataFrame(draws_data[-10:])
        show_cols = [c for c in ["numbers", "comment", "date"] if c in hist_df.columns]
        st.dataframe(hist_df[show_cols], use_container_width=True)

        if advanced_graphs:
            st.subheader("🕸️ Advanced Pair Graph")
            fig = plot_pair_network(pairs_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

# =====================================================
# HISTORY
# =====================================================
elif page == "History":

    st.subheader("📚 History Manager")
    history_data = get_collection_docs("draws", 300)
    df = pd.DataFrame(history_data)
    st.dataframe(df[["numbers", "comment", "date", "_id"]], use_container_width=True)
    
    if not df.empty:
        if st.button("🗑️ Delete Latest Row"):
            latest_id = df.iloc[-1]["_id"]
            delete_doc("draws", latest_id)
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
        add_doc("finance", {
            "stake": stake,
            "profit": profit,
            "date": datetime.now().isoformat()
        })
        st.success("✅ Finance saved")

    st.markdown("---")

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