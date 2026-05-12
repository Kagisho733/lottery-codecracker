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
import random
from collections import defaultdict
from streamlit.runtime.scriptrunner import get_script_run_ctx

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Lottery AI PRO FINAL",
    layout="wide",
    initial_sidebar_state="expanded"
)


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

# =====================================================
# REAL USER AUTH SYSTEM
# =====================================================

def get_user_email():

    try:

        user = st.experimental_user

        if user and user.email:

            return (
                user.email
                .lower()
                .strip()
            )

    except Exception:
        pass

    return None


USER_EMAIL = get_user_email()

# =====================================================
# ADMIN EMAILS
# =====================================================

ADMIN_EMAILS = [
    "kagishomandzukic@gmail.com"
]

# =====================================================
# LOAD APPROVED USERS
# =====================================================

@st.cache_data(ttl=300)
def get_approved_users():

    if db is None:
        return []

    try:

        docs = (
            db.collection("approved_users")
            .stream()
        )

        approved = []

        for doc in docs:

            data = doc.to_dict()

            email = (
                data.get("email", "")
                .lower()
                .strip()
            )

            if email:
                approved.append(email)

        return approved

    except Exception as e:

        st.error(
            f"Approved users error: {e}"
        )

        return []


APPROVED_USERS = get_approved_users()

# =====================================================
# ROLE DETECTION
# =====================================================

IS_ADMIN = (
    USER_EMAIL is not None
    and USER_EMAIL in [
        email.lower().strip()
        for email in ADMIN_EMAILS
    ]
)

IS_APPROVED_USER = (
    USER_EMAIL is not None
    and USER_EMAIL in APPROVED_USERS
)

NUMBERS = list(range(1, 25))

# =====================================================
# ACCESS CONTROL
# =====================================================

if USER_EMAIL is None:

    st.error("""
    Please login first.
    
    Open this app using your invitation link
    and sign in with Google.
    """)

    st.stop()

if not IS_ADMIN and not IS_APPROVED_USER:

    st.error(f"""
    Access denied.

    Your Gmail is not approved:
    
    {USER_EMAIL}
    """)

    st.stop()

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

@st.cache_data(ttl=300, show_spinner=False)
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

        ref.set({
            "pair": pair_key,
            "count": firestore.Increment(1),
            "updated": datetime.now().isoformat()
        }, merge=True)

    except Exception as e:
        st.error(f"Pair update error: {e}")

# =====================================================
# ANALYTICS ENGINE
# =====================================================

@st.cache_data(show_spinner=False)
def build_model(draws_data):

    try:

        cleaned_draws = []

        # =========================================
        # CLEAN & VALIDATE DRAWS
        # =========================================

        for row in draws_data:

            nums = row.get("numbers", [])

            # must be list
            if not isinstance(nums, list):
                continue

            cleaned_nums = []

            for n in nums:

                try:

                    n = int(n)

                    if 1 <= n <= 24:
                        cleaned_nums.append(n)

                except:
                    continue

            # must contain exactly 12 unique numbers
            if len(cleaned_nums) != 12:
                continue

            if len(set(cleaned_nums)) != 12:
                continue

            cleaned_draws.append(cleaned_nums)

        draws = cleaned_draws

        # =========================================
        # EMPTY SAFETY
        # =========================================

        if not draws:

            return (
                [],
                Counter(),
                {},
                {},
                {}
            )

        # =========================================
        # FREQUENCY ENGINE
        # =========================================

        freq = Counter()

        for row in draws:

            for n in row:

                freq[n] += 1

        # =========================================
        # RECENCY ENGINE
        # =========================================

        rec = {
            n: 0
            for n in NUMBERS
        }

        recent_draws = draws[-100:]

        for i, row in enumerate(reversed(recent_draws)):

            weight = 0.9 ** i

            for n in row:

                if n in rec:
                    rec[n] += weight

        # =========================================
        # PROBABILITY ENGINE
        # =========================================

        total = max(
            len(draws) * 12,
            1
        )

        freq_p = {
            n: freq.get(n, 0) / total
            for n in NUMBERS
        }

        rec_sum = sum(rec.values())

        if rec_sum == 0:
            rec_sum = 1

        rec_p = {
            n: rec[n] / rec_sum
            for n in NUMBERS
        }

        # =========================================
        # FINAL RETURN
        # =========================================

        return (
            draws,
            freq,
            freq_p,
            rec,
            rec_p
        )

    except Exception as e:

        st.error(f"build_model() error: {e}")

        return (
            [],
            Counter(),
            {},
            {},
            {}
        )
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
# MARKOV CHAIN MODEL
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
# MONTE CARLO SIMULATION
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
# DYNAMIC ADMIN CSS
# =====================================================

if IS_ADMIN:

    HIDE_STREAMLIT_STYLE = ""

else:

    HIDE_STREAMLIT_STYLE = """

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    """

st.markdown(f"""
<style>

{HIDE_STREAMLIT_STYLE}

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


# -----------------------------
# ADMIN EMAILS
# -----------------------------
ADMIN_EMAILS = [
    "kagishomandzukic@gmail.com"
]

# -----------------------------
# DETECT ADMIN
# -----------------------------
# =====================================================
# ROLE DETECTION
# =====================================================

IS_ADMIN = (
    USER_EMAIL is not None
    and USER_EMAIL.lower().strip() in [
        email.lower().strip()
        for email in ADMIN_EMAILS
    ]
)

IS_APPROVED_USER = (
    USER_EMAIL is not None
    and USER_EMAIL.lower().strip() in APPROVED_USERS
)

# =====================================================
# ADMIN SIDEBAR
# =====================================================

if IS_ADMIN:

    # SHOW NORMAL STREAMLIT MENU FOR ADMIN
    st.markdown("""
    <style>

    #MainMenu {
        visibility: visible !important;
    }

    header {
        visibility: visible !important;
    }

    [data-testid="stToolbar"] {
        display: block !important;
    }

    </style>
    """, unsafe_allow_html=True)

    st.sidebar.success(f"""
    👑 ADMIN ACCESS

    {USER_EMAIL}
    """)

    # ADMIN NAVIGATION
    # ADMIN NAVIGATION
    pages = [
    "Dashboard",
    "Add Draw",
    "History",
    "Finance",
    "Users",
    "Reset"
    ]

    page = st.sidebar.radio(
        "📂 Admin Navigation",
        pages
    )

    advanced_graphs = st.sidebar.toggle(
        "Advanced Graphs"
    )

# =====================================================
# SUBSCRIBER / INVITED USER MODE
# =====================================================

else:

    st.markdown("""
    <style>

    #MainMenu {
        visibility: hidden !important;
    }

    header {
        visibility: hidden !important;
    }

    footer {
        visibility: hidden !important;
    }

    [data-testid="stToolbar"] {
        display: none !important;
    }

    section[data-testid="stSidebar"] {
        display: none !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # USERS CAN ONLY SEE DASHBOARD
    page = "Dashboard"

    advanced_graphs = False
    
# =====================================================
# ADD DRAW
# =====================================================

if page == "Add Draw":
    
    if not IS_ADMIN:
        st.error("Unauthorized Access")
        st.stop()

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

    st.title("🎰 Lottery AI PRO Dashboard")
    
    if IS_ADMIN:

        st.success(f"""
        👑 ADMIN ACCESS ACTIVE
        
        Logged in as:
        {USER_EMAIL}
        """)

    else:

        st.info(f"""
        🔐 Subscriber Dashboard
        
        Logged in as:
        {USER_EMAIL}
        """)

    st.success("""
    ✅ Connected Successfully

    Reload numbers anytime to view latest draws.
    """)

    draws_data = get_collection_docs("draws", 80)
    
    if IS_ADMIN:
      finance_data = get_collection_docs("finance", 200)
      
    else:
        
      finance_data = []
    
    commentary_data = get_collection_docs("commentary", 10)
    pairs_data = get_collection_docs("pairs", 30)

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

    # =====================================================
    # HYBRID PROBABILITY ENGINE
    # =====================================================

    # base probability
    base_probs = {
        n: ((0.6 * freq_p[n]) + (0.4 * rec_p[n]))
        for n in NUMBERS
    }

    # markov analysis
    transitions = build_markov_chain(draws)
    markov_scores = markov_prediction(transitions)

    # combine markov + base
    combined_probs = {}

    for n in NUMBERS:
        combined_probs[n] = (
            (base_probs.get(n, 0) * 0.7)
            +
            (markov_scores.get(n, 0) * 0.3)
        )

    # monte carlo simulation
    monte_results = monte_carlo_simulation(combined_probs)

    # final probability engine
    # final probability engine
    # final probability engine
    final_probs = {}

    for n in NUMBERS:

        final_probs[n] = (
            combined_probs.get(n, 0)
            +
            (monte_results.get(n, 0) / 100000)
        )

    # =====================================================
    # FINANCE ANALYTICS
    # =====================================================

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
            
    with st.expander("🧠 Markov Chain Scores"):
        st.dataframe(
            pd.DataFrame({
                "Number": list(markov_scores.keys()),
                "Score": list(markov_scores.values())
            }).sort_values(by="Score", ascending=False),
            use_container_width=True
        )
    
    with st.expander("🎲 Monte Carlo Results"):
        st.dataframe(
            pd.DataFrame({
                "Number": list(monte_results.keys()),
                "Simulated Hits": list(monte_results.values())
            }).sort_values(by="Simulated Hits", ascending=False),
            use_container_width=True
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

    if not IS_ADMIN:
        st.error("Unauthorized Access")
        st.stop()

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

    st.subheader("💵 Personal Finance Tracker")

    # =====================================================
    # SAVE USER FINANCE DATA
    # =====================================================

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
            "email": USER_EMAIL,
            "stake": stake,
            "profit": profit,
            "date": datetime.now().isoformat()
        })

        st.success("✅ Finance saved successfully")

    # =====================================================
    # LOAD FINANCE DATA
    # =====================================================

    finance_df = pd.DataFrame(
        get_collection_docs("finance", 500)
    )

    # =====================================================
    # USER VIEW
    # =====================================================

    if not IS_ADMIN:

        finance_df = finance_df[
            finance_df["email"] == USER_EMAIL
        ]

    # =====================================================
    # ADMIN VIEW
    # =====================================================

    if IS_ADMIN:

        st.success("""
        👑 ADMIN FINANCE VIEW
        
        You can see all user finance records.
        """)

    else:

        st.info(f"""
        🔐 Personal Finance Tracker
        
        Logged in as:
        {USER_EMAIL}
        """)

    # =====================================================
    # EMPTY STATE
    # =====================================================

    if finance_df.empty:

        st.warning("""
        No finance records yet.
        
        Add your first finance entry above.
        """)

    else:

        # =====================================================
        # ANALYTICS
        # =====================================================

        spent = finance_df["stake"].sum()

        total_profit = finance_df["profit"].sum()

        roi = (
            (total_profit / spent) * 100
            if spent else 0
        )

        # =====================================================
        # CARDS
        # =====================================================

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(f"""
            <div class='ticket-card'>
                <h4>💸 Total Stake</h4>
                <h2>R {spent:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with c2:

            color = (
                "#22c55e"
                if total_profit >= 0
                else "#ef4444"
            )

            st.markdown(f"""
            <div class='ticket-card'>
                <h4>💰 Total Profit</h4>
                <h2 style='color:{color};'>
                R {total_profit:,.2f}
                </h2>
            </div>
            """, unsafe_allow_html=True)

        with c3:

            roi_color = (
                "#22c55e"
                if roi >= 0
                else "#ef4444"
            )

            st.markdown(f"""
            <div class='ticket-card'>
                <h4>📈 ROI</h4>
                <h2 style='color:{roi_color};'>
                {roi:.2f}%
                </h2>
            </div>
            """, unsafe_allow_html=True)

        # =====================================================
        # FINANCE TABLE
        # =====================================================

        st.subheader("📋 Finance Records")

        display_columns = [
            "stake",
            "profit",
            "date"
        ]

        if IS_ADMIN:
            display_columns.insert(0, "email")

        st.dataframe(
            finance_df[display_columns],
            use_container_width=True
        )

    # =====================================================
    # RESET BUTTON
    # =====================================================

    if IS_ADMIN:

        if st.button("🗑️ Reset Finance Data"):

            reset_collection("finance")

            st.success("""
            ✅ All finance data reset successfully.
            """)
# =====================================================
# USERS PAGE
# =====================================================

elif page == "Users":

    if not IS_ADMIN:
        st.stop()

    st.title("👥 Approved Users")

    # =========================================
    # ADD USER
    # =========================================

    with st.form("add_user_form"):

        new_email = st.text_input(
            "User Gmail"
        )

        submitted = st.form_submit_button(
            "Approve User"
        )

    if submitted:

        email = (
            new_email
            .lower()
            .strip()
        )

        if email:

            db.collection(
                "approved_users"
            ).add({
                "email": email,
                "approved_at":
                datetime.now().isoformat()
            })

            st.success(
                f"{email} approved successfully"
            )

            st.rerun()

    # =========================================
    # SHOW USERS
    # =========================================

    docs = db.collection(
        "approved_users"
    ).stream()

    users = []

    for doc in docs:

        data = doc.to_dict()

        users.append({
            "id": doc.id,
            "email": data.get("email"),
            "approved_at":
            data.get("approved_at")
        })

    if users:

        users_df = pd.DataFrame(users)

        st.dataframe(
            users_df,
            use_container_width=True
        )

        # =====================================
        # REMOVE USER
        # =====================================

        remove_email = st.selectbox(
            "Remove User",
            users_df["email"]
        )

        if st.button("❌ Remove Access"):

            target = users_df[
                users_df["email"]
                == remove_email
            ]

            if not target.empty:

                doc_id = target.iloc[0]["id"]

                db.collection(
                    "approved_users"
                ).document(doc_id).delete()

                st.success(
                    "User removed"
                )

                st.rerun()

    else:

        st.info(
            "No approved users yet."
        )
# =====================================================
# RESET
# =====================================================

elif page == "Reset":

    if not IS_ADMIN:
        st.error("Unauthorized Access")
        st.stop()
    
    st.title("⚠️ Reset All Firebase Data")

    if st.button("🗑️ Reset Everything"):

        for name in COLLECTIONS:
            reset_collection(name)

        st.success("""
        ✅ Reset completed successfully.
        All collections were cleared.
        """)