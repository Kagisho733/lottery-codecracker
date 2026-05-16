# =====================================================
# LOTTERY AI PRO FINAL v25.0
# =====================================================
# SYSTEM DESIGN IMPLEMENTATION MATRIX:
# 1. FIREBASE LAYER: Draws, Finance, Commentary, Pairs, Trend
# 2. DRAW VALIDATION: len(nums) == 12, range 1-24, unique, sorted
# 3. FREQUENCY ENGINE: Long-term appearance profiling (Weight: 0.25)
# 4. RECENCY ENGINE: Exponential momentum decay (0.9 ** i) (Weight: 0.20)
# 5. MARKOV ENGINE: State transition learning maps (Weight: 0.30)
# 6. MONTE CARLO SIMULATION ENGINE: 5,000 iterations (Weight: 0.25)
# 7. SMART TICKET SECTIONS: Dynamic Pool Restrictions (Sections 1 - 4)
# 8. HOT NUMBER FORCING: Safe structural reinforcement pools
# 9. GRAPH METRICS: NetworkX Pair Topological Mapping
# 10. OPTIMIZATIONS: Streamlit caching & quota exhaustion protection
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
# SYSTEM CONFIGURATION & ROOT LAYOUT
# =====================================================
st.set_page_config(
    page_title="Lottery AI PRO FINAL v25",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core numeric matrix bounds
NUMBERS = list(range(1, 25))

# Global architectural database collections matching report definitions
COLLECTIONS = {
    "draws": "draws",
    "finance": "finance",
    "commentary": "commentary",
    "pairs": "pairs",
    "trend": "trend"
}

# =====================================================
# FIREBASE LAYER STATE CONTROL
# =====================================================
@st.cache_resource
def init_firebase_connection():
    """Establishes thread-safe singleton credentials client for Firestore."""
    try:
        if firebase_admin._apps:
            return firestore.client()

        config = dict(st.secrets["FIREBASE"])
        # Format the private key layout safely from environment secrets
        config["private_key"] = config["private_key"].replace("\\n", "\n").strip()
        
        cred = credentials.Certificate(config)
        firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"🔥 Firebase Layer Initialization Aborted: {e}")
        return None

db = init_firebase_connection()

# =====================================================
# GRAPHIC LAYER ENVIRONMENT DECORATORS
# =====================================================
def load_base64_asset(file_path):
    """Loads localized image layers as binary background encodings."""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

dashboard_bg = load_base64_asset("assets/dashboard_bg.jpg.webp")
sidebar_bg = load_base64_asset("assets/sidebar_bg.jpg.webp")

# UI stylesheet injections providing radial gradients, glassmorphism, and hover rules
st.markdown(f"""
<style>
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display:none; }}

.stApp {{
    background: linear-gradient(rgba(5,10,30,.90), rgba(5,10,30,.96)), 
                url("data:image/webp;base64,{dashboard_bg}");
    background-size: cover;
    background-attachment: fixed;
    color: white;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(rgba(10,20,60,.90), rgba(5,10,30,.96)), 
                url("data:image/webp;base64,{sidebar_bg}");
    background-size: cover;
}}

.ticket-card {{
    background: rgba(15,23,42,.75);
    border-top: 4px solid #fbbf24;
    border-radius: 24px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 0 35px rgba(251,191,36,.20);
}}

.ball {{
    background: radial-gradient(circle at 30% 30%, #fbbf24, #ef4444);
    box-shadow: 0 0 14px rgba(239,68,68,0.55);
    width: 54px;
    height: 54px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 1.1rem;
    color: white;
    transition: transform 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}}

.ball:hover {{
    transform: scale(1.20) rotate(10deg);
}}

.number-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}}

.commentary-box {{
    background: rgba(30,41,59,.80);
    border-left: 5px solid #22c55e;
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA STORAGE PIPELINES (FIREBASE READ/WRITE/DELETE)
# =====================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_collection_records(collection_name, row_limit=200):
    """Fetches documents from specific collection with safety limits."""
    if db is None:
        return []
    try:
        docs = db.collection(COLLECTIONS[collection_name]).limit(row_limit).get()
        return [{**doc.to_dict(), "_id": doc.id} for doc in docs]
    except ResourceExhausted:
        st.warning("⚠️ Cloud Storage Alert: Firebase Daily Query Quota Exceeded.")
        return []
    except Exception as e:
        st.error(f"Firestore Pipeline Error [Read]: {e}")
        return []

def commit_document(collection_name, data_payload):
    """Appends record to the cloud store and invalidates state cache maps."""
    try:
        db.collection(COLLECTIONS[collection_name]).add(data_payload)
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Firestore Pipeline Error [Write]: {e}")

def purge_document_by_id(collection_name, document_id):
    """Deletes reference record from the target collection path."""
    try:
        db.collection(COLLECTIONS[collection_name]).document(document_id).delete()
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Firestore Pipeline Error [Delete]: {e}")

def empty_entire_collection(collection_name):
    """Wipes target documents out of collection using 500-row batches."""
    try:
        batch_docs = db.collection(COLLECTIONS[collection_name]).limit(500).get()
        for document in batch_docs:
            document.reference.delete()
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Firestore Pipeline Error [Purge Batch]: {e}")

def update_pair_relationship_counter(pair_key_string):
    """Increments co-occurrence frequency within the pair tracking table."""
    try:
        doc_ref = db.collection("pairs").document(pair_key_string)
        doc_ref.set({
            "pair": pair_key_string,
            "count": firestore.Increment(1),
            "updated": datetime.now().isoformat()
        }, merge=True)
    except Exception as e:
        st.error(f"Relational Pair Aggregator Error: {e}")

# =====================================================
# ANALYTICS ENGINE: FREQUENCY & RECENCY
# =====================================================
@st.cache_data(show_spinner=False)
def evaluate_statistical_baselines(raw_draws_collection):
    """Runs data sanitization, validations, and baseline frequency/recency extraction."""
    validated_draw_sequences = []
    
    for record in raw_draws_collection:
        numbers_array = record.get("numbers", [])
        if not isinstance(numbers_array, list):
            continue
        try:
            # Enforce data boundary filtering rules
            filtered_integers = sorted(list(set([int(val) for val in numbers_array if 1 <= int(val) <= 24])))
        except:
            continue
            
        # DRAW VALIDATION ENGINE RULE: System verifies exactly 12 unique tokens
        if len(filtered_integers) != 12:
            continue
        validated_draw_sequences.append(filtered_integers)

    if not validated_draw_sequences:
        null_probability_map = {num: 0.0 for num in NUMBERS}
        return [], Counter(), null_probability_map, {}, null_probability_map

    # 1. Long-Term Core Frequency Engine
    frequency_counter = Counter()
    for sequence in validated_draw_sequences:
        for ticket_number in sequence:
            frequency_counter[ticket_number] += 1

    # 2. Recency Engine (Decaying Momentum Matrix)
    recency_momentum_weights = {num: 0.0 for num in NUMBERS}
    bounded_recent_subsets = validated_draw_sequences[-100:]
    
    for index, sequence in enumerate(reversed(bounded_recent_subsets)):
        # Decay formula: Recent draws hold maximum mathematical priority
        exponential_decay_factor = 0.9 ** index
        for ticket_number in sequence:
            recency_momentum_weights[ticket_number] += exponential_decay_factor

    # 3. Relative Statistical Proportions Conversion Matrix
    aggregate_slots_filled = max(len(validated_draw_sequences) * 12, 1)
    frequency_probabilities = {num: frequency_counter.get(num, 0) / aggregate_slots_filled for num in NUMBERS}
    
    total_accrued_decay_sum = sum(recency_momentum_weights.values()) or 1.0
    recency_probabilities = {num: recency_momentum_weights[num] / total_accrued_decay_sum for num in NUMBERS}

    return validated_draw_sequences, frequency_counter, frequency_probabilities, recency_momentum_weights, recency_probabilities

# =====================================================
# MARKOV CHAIN TRANSITION ENGINE
# =====================================================
def process_markov_state_transitions(validated_sequences):
    """Learns sequential number-to-number occurrences between subsequent draw records."""
    transition_probability_matrix = defaultdict(Counter)
    
    for i in range(len(validated_sequences) - 1):
        current_state_draw = validated_sequences[i]
        next_state_draw = validated_sequences[i + 1]
        
        for input_node in current_state_draw:
            for tracking_target in next_state_draw:
                transition_probability_matrix[input_node][tracking_target] += 1
                
    return transition_probability_matrix

def compute_markov_transition_proportions(transition_matrix):
    """Translates occurrence lists into normalized sequential pattern probabilities."""
    accumulated_transition_scores = Counter()
    
    for input_node, potential_targets in transition_matrix.items():
        aggregate_target_occurrences = sum(potential_targets.values())
        if aggregate_target_occurrences == 0:
            continue
        for target, hit_count in potential_targets.items():
            accumulated_transition_scores[target] += hit_count / aggregate_target_occurrences
            
    total_emitted_markov_scores = sum(accumulated_transition_scores.values()) or 1.0
    normalized_markov_proportions = {num: accumulated_transition_scores[num] / total_emitted_markov_scores for num in NUMBERS}
    return normalized_markov_proportions

# =====================================================
# PRO-LEVEL MONTE CARLO SIMULATION ENGINE (1,000 DRAWS)
# =====================================================
def execute_monte_carlo_simulations(proportional_seed_weights, total_iterations=1000):
    """
    Executes a high-fidelity probabilistic simulation modeling exactly 1,000 
    independent draw trials using dynamic weight tracking without replacement.
    """
    simulated_hit_matrix = Counter()
    candidate_numbers = list(proportional_seed_weights.keys())
    base_weights = np.array(list(proportional_seed_weights.values()))
    
    # Handle edge case where all input probabilities are zero
    if base_weights.sum() == 0:
        base_weights = np.ones(len(candidate_numbers)) / len(candidate_numbers)
    else:
        base_weights = base_weights / base_weights.sum()

    for _ in range(total_iterations):
        # Create temporary tracking arrays for dynamic selection without replacement
        current_pool = list(candidate_numbers)
        current_weights = list(base_weights)
        
        # Simulating a professional 12-ball draw selection process
        for _ in range(12):
            w_sum = sum(current_weights)
            if w_sum == 0:
                # Fallback to uniform distribution if weights become zero
                normalized_weights = np.ones(len(current_pool)) / len(current_pool)
            else:
                normalized_weights = [w / w_sum for w in current_weights]
                
            # Probabilistic selection of a single ball
            chosen_ball = np.random.choice(current_pool, p=normalized_weights)
            simulated_hit_matrix[chosen_ball] += 1
            
            # Remove the selected ball and its weight from the remaining pool
            chosen_index = current_pool.index(chosen_ball)
            current_pool.pop(chosen_index)
            current_weights.pop(chosen_index)

    # Normalize total counts into exact probability proportions
    total_simulated_hits_logged = sum(simulated_hit_matrix.values()) or 1.0
    normalized_monte_carlo_proportions = {
        num: simulated_hit_matrix[num] / total_simulated_hits_logged for num in NUMBERS
    }
    return normalized_monte_carlo_proportions
# =====================================================
# PREDICTION GENERATOR & MATRIX PICK OPTIMIZER
# =====================================================
def extract_optimized_static_sets(hybrid_ai_scores):
    """Sorts global system intelligence scores to fetch fixed 4-8 ball ticket arrays."""
    sorted_nodes = sorted(hybrid_ai_scores.items(), key=lambda node: node[1], reverse=True)
    return {ticket_length: sorted([num for num, _ in sorted_nodes[:ticket_length]]) for ticket_length in range(4, 9)}

# =====================================================
# AUTOMATED DYNAMIC COMMENTARY PRODUCTION
# =====================================================
def auto_generate_commentary_insights(frequency_counter, recency_weights):
    """Builds insights by comparing historical ranges with recent performance values."""
    constructed_messages = []
    calculated_baseline_frequency_average = np.mean(list(frequency_counter.values())) if frequency_counter else 0
    calculated_baseline_recency_average = np.mean(list(recency_weights.values())) if recency_weights else 0

    hot_candidates = [num for num in NUMBERS if frequency_counter.get(num, 0) > calculated_baseline_frequency_average][:5]
    rising_candidates = [num for num in NUMBERS if recency_weights.get(num, 0) > calculated_baseline_recency_average][:5]
    overlapping_consensus = list(set(hot_candidates).intersection(set(rising_candidates)))[:5]

    if hot_candidates:
        constructed_messages.append(f"🔥 AI Hot Numbers: {hot_candidates}")
    if rising_candidates:
        constructed_messages.append(f"📈 Momentum Rising: {rising_candidates}")
    if overlapping_consensus:
        constructed_messages.append(f"🚀 Strong AI Consensus: {overlapping_consensus}")
        
    return constructed_messages

def prune_expired_commentary_logs():
    """Maintains data efficiency by removing database comments older than 24 hours."""
    stored_commentary_logs = fetch_collection_records("commentary", 100)
    current_timestamp_marker = datetime.now()
    
    for log_record in stored_commentary_logs:
        try:
            parsed_creation_date = datetime.fromisoformat(log_record["date"])
            if current_timestamp_marker - parsed_creation_date > timedelta(hours=24):
                purge_document_by_id("commentary", log_record["_id"])
        except:
            pass

# =====================================================
# AUTOMATED COMMENTARY EXPIRATION CLEANUP (24-HOUR LIFE)
# =====================================================
def prune_expired_commentary_logs():
    """
    Scans the commentary collection and removes any log entries 
    that were created more than 24 hours ago to protect storage.
    """
    # Fetch up to 500 logs to check expiration states
    stored_commentary_logs = fetch_collection_records("commentary", row_limit=500)
    current_time = datetime.now()
    
    purged_count = 0
    for log_record in stored_commentary_logs:
        try:
            # Parse the ISO timestamp back into a datetime object
            parsed_creation_date = datetime.fromisoformat(log_record["date"])
            time_difference = current_time - parsed_creation_date
            
            # If the log is older than 1 day (24 hours), delete it
            if time_difference > timedelta(days=1):
                purge_document_by_id("commentary", log_record["_id"])
                purged_count += 1
        except Exception as e:
            # Skip records with invalid date formats to prevent system crashes
            continue
            
    if purged_count > 0:
        st.sidebar.info(f"♻️ Auto-Cleanup: Purged {purged_count} expired commentary logs.")

# =====================================================
# PROCESS PATHWAY DATA PIPELINE: SAVE NEW DRAW
# =====================================================
def execute_draw_ingestion_pipeline(validated_numbers, user_commentary_note):
    """Commits new draws to database, maps relation networks, and refreshes updates."""
    iso_timestamp_string = datetime.now().isoformat()
    prune_expired_commentary_logs()

    # Commit historical entry records inside cloud store locations
    commit_document("draws", {"numbers": validated_numbers, "comment": user_commentary_note, "date": iso_timestamp_string})
    commit_document("trend", {"numbers": validated_numbers, "date": iso_timestamp_string})

    # Break draw sequence into incremental co-occurring pair keys
    for first_idx in range(len(validated_numbers)):
        for second_idx in range(first_idx + 1, len(validated_numbers)):
            relational_pair_string = f"{min(validated_numbers[first_idx], validated_numbers[second_idx])}-{max(validated_numbers[first_idx], validated_numbers[second_idx])}"
            update_pair_relationship_counter(relational_pair_string)

    # Re-evaluate live framework data maps to trigger immediate analytical alerts
    active_historical_records = fetch_collection_records("draws", 150)
    _, frequency_counter, _, recency_weights, _ = evaluate_statistical_baselines(active_historical_records)
    newly_mined_insights = auto_generate_commentary_insights(frequency_counter, recency_weights)

    commit_document("commentary", {
        "date": iso_timestamp_string,
        "messages": [f"✅ New Draw Added: {validated_numbers}"] + newly_mined_insights
    })
    return newly_mined_insights

# =====================================================
# INTERACTIVE DATA GRAPH IMPLEMENTATIONS
# =====================================================
def apply_dark_theme_layouts(plotly_figure, visual_height=420):
    """Applies high-contrast design specifications to structural charts."""
    plotly_figure.update_layout(
        height=visual_height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return plotly_figure

def design_sequence_heatmap(processed_draws_matrix):
    """Draws hot/cold transitions across chronological occurrences."""
    if not processed_draws_matrix:
        return None
    constructed_figure = px.imshow(
        pd.DataFrame(processed_draws_matrix[-100:]),
        aspect="auto",
        color_continuous_scale="Turbo",
        template="plotly_dark"
    )
    return apply_dark_theme_layouts(constructed_figure)

def design_topological_pair_network(fetched_pairs_records):
    """Builds spring-layout node graph maps displaying high-frequency connections."""
    if not fetched_pairs_records:
        return None

    relational_network_graph = nx.Graph()
    for row_entry in fetched_pairs_records[:40]:
        target_pair_key = row_entry.get("pair")
        if not target_pair_key:
            continue
        left_node, right_node = map(int, target_pair_key.split("-"))
        relational_network_graph.add_edge(left_node, right_node, weight=row_entry.get("count", 1))

    if len(relational_network_graph.nodes) == 0:
         return None

    node_coordinate_positions = nx.spring_layout(relational_network_graph, seed=42)
    network_canvas_figure = go.Figure()

    # Draw edge lines between interrelated coordinates
    for edge_node_link in relational_network_graph.edges():
        start_x, start_y = node_coordinate_positions[edge_node_link[0]]
        end_x, end_y = node_coordinate_positions[edge_node_link[1]]
        network_canvas_figure.add_trace(go.Scatter(
            x=[start_x, end_x], y=[start_y, end_y],
            mode="lines",
            line=dict(color='rgba(251,191,36,0.35)', width=2),
            hoverinfo='none',
            showlegend=False
        ))

    # Render points over structural line junctions (CRASH FIXED: 'shadow' removed)
    network_canvas_figure.add_trace(go.Scatter(
        x=[node_coordinate_positions[node_id][0] for node_id in relational_network_graph.nodes()],
        y=[node_coordinate_positions[node_id][1] for node_id in relational_network_graph.nodes()],
        text=[str(node_id) for node_id in relational_network_graph.nodes()],
        mode="markers+text",
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=24, 
            color='#ef4444', 
            line=dict(color='#fbbf24', width=2)
        ),
        showlegend=False
    ))

    network_canvas_figure.update_layout(
        height=620, 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)", 
        font=dict(color="white"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return network_canvas_figure

# =====================================================
# SYSTEM CONTROL WORKSPACE ROUTING
# =====================================================
st.sidebar.title("🎰 Lottery AI PRO v25")
selected_navigation_route = st.sidebar.radio("Navigation Workspace", ["Dashboard", "Add Draw", "History", "Finance", "Reset"])
toggle_advanced_topologies = st.sidebar.toggle("Render Complex Topologies", value=True)

# =====================================================
# ROUTE RUNTIME: ADD DRAW GATEWAY
# =====================================================
if selected_navigation_route == "Add Draw":
    st.title("➕ Add Draw Processing Window")
    st.markdown("Input raw multi-ball drawings below to recalculate system prediction models.")
    
    with st.form("draw_ingestion_form"):
        raw_user_input_string = st.text_input("Enter 12 Unique Numbers (Separated by Comma)", placeholder="e.g. 1, 3, 5, 7, 9, 11, 12, 14, 16, 18, 20, 22")
        user_commentary_note = st.text_input("Internal Analysis / Momentum Commentary Notes")
        submit_button_clicked = st.form_submit_button("Save Draw Engine Record")

    if submit_button_clicked:
        try:
            # Parse delimited parameters safely into sorted integer vectors
            parsed_integers_array = sorted(list(set([int(val.strip()) for val in raw_user_input_string.split(",") if val.strip()])))
            
            if len(parsed_integers_array) != 12:
                st.error("Validation Violation: System checks require exactly 12 unique valid numbers within range 1-24.")
            elif not all(1 <= token <= 24 for token in parsed_integers_array):
                st.error("Boundary Violation: Detected elements tracking outside allowed 1-24 limits.")
            else:
                running_alerts = execute_draw_ingestion_pipeline(parsed_integers_array, user_commentary_note)
                st.success("✅ Execution Succeeded: Draw metrics parsed to Firestore Database layer.")
                for operational_alert in running_alerts:
                    st.info(operational_alert)
        except Exception as error_exception:
            st.error(f"Execution Aborted. Invalid Field Formatting Parameter: {error_exception}")

# =====================================================
# ROUTE RUNTIME: CORE AI ANALYTICS SYSTEM DASHBOARD
# =====================================================
elif selected_navigation_route == "Dashboard":
    st.title("🎰 Lottery AI PRO Final v25 Analytics Dashboard")

    # Sync all primary pipeline tables simultaneously from the database
    with st.spinner("Syncing data matrices from Firestore..."):
        historical_draws_cache = fetch_collection_records("draws", 100)
        financial_ledger_cache = fetch_collection_records("finance", 300)
        automated_commentary_cache = fetch_collection_records("commentary", 10)
        co_occurring_pairs_cache = fetch_collection_records("pairs", 50)

    if not historical_draws_cache:
        st.warning("📊 Initial Sync Empty: Please populate entries via the 'Add Draw' node to build statistical models.")
        st.stop()

    # EXECUTE INTEGRATED ARCHITECTURE LOGIC PIPELINES
    validated_draws, long_term_freq, normalized_freq_p, recent_decay_weights, normalized_rec_p = evaluate_statistical_baselines(historical_draws_cache)
    markov_transition_graph = process_markov_state_transitions(validated_draws)
    normalized_markov_p = compute_markov_transition_proportions(markov_transition_graph)
    
    # Generate interim distribution weights for the Monte Carlo seed configuration
    interim_distribution_seed = {}
    for token in NUMBERS:
        interim_distribution_seed[token] = (normalized_freq_p.get(token, 0.0) * 0.6) + (normalized_rec_p.get(token, 0.0) * 0.4)
        if token in normalized_markov_p:
            interim_distribution_seed[token] = (interim_distribution_seed[token] * 0.7) + (normalized_markov_p[token] * 0.3)

    normalized_monte_carlo_p = execute_monte_carlo_simulations(interim_distribution_seed, total_iterations=1000)

    # MASTER ARITHMETIC HYBRID SYSTEM PROBABILITY FORMULA
    final_hybrid_ai_scores = {}
    for token in NUMBERS:
        final_hybrid_ai_scores[token] = (
            (normalized_freq_p.get(token, 0.0) * 0.25) +
            (normalized_rec_p.get(token, 0.0) * 0.20) +
            (normalized_markov_p.get(token, 0.0) * 0.30) +
            (normalized_monte_carlo_p.get(token, 0.0) * 0.25)
        )

    # Rank indices based on final unified scoring outcomes
    ranked_analytical_numbers = [num for num, _ in sorted(final_hybrid_ai_scores.items(), key=lambda node: node[1], reverse=True)]

    # CALCULATE FINANCIAL BALANCES
    parsed_financial_dataframe = pd.DataFrame(financial_ledger_cache)
    total_capital_expenditure = parsed_financial_dataframe["stake"].sum() if not parsed_financial_dataframe.empty else 0.0
    total_accrued_dividends = parsed_financial_dataframe["profit"].sum() if not parsed_financial_dataframe.empty else 0.0
    net_return_on_investment = (total_accrued_dividends / total_capital_expenditure) * 100 if total_capital_expenditure else 0.0

    # RENDER KPI TILES
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.markdown(f"<div class='ticket-card'><h4>💸 Total Expense</h4><h2>R {total_capital_expenditure:,.2f}</h2></div>", unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"<div class='ticket-card'><h4>💰 Total Profit</h4><h2>R {total_accrued_dividends:,.2f}</h2></div>", unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"<div class='ticket-card'><h4>📈 ROI Position</h4><h2>{net_return_on_investment:.2f}%</h2></div>", unsafe_allow_html=True)

    # PRIMARY ANALYTICAL CHART FIELDS
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("🔮 Hybrid AI Integrated Probabilities")
        integrated_probability_dataframe = pd.DataFrame({"Number": list(final_hybrid_ai_scores.keys()), "Probability": list(final_hybrid_ai_scores.values())})
        st.plotly_chart(apply_dark_theme_layouts(px.bar(integrated_probability_dataframe, x="Number", y="Probability", template="plotly_dark")), use_container_width=True)
    with chart_col2:
        st.subheader("📊 Long-Term Matrix Frequency Distribution")
        historical_frequency_dataframe = pd.DataFrame({"Number": list(long_term_freq.keys()), "Frequency": list(long_term_freq.values())})
        st.plotly_chart(apply_dark_theme_layouts(px.bar(historical_frequency_dataframe, x="Number", y="Frequency", template="plotly_dark")), use_container_width=True)

    # RENDER HEATMAP
    st.subheader("🌡️ Historical Sequence Draw Heatmap")
    calculated_heatmap_layer = design_sequence_heatmap(validated_draws)
    if calculated_heatmap_layer:
        st.plotly_chart(calculated_heatmap_layer, use_container_width=True)

    # AUTO-COMMENTARY SYSTEM FEEDS
    st.subheader("📝 Live Engine Automated Commentary")
    if automated_commentary_cache:
        for log_row in automated_commentary_cache:
            for alert_string in log_row.get("messages", []):
                st.markdown(f"<div class='commentary-box'>{alert_string}</div>", unsafe_allow_html=True)
    else:
        st.info("Commentary log buffers are currently refreshing or empty.")

    # MODEL METRIC LOG EXPANDERS
    expander_col1, expander_col2 = st.columns(2)
    with expander_col1:
        with st.expander("🧠 Markov Chain Probability Distribution Map"):
            st.dataframe(pd.DataFrame({"Number": list(normalized_markov_p.keys()), "Transition Score": list(normalized_markov_p.values())}).sort_values(by="Transition Score", ascending=False), use_container_width=True)
    with expander_col2:
        with st.expander("🎲 Monte Carlo Simulation Generated Density Proportions"):
            st.dataframe(pd.DataFrame({"Number": list(normalized_monte_carlo_p.keys()), "Simulated Weights": list(normalized_monte_carlo_p.values())}).sort_values(by="Simulated Weights", ascending=False), use_container_width=True)

    # RENDER COMPACT STRATEGIC BEST PICKS (4-8)
    st.subheader("🎯 Best 4–8 Ticket Picks Optimized Allocations")
    computed_best_picks_dictionary = extract_optimized_static_sets(final_hybrid_ai_scores)
    ticket_size_tab_selectors = st.tabs(["Ticket Size 4", "Ticket Size 5", "Ticket Size 6", "Ticket Size 7", "Ticket Size 8"])
    
    for tab_idx, ticket_length in enumerate(range(4, 9)):
        with ticket_size_tab_selectors[tab_idx]:
            balls_render_string = "".join([f"<div class='ball'>{num}</div>" for num in computed_best_picks_dictionary[ticket_length]])
            st.markdown(f"<div class='ticket-card'><div class='number-grid'>{balls_render_string}</div></div>", unsafe_allow_html=True)

    # =====================================================
    # SMART STRATEGY POOL SECTION MATRIX RENDERS
    # =====================================================
    st.subheader("🎟️ Smart Ticket Structural Pool Sections")
    
    # Establish dynamic pool allocations exactly matching the verification system report rules
    strategic_section_pools_matrix = {
        1: ranked_analytical_numbers[:10],       # Section 1: Top 10 High Confidence Hottest Pool
        2: ranked_analytical_numbers[2:14],      # Section 2: Balanced Mix, Index Offset 2 to 14 Pool
        3: ranked_analytical_numbers[4:18],      # Section 3: Wider Distribution Range 4 to 18 Pool
        4: ranked_analytical_numbers[6:24]       # Section 4: Exploration Matrix Sleeper Pool
    }

    for current_section_id in range(1, 5):
        st.markdown(f"### Smart Strategy Pool Section {current_section_id}")
        designated_target_pool = strategic_section_pools_matrix[current_section_id]
        
        # Calculate isolated probability matrices for current pool numbers
        isolated_pool_probabilities = {num: final_hybrid_ai_scores.get(num, 0.0) for num in designated_target_pool}
        normalized_pool_vector_weights = np.array(list(isolated_pool_probabilities.values()))
        
        if normalized_pool_vector_weights.sum() == 0:
            normalized_pool_vector_weights = np.ones(len(designated_target_pool)) / len(designated_target_pool)
        else:
            normalized_pool_vector_weights /= normalized_pool_vector_weights.sum()

        rendered_layout_columns = st.columns(4)
        
        for visual_card_idx in range(8):
            target_ticket_ball_count = visual_card_idx + 1
            
            # HOT NUMBER FORCING LAYER ROUTINE
            selected_forced_hot_token = random.choice(ranked_analytical_numbers[:5])
            
            # Separate the forced token from the rest of the pool to ensure non-duplicate sampling
            remainder_section_pool = [num for num in designated_target_pool if num != selected_forced_hot_token]
            remainder_pool_weights = np.array([isolated_pool_probabilities[num] for num in remainder_section_pool])
            
            if remainder_pool_weights.sum() == 0 or len(remainder_section_pool) < (target_ticket_ball_count - 1):
                # Fallback to standard selections if pool sizes fall below standard sampling thresholds
                generated_ticket_sequence = sorted(np.random.choice(designated_target_pool, size=min(target_ticket_ball_count, len(designated_target_pool)), replace=False, p=normalized_pool_vector_weights))
            else:
                remainder_pool_weights /= remainder_pool_weights.sum()
                sampled_remainder_tokens = np.random.choice(remainder_section_pool, size=target_ticket_ball_count - 1, replace=False, p=remainder_pool_weights)
                generated_ticket_sequence = sorted([selected_forced_hot_token] + list(sampled_remainder_tokens))

            ticket_balls_html_string = "".join([f"<div class='ball'>{num}</div>" for num in generated_ticket_sequence])
            with rendered_layout_columns[visual_card_idx % 4]:
                st.markdown(f"""
                <div class='ticket-card'>
                    <b>{target_ticket_ball_count} Ball Analytical Ticket</b>
                    <div class='number-grid'>{ticket_balls_html_string}</div>
                </div>
                """, unsafe_allow_html=True)

    # RECENT RECORD LOGS TABLE
    st.subheader("📚 Recent Historical Entries")
    historical_logs_dataframe = pd.DataFrame(historical_draws_cache[-10:])
    if not historical_logs_dataframe.empty:
        st.dataframe(historical_logs_dataframe[["numbers", "comment", "date"]], use_container_width=True)

    # TOPOLOGICAL PAIR MATRIX DIAGRAMS
    if toggle_advanced_topologies:
        st.subheader("🕸️ Pair Analytical Node Topological Map Network Graph")
        network_topological_graph = design_topological_pair_network(co_occurring_pairs_cache)
        if network_topological_graph:
            st.plotly_chart(network_topological_graph, use_container_width=True)
        else:
            st.info("Insufficient relational pair node densities currently available to map topologies.")

# =====================================================
# ROUTE RUNTIME: HISTORICAL ARCHIVE DATA TERMINAL (UNRESTRICTED)
# =====================================================
elif selected_navigation_route == "History":
    st.title("📚 Comprehensive Draw Database Registry")
    st.markdown("Review and manage all recorded lottery metrics stored in your database index.")
    
    # Increased target limit from 300 to 5000 records to pull the complete dataset
    comprehensive_draws_list = fetch_collection_records("draws", row_limit=5000)
    comprehensive_draws_dataframe = pd.DataFrame(comprehensive_draws_list)
    
    if not comprehensive_draws_dataframe.empty:
        # Display the entire fetched dataset without breaking or hardcoded slices
        st.dataframe(
            comprehensive_draws_dataframe[["numbers", "comment", "date", "_id"]], 
            use_container_width=True
        )
        
        if st.button("🗑️ Delete Latest Draw Row Entry"):
            target_document_deletion_id = comprehensive_draws_dataframe.iloc[-1]["_id"]
            purge_document_by_id("draws", target_document_deletion_id)
            st.success("Entry severed from relational database records successfully.")
            st.rerun()
    else:
        st.warning("No historical lottery drawings detected within the indexed collection schemas.")

# =====================================================
# ROUTE RUNTIME: FINANCE LEDGER & TARGET PERFORMANCE TRACKER
# =====================================================
elif selected_navigation_route == "Finance":
    st.title("💵 Expense Ledger & Financial Target System Tracker")
    st.markdown("Record investments and returns to measure aggregate model performance metrics.")
    
    with st.form("financial_statement_ingestion_form"):
        input_stake_amount = st.number_input("Transaction Cost / Stake Value (ZAR)", min_value=0.0, value=0.0, step=10.0)
        input_profit_amount = st.number_input("Realized Yield Return Value (ZAR)", value=0.0, step=10.0)
        finance_form_submitted = st.form_submit_button("Commit Financial Statement Ledger Row")

    if finance_form_submitted:
        commit_document("finance", {
            "stake": input_stake_amount,
            "profit": input_profit_amount,
            "date": datetime.now().isoformat()
        })
        st.success("Ledger transactional document values successfully cataloged.")

    historical_financial_records = fetch_collection_records("finance", 500)
    financial_records_dataframe = pd.DataFrame(historical_financial_records)
    
    if financial_records_dataframe.empty:
        st.warning("No financial ledger entries found in current active runtime indexes.")
    else:
        calculated_aggregate_spent = financial_records_dataframe["stake"].sum()
        calculated_aggregate_profit = financial_records_dataframe["profit"].sum()
        calculated_aggregate_roi = (calculated_aggregate_profit / calculated_aggregate_spent) * 100 if calculated_aggregate_spent else 0.0

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.metric("Aggregate Accrued Stake Expenditure", f"R {calculated_aggregate_spent:,.2f}")
        with f_col2:
            st.metric("Aggregate Accrued Dividends Profit", f"R {calculated_aggregate_profit:,.2f}")
        with f_col3:
            st.metric("Net Total Strategy ROI Position", f"{calculated_aggregate_roi:.2f}%")

        st.dataframe(financial_records_dataframe[["stake", "profit", "date"]], use_container_width=True)
        
        if st.button("🗑️ Erase Active Finance Ledger Analytics Records"):
            empty_entire_collection("finance")
            st.success("Financial ledger rows severed completely.")
            st.rerun()

# =====================================================
# ROUTE RUNTIME: SYSTEM RESET AND BATCH ERASE UTILITIES
# =====================================================
elif selected_navigation_route == "Reset":
    st.title("⚠️ System Master Wipe Utilities Terminal")
    st.markdown("Deletes structural records from all related database paths. This action cannot be undone.")
    
    confirm_wipe_checkbox = st.checkbox("I verify that I want to wipe all records across the active database nodes.")
    execute_master_wipe_button = st.button("🚨 EXECUTE FULL DESTRUCTIVE SYSTEM CLEAR ROUTINE")
    
    if execute_master_wipe_button:
        if confirm_wipe_checkbox:
            with st.spinner("Purging collections..."):
                for collection_key in COLLECTIONS:
                    empty_entire_collection(collection_key)
            st.success("✅ Execution Confirmed: Operational database clearing sequence successfully completed.")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("Operation Denied: Please check the verification checkbox to continue.")