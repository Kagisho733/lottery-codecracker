# ===============================
# LOTTERY AI SUPER SYSTEM PRO DASHBOARD - SaaS Edition
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import os, json, random
from datetime import datetime
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="🎰 Lottery AI PRO SaaS",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# FILE STORAGE
# -------------------------------
DRAW_FILE = "draws.json"

def save_draw(draw):
    data=[]
    if os.path.exists(DRAW_FILE):
        with open(DRAW_FILE,"r") as f: data=json.load(f)
    data.append({"numbers": draw,"date": str(datetime.now())})
    with open(DRAW_FILE,"w") as f: json.dump(data,f)

def load_draws():
    if not os.path.exists(DRAW_FILE): return []
    with open(DRAW_FILE,"r") as f: data=json.load(f)
    return [d.get("numbers",[]) for d in data]

def reset_data():
    if os.path.exists(DRAW_FILE): os.remove(DRAW_FILE)

# -------------------------------
# ANALYSIS & AI
# -------------------------------
def analyze(draws):
    numbers=list(range(1,25))
    all_nums=[n for d in draws for n in d]
    freq=Counter(all_nums)
    df=pd.DataFrame({"Number":numbers,"Frequency":[freq.get(n,0) for n in numbers]})
    df["Freq_norm"]=df["Frequency"]/max(df["Frequency"].max(),1)
    most=df.sort_values("Frequency",ascending=False).head(5)
    least=df.sort_values("Frequency",ascending=True).head(5)
    return df, most, least

def train_model(draws):
    if len(draws)<5: return None
    nums=list(range(1,25)); X=[]; y=[]
    for i in range(1,len(draws)):
        prev,curr=draws[i-1],draws[i]
        X.append([1 if n in prev else 0 for n in nums])
        y.append([1 if n in curr else 0 for n in nums])
    model=RandomForestClassifier()
    model.fit(X,y)
    return model

def ai_scores(draws,model):
    nums=list(range(1,25))
    last=draws[-1]; X=[[1 if n in last else 0 for n in nums]]
    probs=model.predict_proba(X)
    scores={}
    for i,n in enumerate(nums):
        scores[n]=probs[i][0][1] if len(probs[i][0])>1 else 0
    return scores

def calculate_probabilities(df):
    total=df["Hybrid"].sum()
    df["Prob"]=df["Hybrid"]/total if total>0 else 0
    return df

def ticket_probability(ticket,df):
    probs=df.set_index("Number")["Prob"].to_dict(); p=1
    for n in ticket: p*=probs.get(n,0.0001)
    return p

def expected_value(prob,payout): return prob*payout

def ticket_risk(ticket,df):
    scores=df.set_index("Number").loc[ticket]["Hybrid"]
    return np.std(scores)

def generate_ticket(df,count):
    hot=df.sort_values("Hybrid",ascending=False).head(12)["Number"].tolist()
    return sorted(random.sample(hot,count))

# -------------------------------
# COLOR FUNCTION
# -------------------------------
def card_color(is_best,risk,min_risk,max_risk):
    if is_best: return "#4CAF50"
    elif risk==min_risk: return "#2196F3"
    elif risk==max_risk: return "#F44336"
    else: return "#FFC107"

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("🎰 Lottery AI SaaS PRO")
page=st.sidebar.radio("Navigation",["Dashboard","Add Draw","Reset Database","History"])
theme=st.sidebar.radio("Theme",["💡 Light","🌙 Dark"])

# -------------------------------
# THEME STYLING
# -------------------------------
if theme=="🌙 Dark":
    st.markdown("""<style>.stApp{background-color:#121212;color:#fff;}</style>""",unsafe_allow_html=True)
else:
    st.markdown("""<style>.stApp{background-color:#f5f7fa;color:#333;}</style>""",unsafe_allow_html=True)

# -------------------------------
# ADD DRAW PAGE
# -------------------------------
if page=="Add Draw":
    st.header("➕ Add Draw")
    inp=st.text_input("Enter 12 numbers (comma separated)")
    if st.button("💾 Save Draw"):
        try:
            nums=list(map(int,inp.split(",")))
            if len(nums)==12: save_draw(nums); st.success("✅ Draw saved!")
            else: st.error("❌ Enter exactly 12 numbers")
        except: st.error("❌ Invalid input")

# -------------------------------
# RESET DATABASE PAGE
# -------------------------------
elif page=="Reset Database":
    st.header("⚠️ Reset Database")
    if st.button("🗑 Clear All Data"):
        reset_data(); st.warning("Database cleared!")

# -------------------------------
# HISTORY PAGE
# -------------------------------
elif page=="History":
    draws=load_draws()
    if draws:
        st.header("📜 Draw History")
        st.dataframe(pd.DataFrame(draws,columns=[f"Draw {i+1}" for i in range(len(draws[0]))]),use_container_width=True)
    else: st.warning("No data available")

# -------------------------------
# DASHBOARD PAGE
# -------------------------------
else:
    draws=load_draws()
    if not draws: st.warning("No data yet. Add draws first."); st.stop()
    st.header("📊 Lottery AI PRO Dashboard")

    # ANALYSIS
    df, most, least=analyze(draws)
    model=train_model(draws)
    df["AI_norm"]=0
    if model:
        ai=ai_scores(draws,model)
        df["AI"]=df["Number"].map(ai)
        df["AI_norm"]=df["AI"]/max(df["AI"].max(),1)
    df["Hybrid"]=(df["Freq_norm"]*0.5)+(df["AI_norm"]*0.5)
    df=calculate_probabilities(df)

    # BET SIMULATION SLIDER
    st.subheader("🎯 Bet Simulator")
    ball_count=st.slider("Select Number of Balls",1,8,3)
    payout_dict={1:1.4,2:3,3:8,4:17,5:41,6:111,7:351,8:1201}
    ticket=generate_ticket(df,ball_count)
    prob=ticket_probability(ticket,df)
    ev=expected_value(prob,payout_dict[ball_count])
    risk=ticket_risk(ticket,df)
    st.markdown(f"**🎟 Ticket:** {ticket}  |  **💰 Payout:** {payout_dict[ball_count]}x  |  **📊 Probability:** {prob:.6f}  |  **📈 EV:** {ev:.6f}  | ⚠️ Risk: {risk:.5f}")

    # TOP PREDICTED NUMBERS PANEL
    st.subheader("🔥 AI Predicted Numbers")
    top_numbers=df.sort_values("AI_norm",ascending=False).head(5)
    st.metric("Top Predicted Number",top_numbers.iloc[0]["Number"])
    st.write(top_numbers[["Number","AI_norm"]].reset_index(drop=True))

    # INTERACTIVE CHARTS
    st.subheader("📈 Analytics Charts")
    col1,col2=st.columns(2)
    with col1:
        fig_freq=px.bar(df,x="Number",y="Frequency",color="Frequency",color_continuous_scale="Viridis",
                        title="Number Frequency",labels={"Frequency":"Frequency"})
        st.plotly_chart(fig_freq,use_container_width=True)
    with col2:
        fig_hybrid=px.bar(df,x="Number",y="Hybrid",color="Hybrid",color_continuous_scale="Plasma",
                          title="Hybrid Score per Number",labels={"Hybrid":"Hybrid Score"})
        st.plotly_chart(fig_hybrid,use_container_width=True)

    # BET CARDS INTERACTIVE
    st.subheader("🎯 Smart Bet Recommendations")
    bet_data=[("1 Ball",1.4,1),("2 Balls",3,2),("3 Balls",8,3),("4 Balls",17,4),
              ("5 Balls",41,5),("6 Balls",111,6),("7 Balls",351,7),("8 Balls",1201,8)]
    results=[]
    for label,payout,count in bet_data:
        ticket=generate_ticket(df,count)
        prob=ticket_probability(ticket,df)
        ev=expected_value(prob,payout)
        risk=ticket_risk(ticket,df)
        results.append({"Bet Type":label,"Numbers":ticket,"Payout":payout,
                        "Probability":prob,"EV":ev,"Risk":risk})
    bet_df=pd.DataFrame(results)
    bet_df["Safety Rank"]=bet_df["Risk"].rank(method="min")
    bet_df=bet_df.sort_values("Risk")
    best_idx=bet_df["EV"].idxmax(); min_risk=bet_df["Risk"].min(); max_risk=bet_df["Risk"].max()

    cols=st.columns(3)
    for idx,row in bet_df.iterrows():
        is_best=row.name==best_idx
        col_idx=idx%3
        with cols[col_idx]:
            color=card_color(is_best,row["Risk"],min_risk,max_risk)
            st.markdown(f"""
            <div style="
                background:{color};
                border-radius:20px;
                padding:20px;
                margin-bottom:15px;
                color:white;
                transition: transform 0.2s;
            ">
                <h3 style="text-align:center">{row['Bet Type']} {"🟢 BEST BET" if is_best else ""}</h3>
                <p><strong>🎟 Numbers:</strong> {row['Numbers']}</p>
                <p><strong>💰 Payout:</strong> {row['Payout']}x</p>
                <p><strong>📊 Probability:</strong> {row['Probability']:.6f}</p>
                <p><strong>📈 EV:</strong> {row['EV']:.6f}</p>
                <p><strong>⚠️ Risk Score:</strong> {row['Risk']:.5f}</p>
            </div>
            """,unsafe_allow_html=True)

    # FULL TABLE
    st.subheader("📋 Full Analysis Table")
    st.dataframe(df.sort_values("Hybrid",ascending=False),use_container_width=True)