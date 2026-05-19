import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from st_aggrid import AgGrid, GridOptionsBuilder

# --- Page Config & Styling ---
st.set_page_config(page_title="Bank Churn Analytics", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #333333;
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    .metric-card {
        background: #F8F9FA;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 20px;
    }
    .hero-section {
        text-align: center;
        padding: 50px 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- State Management ---
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None

# --- Helpers ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/churn_data.csv')
        trend = pd.read_csv('data/historical_churn.csv')
        return df, trend
    except FileNotFoundError:
        return None, None

def login(username, password):
    try:
        response = requests.post(f"{API_URL}/token", data={"username": username, "password": password})
        if response.status_code == 200:
            st.session_state.token = response.json().get("access_token")
            st.session_state.user = username
            return True
        return False
    except:
        return False

# --- Unauthenticated Views ---
def show_landing_page():
    # Top Nav imitation
    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
    with col1:
        st.markdown("### 🏦 Bank Churn Predictor")
    
    st.markdown("---")
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 style='color: #2C3E50;'>Predict & Prevent Customer Churn</h1>
        <h3 style='color: #34495E;'>Identify at-risk accounts and deploy retention offers immediately.</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    col_feat1.info("✔️ **Boost Retention**\nTarget interventions before customers leave.")
    col_feat2.info("✔️ **Data-Driven Actions**\nUse XGBoost and K-Means for smart segmentation.")
    col_feat3.info("✔️ **Explainable AI**\nUnderstand exactly why a customer is at risk via SHAP.")
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # About Section
    st.markdown("## ℹ️ About the Platform")
    st.write("Our AI-driven dashboard helps your team save revenue by reducing churn. See top KPIs and customer insights at a glance.")
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # Login Section
    st.markdown("## 🔐 Login to Access Dashboard")
    with st.form("login_form"):
        user = st.text_input("Username (hint: admin)")
        pwd = st.text_input("Password (hint: secret)", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if login(user, pwd):
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials.")

# --- Authenticated Views (Dashboard) ---
def show_dashboard():
    df, trend_df = load_data()
    if df is None:
        st.error("Data not found. Run data_gen.py")
        return
        
    st.title(f"📊 Dashboard Overview (Welcome, {st.session_state.user})")
    
    # Calculate 8 KPIs
    total_customers = len(df)
    churn_rate = df['Exited'].mean()
    crr = 1 - churn_rate
    
    churned_rev = df[df['Exited'] == 1]['MonthlyRevenue'].sum()
    total_rev = df['MonthlyRevenue'].sum()
    rev_churn = churned_rev / total_rev if total_rev > 0 else 0
    
    arpa = df['MonthlyRevenue'].mean()
    clv = df['CLV'].mean()
    nps = df['NPS'].mean()
    cross_sell = df['NumOfProducts'].mean()
    active_pct = df['IsActiveMember'].mean()
    
    # Render KPI Cards
    st.markdown("### Top KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><h4>Churn Rate</h4><h2>{churn_rate:.1%}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h4>Retention Rate</h4><h2>{crr:.1%}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h4>Revenue Churn</h4><h2>{rev_churn:.1%}</h2></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><h4>Avg Revenue (ARPA)</h4><h2>${arpa:,.0f}</h2></div>", unsafe_allow_html=True)
    
    c5, c6, c7, c8 = st.columns(4)
    c5.markdown(f"<div class='metric-card'><h4>Avg CLV</h4><h2>${clv:,.0f}</h2></div>", unsafe_allow_html=True)
    c6.markdown(f"<div class='metric-card'><h4>Avg NPS</h4><h2>{nps:.1f} / 10</h2></div>", unsafe_allow_html=True)
    c7.markdown(f"<div class='metric-card'><h4>Products / Customer</h4><h2>{cross_sell:.1f}</h2></div>", unsafe_allow_html=True)
    c8.markdown(f"<div class='metric-card'><h4>Active Members</h4><h2>{active_pct:.1%}</h2></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    ch1, ch2 = st.columns(2)
    with ch1:
        if trend_df is not None:
            fig1 = px.line(trend_df, x="Month", y="ChurnRate", title="Churn Trend over Time (Simulated)")
            fig1.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig1, use_container_width=True)
    with ch2:
        seg_churn = df.groupby('Geography')['Exited'].mean().reset_index()
        fig2 = px.bar(seg_churn, x="Geography", y="Exited", color="Geography", title="Churn by Region")
        fig2.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown("---")
    
    # Interactive Table using AgGrid
    st.markdown("### 📋 At-Risk Customer List")
    st.markdown("Filter and select a customer below to drill down into their specific risk factors.")
    
    # We'll display customers with higher churn probability (here we approximate with Exited=1 for demo or just show all)
    display_df = df[['CustomerId', 'Geography', 'Age', 'Balance', 'NumOfProducts', 'IsActiveMember', 'NPS', 'CLV', 'Exited']].head(500)
    
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
    gb.configure_selection('single', use_checkbox=True)
    gb.configure_default_column(editable=False, groupable=True, filter=True)
    grid_options = gb.build()
    
    grid_response = AgGrid(
        display_df,
        gridOptions=grid_options,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=True,
        theme='material',
        height=400
    )
    
    selected = grid_response['selected_rows']
    if selected is not None and len(selected) > 0:
        # Streamlit-aggrid sometimes returns dict, sometimes dataframe
        if isinstance(selected, pd.DataFrame):
            cust_id = selected.iloc[0]['CustomerId']
        else:
            cust_id = selected[0]['CustomerId']
            
        st.markdown(f"### 🔍 Detailed Analysis for Customer #{cust_id}")
        
        # Fetch actual row data
        cust_data = df[df['CustomerId'] == cust_id].iloc[0]
        
        payload = {
            "CustomerId": int(cust_data['CustomerId']),
            "CreditScore": int(cust_data['CreditScore']),
            "Geography": str(cust_data['Geography']),
            "Gender": str(cust_data['Gender']),
            "Age": int(cust_data['Age']),
            "Tenure": int(cust_data['Tenure']),
            "Balance": float(cust_data['Balance']),
            "NumOfProducts": int(cust_data['NumOfProducts']),
            "HasCrCard": int(cust_data['HasCrCard']),
            "IsActiveMember": int(cust_data['IsActiveMember']),
            "EstimatedSalary": float(cust_data['EstimatedSalary']),
            "MonthlyRevenue": float(cust_data['MonthlyRevenue']),
            "NPS": int(cust_data['NPS']),
            "CLV": float(cust_data['CLV'])
        }
        
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        
        with st.spinner("Fetching Real-Time SHAP Analysis..."):
            try:
                res = requests.post(f"{API_URL}/predict", json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    
                    risk_color = "#E74C3C" if data["risk_level"] == "High" else "#F39C12" if data["risk_level"] == "Medium" else "#2ECC71"
                    
                    rc1, rc2 = st.columns([1, 2])
                    with rc1:
                        st.markdown(f"**Risk Level:** <span style='color:{risk_color}; font-size: 20px'>{data['risk_level']} ({data['churn_probability']:.1%})</span>", unsafe_allow_html=True)
                        st.info(f"**Segment:** {data['segment']}\n\n**Action Plan:** {data['recommended_action']}")
                    
                    with rc2:
                        shap_data = pd.DataFrame(data["shap_values"])
                        shap_data['color'] = np.where(shap_data['value'] > 0, '#E74C3C', '#2ECC71')
                        fig = px.bar(shap_data, x="value", y="feature", orientation='h',
                                     title="Local Feature Contributions (SHAP)",
                                     color="color", color_discrete_map="identity")
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"API Connection Failed: {e}")


# --- Main Router ---
def main():
    if not st.session_state.token:
        # Unauthenticated: Landing Page
        show_landing_page()
    else:
        # Authenticated: Sidebar + Dashboard
        st.sidebar.title("🏦 Navigation")
        st.sidebar.success(f"User: {st.session_state.user}")
        menu = ["Dashboard (KPIs & Table)", "Logout"]
        choice = st.sidebar.radio("Go to", menu)
        
        if choice == "Dashboard (KPIs & Table)":
            show_dashboard()
        elif choice == "Logout":
            st.session_state.token = None
            st.session_state.user = None
            st.rerun()

if __name__ == "__main__":
    main()
