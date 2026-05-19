import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import textwrap
from st_aggrid import AgGrid, GridOptionsBuilder

# --- Page Config & Styling ---
st.set_page_config(page_title="Bank Churn Analytics", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Inter', sans-serif;
    }
    
    /* KPI Metric Cards (Dashboard) */
    .metric-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #E2E8F0;
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
    # Hide default Streamlit padding/header for the landing page
    st.markdown(textwrap.dedent("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 0rem !important; padding-bottom: 0rem !important; padding-left: 0rem !important; padding-right: 0rem !important; max-width: 100% !important;}
        
        .landing-body {
            background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
            color: #111827;
            margin: 0;
            padding: 0;
            overflow-x: hidden;
            width: 100%;
        }
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 80px;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .nav-logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 24px;
            font-weight: 800;
            color: #1e3a8a;
        }
        .nav-links {
            display: flex;
            gap: 32px;
        }
        .nav-link {
            text-decoration: none;
            color: #374151;
            font-weight: 600;
            font-size: 15px;
        }
        .nav-link.active {
            color: #2563eb;
            position: relative;
        }
        .nav-link.active::after {
            content: '';
            position: absolute;
            bottom: -6px;
            left: 0;
            width: 100%;
            height: 2px;
            background-color: #2563eb;
        }
        .nav-buttons {
            display: flex;
            gap: 16px;
        }
        .btn {
            padding: 10px 24px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 15px;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        .btn-outline {
            background: white;
            border: 1px solid #d1d5db;
            color: #374151;
        }
        .btn-primary {
            background: #2563eb;
            border: 1px solid #2563eb;
            color: white;
        }
        
        .hero-container {
            display: flex;
            padding: 80px 80px 40px 80px;
            justify-content: space-between;
            align-items: center;
        }
        .hero-text {
            flex: 1.2;
            padding-right: 40px;
        }
        .ai-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background-color: #eff6ff;
            color: #1e40af;
            padding: 6px 16px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-bottom: 24px;
        }
        .hero-title {
            font-size: 4.5rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 24px;
            color: #111827;
            margin-top: 0;
        }
        .hero-title span {
            color: #2563eb;
        }
        .hero-subtitle {
            font-size: 1.1rem;
            color: #4b5563;
            line-height: 1.6;
            margin-bottom: 40px;
            max-width: 90%;
        }
        .hero-actions {
            display: flex;
            gap: 16px;
            margin-bottom: 60px;
        }
        
        .features-row {
            display: flex;
            gap: 24px;
        }
        .feature {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .feature-icon {
            background: #eff6ff;
            color: #2563eb;
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        .feature h4 {
            margin: 0;
            font-size: 15px;
            font-weight: 700;
            color: #111827;
        }
        .feature p {
            margin: 0;
            font-size: 13px;
            color: #6b7280;
            line-height: 1.5;
        }
        
        .hero-image {
            flex: 1;
            position: relative;
        }
        .hero-image img {
            width: 100%;
            height: auto;
            border-radius: 20px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            border: 4px solid white;
        }
        
        .trusted-section {
            padding: 40px 80px 80px 80px;
            text-align: center;
            background: white;
        }
        .trusted-title {
            font-size: 12px;
            font-weight: 700;
            color: #9ca3af;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 32px;
        }
        .logos-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
            opacity: 0.8;
            max-width: 1000px;
            margin: 0 auto;
        }
        .logos-container img {
            height: 40px;
            max-width: 140px;
            object-fit: contain;
        }
        
        .shield-icon {
            width: 32px;
            height: 32px;
            fill: #2563eb;
        }
        
        .login-wrapper {
            padding: 80px 80px;
            background: #F8FAFC;
            border-top: 1px solid #e5e7eb;
        }
        </style>

        <div class="landing-body">
            <div class="navbar">
                <div class="nav-logo">
                    <svg class="shield-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                      <path d="M11 7h2v5h-2z" fill="#fff"/>
                      <path d="M11 14h2v2h-2z" fill="#fff"/>
                    </svg>
                    ChurnShield
                </div>
                <div class="nav-links">
                    <a href="#" class="nav-link active">Home</a>
                    <a href="#" class="nav-link">Product</a>
                    <a href="#" class="nav-link">Solutions</a>
                    <a href="#" class="nav-link">Insights</a>
                    <a href="#" class="nav-link">About Us</a>
                </div>
                <div class="nav-buttons">
                    <a href="#login-section" class="btn btn-outline">Login</a>
                    <a href="#login-section" class="btn btn-primary">Get Started</a>
                </div>
            </div>
            
            <div class="hero-container">
                <div class="hero-text">
                    <div class="ai-badge">
                        <span>✨</span> AI-POWERED BANKING SOLUTION
                    </div>
                    <h1 class="hero-title">Predict Churn.<br><span>Prevent Loss.</span></h1>
                    <p class="hero-subtitle">
                        ChurnShield uses advanced machine learning to identify at-risk customers, helping banks take proactive actions that improve retention and boost profitability.
                    </p>
                    <div class="hero-actions">
                        <a href="#login-section" class="btn btn-primary" style="padding: 14px 28px;">Explore Dashboard &rarr;</a>
                        <a href="#login-section" class="btn btn-outline" style="padding: 14px 28px;">How It Works &#9654;</a>
                    </div>
                    
                    <div class="features-row">
                        <div class="feature">
                            <div class="feature-icon">🧠</div>
                            <h4>AI Predictions</h4>
                            <p>Identify customers<br>likely to churn</p>
                        </div>
                        <div class="feature">
                            <div class="feature-icon">📊</div>
                            <h4>Smart Insights</h4>
                            <p>Understand key<br>churn drivers</p>
                        </div>
                        <div class="feature">
                            <div class="feature-icon">🛡️</div>
                            <h4>Actionable Strategies</h4>
                            <p>Take data-driven<br>retention actions</p>
                        </div>
                    </div>
                </div>
                
                <div class="hero-image">
                    <!-- The image has been replaced with a high-quality relevant Unsplash image. You can replace this src with your specific image path if needed -->
                    <img src="https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=1000&auto=format&fit=crop" alt="Bank Building">
                </div>
            </div>
            
            <div class="trusted-section">
                <div class="trusted-title">TRUSTED BY LEADING BANKS</div>
                <div class="logos-container">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/2/28/HDFC_Bank_Logo.svg" alt="HDFC Bank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/1/12/ICICI_Bank_Logo.svg" alt="ICICI Bank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/1/1a/Axis_Bank_logo.svg" alt="Axis Bank">
                    <img src="https://upload.wikimedia.org/wikipedia/en/5/58/State_Bank_of_India_logo.svg" alt="SBI">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Kotak_Mahindra_Bank_logo.svg/2560px-Kotak_Mahindra_Bank_logo.svg.png" alt="Kotak Mahindra Bank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/IndusInd_Bank_logo.svg/2560px-IndusInd_Bank_logo.svg.png" alt="IndusInd Bank">
                </div>
            </div>
            <div id="login-section" class="login-wrapper"></div>
        </div>
    """), unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; margin-top: -60px; margin-bottom: 30px; position: relative; z-index: 10; color: #111827;'>🔐 Login to Access Dashboard</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            user = st.text_input("Username (hint: admin)")
            pwd = st.text_input("Password (hint: secret)", type="password")
            submit = st.form_submit_button("Login to Dashboard", use_container_width=True)
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
