import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Page Config & Styling ---
st.set_page_config(page_title="Bank Churn Analytics", layout="wide", page_icon="🏦")

# Inject Custom CSS for a White/Light Theme and Typography
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #333333;
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .css-1d391kg, .css-1lcbmhc { 
        background-color: #F8F9FA;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .metric-card {
        background: #F8F9FA;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

API_URL = "http://backend:8000" # For Docker, we'll configure it. If local, we'll try localhost
API_URL_LOCAL = "http://localhost:8000"

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
        return df
    except FileNotFoundError:
        return None

def login(username, password):
    try:
        response = requests.post(f"{API_URL_LOCAL}/token", data={"username": username, "password": password})
        if response.status_code == 200:
            st.session_state.token = response.json().get("access_token")
            st.session_state.user = username
            return True
        else:
            return False
    except:
        return False

# --- Views ---
def show_welcome():
    st.title("🏦 Welcome to Bank Churn Analytics")
    st.markdown("""
    ### Predict, Understand, and Prevent Customer Churn
    Our end-to-end Machine Learning platform helps financial institutions retain their most valuable assets: their customers. 
    
    Using advanced **XGBoost** models and **SHAP** explainability, we provide actionable intelligence on *why* customers leave and *how* to keep them.
    """)
    st.image("https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?auto=format&fit=crop&w=1200&q=80", caption="Modern Banking Solutions", use_column_width=True)

def show_about():
    st.title("ℹ️ About the Platform")
    st.markdown("""
    **Architecture & Technology Stack:**
    - **Frontend:** Streamlit (Python) for rapid, interactive analytics.
    - **Backend:** FastAPI for high-performance, asynchronous REST APIs.
    - **Modeling:** XGBoost for predictive accuracy, K-Means for segmentation.
    - **Explainability:** SHAP (SHapley Additive exPlanations) for global and local insights.
    
    **Security:** Role-based access control and JWT authentication (simulated for this demo).
    """)

def show_auth():
    st.title("🔐 Authentication")
    if st.session_state.token:
        st.success(f"Logged in as: {st.session_state.user}")
        if st.button("Logout"):
            st.session_state.token = None
            st.session_state.user = None
            st.rerun()
    else:
        st.markdown("Please log in to access the Dashboard and ROI Calculator.")
        with st.form("login_form"):
            user = st.text_input("Username (hint: admin)")
            pwd = st.text_input("Password (hint: secret)", type="password")
            submit = st.form_submit_button("Login")
            if submit:
                if login(user, pwd):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try admin/secret.")

def show_dashboard():
    st.title("📊 Churn Analytics Dashboard")
    if not st.session_state.token:
        st.warning("Please Login to access this feature.")
        return

    df = load_data()
    if df is None:
        st.error("Data not found. Please run data generation script.")
        return

    churn_rate = df['Exited'].mean()
    total = len(df)
    
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='metric-card'><h3>Total Customers</h3><h2>{total:,}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>Overall Churn Rate</h3><h2>{churn_rate:.2%}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><h3>Avg. Age</h3><h2>{df['Age'].mean():.1f}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # Real-time Scoring
    st.subheader("🔍 Real-time Customer Scoring & Explainability")
    
    with st.form("prediction_form"):
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            customer_id = st.number_input("Customer ID", value=10001)
            credit_score = st.slider("Credit Score", 300, 850, 600)
            geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col_in2:
            age = st.slider("Age", 18, 92, 45)
            tenure = st.slider("Tenure (Years)", 0, 10, 2)
            balance = st.number_input("Balance ($)", value=120000.0)
        with col_in3:
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
            has_crcard = st.selectbox("Has Credit Card?", [1, 0])
            is_active_member = st.selectbox("Is Active Member?", [1, 0], index=1) # 0 means not active
            estimated_salary = st.number_input("Estimated Salary ($)", value=80000.0)
            
        predict_btn = st.form_submit_button("Predict Risk")

    if predict_btn:
        payload = {
            "CustomerId": customer_id, "CreditScore": credit_score, "Geography": geography,
            "Gender": gender, "Age": age, "Tenure": tenure, "Balance": balance,
            "NumOfProducts": num_products, "HasCrCard": has_crcard, 
            "IsActiveMember": is_active_member, "EstimatedSalary": estimated_salary
        }
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        
        with st.spinner("Scoring customer and calculating SHAP values..."):
            try:
                res = requests.post(f"{API_URL_LOCAL}/predict", json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    
                    risk_color = "#E74C3C" if data["risk_level"] == "High" else "#F39C12" if data["risk_level"] == "Medium" else "#2ECC71"
                    
                    r_col1, r_col2 = st.columns(2)
                    with r_col1:
                        st.markdown(f"### Prediction: <span style='color:{risk_color}'>{data['risk_level']} Risk</span> ({data['churn_probability']:.1%})", unsafe_allow_html=True)
                        st.info(f"**Segment:** {data['segment']}\n\n**Action Plan:** {data['recommended_action']}")
                        
                    with r_col2:
                        st.markdown("### Top Factors Driving This Prediction")
                        # SHAP Waterfall Simulation
                        shap_data = pd.DataFrame(data["shap_values"])
                        shap_data['color'] = np.where(shap_data['value'] > 0, '#E74C3C', '#2ECC71')
                        fig = px.bar(shap_data, x="value", y="feature", orientation='h',
                                     title="SHAP Feature Importance (Local)",
                                     color="color", color_discrete_map="identity")
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")

def show_roi():
    st.title("💰 Business ROI Calculator")
    if not st.session_state.token:
        st.warning("Please Login to access this feature.")
        return

    st.markdown("Calculate the estimated financial impact of the churn prediction system.")
    
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Total Customer Base", value=50000)
        ARPU = st.number_input("Avg Revenue Per User ($/year)", value=12000.0)
        current_churn = st.slider("Current Churn Rate (%)", 0.0, 50.0, 20.0) / 100
        reduction = st.slider("Expected Relative Churn Reduction (%)", 0.0, 50.0, 15.0) / 100
        cost = st.number_input("Cost per Retention Offer ($)", value=500.0)

    with col2:
        lost_revenue = N * ARPU * current_churn
        retained = N * current_churn * reduction
        saved_revenue = retained * ARPU
        total_cost = retained * cost
        net_saving = saved_revenue - total_cost
        
        st.markdown(f"### Estimated Impact")
        st.markdown(f"**Current Lost Revenue:** ${lost_revenue:,.2f}")
        st.markdown(f"**Customers Retained:** {int(retained):,}")
        st.markdown(f"**Gross Revenue Saved:** ${saved_revenue:,.2f}")
        st.markdown(f"**Campaign Cost:** ${total_cost:,.2f}")
        st.markdown(f"## Net Annual Savings: <span style='color:#2ECC71'>${net_saving:,.2f}</span>", unsafe_allow_html=True)


# --- Navigation ---
def main():
    st.sidebar.title("🏦 Navigation")
    menu = ["Welcome", "About", "Login / Auth", "Churn Dashboard", "ROI Calculator"]
    choice = st.sidebar.radio("Go to", menu)
    
    st.sidebar.markdown("---")
    if st.session_state.token:
        st.sidebar.success(f"Logged in as {st.session_state.user}")
    else:
        st.sidebar.error("Not logged in")

    if choice == "Welcome":
        show_welcome()
    elif choice == "About":
        show_about()
    elif choice == "Login / Auth":
        show_auth()
    elif choice == "Churn Dashboard":
        show_dashboard()
    elif choice == "ROI Calculator":
        show_roi()

if __name__ == "__main__":
    main()
