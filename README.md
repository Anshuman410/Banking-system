# ChurnShield: Bank Churn Analytics & Prediction System

ChurnShield is an end-to-end, AI-powered banking solution designed to predict customer churn, explain the underlying risk factors, and recommend actionable retention strategies. It features a scalable FastAPI backend for machine learning inference and a modern, interactive Streamlit frontend for bank administrators and customers.

## Why is this model needed?
Customer retention is significantly more cost-effective than customer acquisition. In the banking sector, identifying exactly *which* customers are likely to leave and understanding *why* they are leaving is critical to preventing revenue loss. 

This model solves this by:
1. **Predicting Churn:** Uses an XGBoost classifier to accurately predict the probability of a customer exiting the bank.
2. **Explainability (SHAP):** Unlike "black-box" models, this system uses SHAP (SHapley Additive exPlanations) to explain the local feature contributions for every single prediction, telling you exactly which factors (e.g., low balance, age, inactivity) are driving the churn risk.
3. **Customer Segmentation:** Uses K-Means clustering to group at-risk customers into actionable segments (e.g., "High Balance At-Risk", "Low Engagement"), mapping them to specific retention strategies.

## Project Structure
```text
.
├── data/                  # Generated datasets (churn_data.csv)
├── images/                # UI Assets and Logos
├── models/                # Serialized ML Models (XGBoost, KMeans, SHAP, preprocessor)
├── src/
│   ├── backend/           # FastAPI application (app.py)
│   └── frontend/          # Streamlit dashboard (dashboard.py)
├── data_gen.py            # Script to generate synthetic banking data
├── train_model.py         # ML Pipeline script for training and exporting models
├── docker-compose.yml     # Docker orchestration
└── requirements.txt       # Python dependencies
```

## How to Reproduce the Model

If you want to train the model from scratch, follow these steps:

### 1. Environment Setup
It is recommended to use a Python virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate the Dataset
Run the data generation script. This creates a highly realistic, synthetic banking dataset (`data/churn_data.csv` and `data/historical_churn.csv`) with engineered features like CLV (Customer Lifetime Value) and NPS (Net Promoter Score).
```bash
python data_gen.py
```

### 3. Train the Model Pipeline
Execute the training script. This script performs data preprocessing, trains the XGBoost model, fits the K-Means clustering model for segmentation, calculates the SHAP explainer, and exports all artifacts as `.pkl` files to the `models/` directory.
```bash
python train_model.py
```

## How to Operate the Application

You can run the application either locally using Python or via Docker.

### Option A: Running Locally
You will need two terminal windows.

**Terminal 1: Start the Backend (FastAPI)**
```bash
# Ensure your virtual environment is activated
uvicorn src.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Start the Frontend (Streamlit)**
```bash
# Ensure your virtual environment is activated
# Set the backend URL if running on a different port (defaults to localhost:8000)
export BACKEND_URL="http://localhost:8000"  # On Windows: set BACKEND_URL=http://localhost:8000
streamlit run src/frontend/dashboard.py
```
*The Streamlit frontend will be available at `http://localhost:8501`.*

### Option B: Running with Docker Compose
If you have Docker installed, you can spin up the entire stack with a single command. The Docker configuration automatically builds both the frontend and backend services and links them.
```bash
docker-compose up --build
```

## Application Usage & Roles

The frontend application uses multi-page routing and supports two distinct roles:

1. **Admin User**
   - **Credentials:** `admin` / `secret`
   - **Access:** The comprehensive Admin Dashboard. Features top-level KPIs (Churn Rate, Retention, Revenue Churn, Average CLV), time-series trend charts, and an interactive AG-Grid table of at-risk customers. Selecting a customer reveals real-time SHAP drill-downs and recommended actions.

2. **Customer User**
   - **Access:** Customers can self-register using the "Register as Customer" button on the login page.
   - **Dashboard:** A personalized, simplified Customer Portal showing account status, current balance, and targeted retention offers (simulated based on their segment).
