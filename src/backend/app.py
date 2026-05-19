from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Bank Churn Prediction API", description="API to predict customer churn probability with explainability.")

# Load models
try:
    full_pipeline = joblib.load('models/churn_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    xgb_model = joblib.load('models/xgboost_model.pkl')
    kmeans = joblib.load('models/kmeans_model.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
except FileNotFoundError:
    full_pipeline = None
    print("Warning: Models not found. Please run train_model.py first.")

# --- Authentication Mock ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock database
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    }
}

def fake_hash_password(password: str):
    return "fakehashed" + password

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    if user_dict['hashed_password'] != fake_hash_password(form_data.password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": user_dict["username"], "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_users_db.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user

# --- Prediction Schema ---
class CustomerData(BaseModel):
    CustomerId: int
    CreditScore: int = Field(..., ge=300, le=850)
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class ShapValue(BaseModel):
    feature: str
    value: float

class ChurnPrediction(BaseModel):
    customer_id: int
    churn_probability: float
    risk_level: str
    segment: str
    recommended_action: str
    shap_values: list[ShapValue]

def get_risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    else:
        return "Low"

def get_segment_strategy(segment_id: int) -> dict:
    strategies = {
        0: {"name": "Low Engagement", "action": "Send customized promotional emails and app push notifications."},
        1: {"name": "High Balance At-Risk", "action": "Immediate intervention: Personal banker call with premium retention offer."},
        2: {"name": "Recent Adopters", "action": "Cross-sell bundle offers and provide loyalty rewards."}
    }
    return strategies.get(segment_id, {"name": "General Risk", "action": "Monitor: Regular customer engagement."})

@app.get("/")
def read_root():
    return {"message": "Welcome to the Bank Churn Prediction API. Use /predict endpoint."}

@app.post("/predict", response_model=ChurnPrediction)
def predict_churn(data: CustomerData, current_user: dict = Depends(get_current_user)):
    if full_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded on the server.")

    # Convert request data to DataFrame
    input_data = data.dict()
    df = pd.DataFrame([input_data])
    
    # Feature Engineering
    df['Balance_to_Salary_Ratio'] = df['Balance'] / df['EstimatedSalary']
    df['IsActive_by_CreditCard'] = df['IsActiveMember'] * df['HasCrCard']
    X = df.drop(['CustomerId'], axis=1)

    try:
        # Preprocess manually to get processed data for SHAP and KMeans
        X_processed = preprocessor.transform(X)
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()
            
        # Predict probability using XGBoost
        churn_prob = xgb_model.predict_proba(X_processed)[0][1]
        risk_level = get_risk_level(churn_prob)
        
        # Get Segment (if High or Medium risk)
        segment_name = "N/A (Low Risk)"
        action = "Monitor: Regular customer engagement."
        if risk_level in ["High", "Medium"]:
            seg_id = kmeans.predict(X_processed)[0]
            seg_info = get_segment_strategy(seg_id)
            segment_name = seg_info["name"]
            action = seg_info["action"]

        # Calculate SHAP values
        shap_vals = explainer.shap_values(X_processed)[0]
        
        # Format SHAP output
        shap_list = [{"feature": fname, "value": float(sval)} for fname, sval in zip(feature_names, shap_vals)]
        # Sort by absolute magnitude to get top contributors
        shap_list = sorted(shap_list, key=lambda x: abs(x["value"]), reverse=True)[:5] # Top 5

        return ChurnPrediction(
            customer_id=input_data['CustomerId'],
            churn_probability=round(float(churn_prob), 4),
            risk_level=risk_level,
            segment=segment_name,
            recommended_action=action,
            shap_values=shap_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
