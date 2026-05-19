import pandas as pd
import numpy as np
import os

def generate_data(num_samples=10000):
    np.random.seed(42)
    
    # Generate original features
    customer_id = np.arange(1, num_samples + 1)
    credit_score = np.random.normal(650, 80, num_samples).astype(int)
    credit_score = np.clip(credit_score, 300, 850)
    
    geography = np.random.choice(['France', 'Spain', 'Germany'], size=num_samples, p=[0.5, 0.25, 0.25])
    gender = np.random.choice(['Male', 'Female'], size=num_samples, p=[0.54, 0.46])
    age = np.random.normal(38, 10, num_samples).astype(int)
    age = np.clip(age, 18, 92)
    
    tenure = np.random.randint(0, 11, num_samples)
    
    has_balance = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
    balance = has_balance * np.random.normal(120000, 30000, num_samples)
    balance = np.clip(balance, 0, 250000)
    
    num_products = np.random.choice([1, 2, 3, 4], size=num_samples, p=[0.5, 0.45, 0.04, 0.01])
    has_crcard = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
    is_active_member = np.random.choice([0, 1], size=num_samples, p=[0.48, 0.52])
    estimated_salary = np.random.uniform(10000, 200000, num_samples)
    
    # Generate new KPIs / Features
    # Monthly Revenue (ARPA approximation): Based on balance and products
    monthly_revenue = (balance * 0.005) + (num_products * 50) + np.random.normal(20, 10, num_samples)
    monthly_revenue = np.clip(monthly_revenue, 10, 5000)
    
    # NPS (0 to 10)
    # Active members with high balance and more products tend to have higher NPS
    nps_base = 5 + (is_active_member * 2) + (num_products * 0.5) - ((age > 60) * 1)
    nps = np.random.normal(nps_base, 2, num_samples).astype(int)
    nps = np.clip(nps, 0, 10)
    
    # CLV (Customer Lifetime Value)
    # Approximation: Monthly revenue * 12 * (average lifespan 5 years)
    clv = monthly_revenue * 12 * np.random.uniform(2, 10, num_samples)
    
    df = pd.DataFrame({
        'CustomerId': customer_id,
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_crcard,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
        'MonthlyRevenue': monthly_revenue,
        'NPS': nps,
        'CLV': clv
    })
    
    # Churn probability calculation
    prob = np.zeros(num_samples)
    prob += (df['Age'] - 30) * 0.01
    prob += (df['Geography'] == 'Germany').astype(float) * 0.15
    prob += (df['Gender'] == 'Female').astype(float) * 0.05
    prob -= (df['IsActiveMember'] == 1).astype(float) * 0.15
    prob += (df['NumOfProducts'] == 3).astype(float) * 0.2
    prob += (df['NumOfProducts'] == 4).astype(float) * 0.5
    prob -= (df['NumOfProducts'] == 2).astype(float) * 0.1
    prob -= (df['CreditScore'] - 650) * 0.0005
    
    # Low NPS massively increases churn risk
    prob += (df['NPS'] < 6).astype(float) * 0.3
    # High NPS decreases churn risk
    prob -= (df['NPS'] > 8).astype(float) * 0.2
    
    prob = 1 / (1 + np.exp(-prob)) # Sigmoid
    
    threshold = np.percentile(prob, 80)
    churn = (prob >= threshold).astype(int)
    
    df['Exited'] = churn
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/churn_data.csv', index=False)
    
    # Generate some mock historical monthly churn data for the trend chart
    months = pd.date_range(start='2025-01-01', periods=12, freq='ME')
    base_churn = 0.20
    trend_churn = [base_churn + np.random.normal(0, 0.02) for _ in range(12)]
    trend_df = pd.DataFrame({'Month': months, 'ChurnRate': trend_churn})
    trend_df.to_csv('data/historical_churn.csv', index=False)
    
    print(f"Generated {num_samples} samples and saved to data/churn_data.csv")
    print(f"Generated historical trend data to data/historical_churn.csv")
    print(f"Overall Churn Rate: {df['Exited'].mean():.2%}")

if __name__ == "__main__":
    generate_data()
