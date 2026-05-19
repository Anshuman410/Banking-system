import pandas as pd
import numpy as np
import os

def generate_data(num_samples=10000):
    np.random.seed(42)
    
    # Generate features
    customer_id = np.arange(1, num_samples + 1)
    credit_score = np.random.normal(650, 80, num_samples).astype(int)
    credit_score = np.clip(credit_score, 300, 850)
    
    geography = np.random.choice(['France', 'Spain', 'Germany'], size=num_samples, p=[0.5, 0.25, 0.25])
    gender = np.random.choice(['Male', 'Female'], size=num_samples, p=[0.54, 0.46])
    age = np.random.normal(38, 10, num_samples).astype(int)
    age = np.clip(age, 18, 92)
    
    tenure = np.random.randint(0, 11, num_samples)
    
    # Balance: some people have 0 balance
    has_balance = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
    balance = has_balance * np.random.normal(120000, 30000, num_samples)
    balance = np.clip(balance, 0, 250000)
    
    num_products = np.random.choice([1, 2, 3, 4], size=num_samples, p=[0.5, 0.45, 0.04, 0.01])
    has_crcard = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
    is_active_member = np.random.choice([0, 1], size=num_samples, p=[0.48, 0.52])
    estimated_salary = np.random.uniform(10000, 200000, num_samples)
    
    # Create a base dataframe to calculate probabilities
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
        'EstimatedSalary': estimated_salary
    })
    
    # Calculate churn probability based on some rules to make it realistic
    # Higher age -> higher churn probability up to a point
    prob = np.zeros(num_samples)
    prob += (df['Age'] - 30) * 0.01
    prob += (df['Geography'] == 'Germany').astype(float) * 0.15
    prob += (df['Gender'] == 'Female').astype(float) * 0.05
    prob -= (df['IsActiveMember'] == 1).astype(float) * 0.15
    prob += (df['NumOfProducts'] == 3).astype(float) * 0.2
    prob += (df['NumOfProducts'] == 4).astype(float) * 0.5
    prob -= (df['NumOfProducts'] == 2).astype(float) * 0.1
    prob -= (df['CreditScore'] - 650) * 0.0005
    
    # Normalize probabilities to be roughly between 0 and 1
    prob = 1 / (1 + np.exp(-prob)) # Sigmoid to keep it in [0, 1]
    
    # Adjust overall churn rate to ~20%
    threshold = np.percentile(prob, 80)
    churn = (prob >= threshold).astype(int)
    
    df['Exited'] = churn
    
    # Make sure data directory exists
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/churn_data.csv', index=False)
    print(f"Generated {num_samples} samples and saved to data/churn_data.csv")
    print(f"Overall Churn Rate: {df['Exited'].mean():.2%}")

if __name__ == "__main__":
    generate_data()
