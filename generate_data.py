import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

n = 2000

age = np.random.randint(21, 65, n)
income = np.random.randint(20000, 200000, n)
loan_amount = np.random.randint(50000, 1000000, n)
loan_term = np.random.choice([12, 24, 36, 48, 60, 120, 180, 240, 360], n)
credit_score = np.random.randint(300, 850, n)
employment_years = np.random.randint(0, 30, n)
existing_loans = np.random.randint(0, 5, n)
education = np.random.choice(["Graduate", "Not Graduate"], n)
self_employed = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], n)
coapplicant_income = np.random.randint(0, 80000, n)
dependents = np.random.choice([0, 1, 2, 3], n, p=[0.4, 0.3, 0.2, 0.1])

# Approval logic
score = (
    (credit_score > 650).astype(int) * 3 +
    (income > 60000).astype(int) * 2 +
    (employment_years > 3).astype(int) * 2 +
    (loan_amount < income * 5).astype(int) * 2 +
    (existing_loans < 2).astype(int) * 1 +
    (education == "Graduate").astype(int) * 1 +
    np.random.randint(0, 3, n)
)

approval = (score >= 7).astype(int)

df = pd.DataFrame({
    "age": age,
    "income": income,
    "coapplicant_income": coapplicant_income,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "credit_score": credit_score,
    "employment_years": employment_years,
    "existing_loans": existing_loans,
    "dependents": dependents,
    "education": education,
    "self_employed": self_employed,
    "property_area": property_area,
    "loan_approved": approval
})

df.to_csv("data/loan_data.csv", index=False)
print(f"✅ Dataset created: {len(df)} records")
print(f"   Approved : {approval.sum()} ({approval.mean()*100:.1f}%)")
print(f"   Rejected : {(1-approval).sum()} ({(1-approval.mean())*100:.1f}%)")
