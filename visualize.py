import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/loan_data.csv")

# Plot 1: Approval Distribution
plt.figure(figsize=(7, 5))
counts = df["loan_approved"].value_counts()
plt.pie(counts, labels=["Approved", "Rejected"], autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"], startangle=90)
plt.title("Loan Approval Distribution")
plt.tight_layout()
plt.savefig("outputs/01_approval_distribution.png")
plt.close()
print("✅ Saved: 01_approval_distribution.png")

# Plot 2: Credit Score vs Approval
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="credit_score", hue="loan_approved",
             bins=30, palette={0: "#e74c3c", 1: "#2ecc71"}, alpha=0.7)
plt.title("Credit Score Distribution by Approval Status")
plt.xlabel("Credit Score")
plt.tight_layout()
plt.savefig("outputs/02_credit_score_vs_approval.png")
plt.close()
print("✅ Saved: 02_credit_score_vs_approval.png")

# Plot 3: Income vs Approval
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="loan_approved", y="income",
            palette={0: "#e74c3c", 1: "#2ecc71"})
plt.xticks([0, 1], ["Rejected", "Approved"])
plt.title("Income Distribution by Approval Status")
plt.ylabel("Income (₹)")
plt.tight_layout()
plt.savefig("outputs/03_income_vs_approval.png")
plt.close()
print("✅ Saved: 03_income_vs_approval.png")

# Plot 4: Loan Amount vs Approval
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="loan_approved", y="loan_amount",
            palette={0: "#e74c3c", 1: "#2ecc71"})
plt.xticks([0, 1], ["Rejected", "Approved"])
plt.title("Loan Amount by Approval Status")
plt.tight_layout()
plt.savefig("outputs/04_loan_amount_vs_approval.png")
plt.close()
print("✅ Saved: 04_loan_amount_vs_approval.png")

# Plot 5: Education vs Approval
plt.figure(figsize=(8, 5))
edu = df.groupby("education")["loan_approved"].mean() * 100
sns.barplot(x=edu.index, y=edu.values, palette="Set2")
plt.title("Approval Rate by Education")
plt.ylabel("Approval Rate (%)")
plt.tight_layout()
plt.savefig("outputs/05_education_vs_approval.png")
plt.close()
print("✅ Saved: 05_education_vs_approval.png")

# Plot 6: Property Area vs Approval
plt.figure(figsize=(8, 5))
area = df.groupby("property_area")["loan_approved"].mean() * 100
sns.barplot(x=area.index, y=area.values, palette="coolwarm")
plt.title("Approval Rate by Property Area")
plt.ylabel("Approval Rate (%)")
plt.tight_layout()
plt.savefig("outputs/06_property_area_vs_approval.png")
plt.close()
print("✅ Saved: 06_property_area_vs_approval.png")

# Plot 7: Employment Years vs Approval
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="loan_approved", y="employment_years",
            palette={0: "#e74c3c", 1: "#2ecc71"})
plt.xticks([0, 1], ["Rejected", "Approved"])
plt.title("Employment Years by Approval Status")
plt.tight_layout()
plt.savefig("outputs/07_employment_vs_approval.png")
plt.close()
print("✅ Saved: 07_employment_vs_approval.png")

# Plot 8: Correlation Heatmap
plt.figure(figsize=(10, 7))
corr = df[["age","income","loan_amount","credit_score",
           "employment_years","existing_loans","loan_approved"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/08_correlation_heatmap.png")
plt.close()
print("✅ Saved: 08_correlation_heatmap.png")
