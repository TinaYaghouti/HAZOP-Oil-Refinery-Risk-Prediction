import numpy as np
import pandas as pd
import os

os.makedirs("Data/OTHER_RISKS", exist_ok=True)

# Stragetic Risks

n = 5000
strategic_data = {
    "exchange_rate": np.random.uniform(40000, 60000, n),
    "oil_price": np.random.uniform(70, 120, n),
    "sanctions_index": np.random.uniform(0, 100, n)
}

df_strategic = pd.DataFrame(strategic_data)

def StrategicRisk(row):
    if row["sanctions_index"] > 80 or row["exchange_rate"] > 55000:
        return 2
    elif row["sanctions_index"] > 50 or row["oil_price"] < 80:
        return 1
    else:
        return 0
    
df_strategic["risk"] = df_strategic.apply(StrategicRisk, axis=1)
df_strategic.to_csv("Data/OTHER_RISKS/STRATEGIC_RISK.csv", index=False)

# Financial Risks

financial_data = {
    "usd_irr": np.random.uniform(40000, 60000, n),
    "inflation_rate": np.random.uniform(20, 50, n),
    "interest_rate": np.random.uniform(10, 30, n)
}

df_financial = pd.DataFrame(financial_data)

def FinancialRisk(row):
    if row["usd_irr"] > 60000 or row["inflation_rate"] > 40:
        return 2
    elif row["usd_irr"] > 50000 or row["inflation_rate"] > 30:
        return 1
    else:
        return 0
    
df_financial["risk"] = df_financial.apply(FinancialRisk, axis=1)
df_financial.to_csv("Data/OTHER_RISKS/FINANCIAL_RISK.csv", index=False)

# Legal Risks

legal_data = {
    "audit_score": np.random.uniform(0, 100, n),
    "contract_deviations": np.random.uniform(0, 50, n),
    "compliance_violations": np.random.uniform(0, 20, n)
}

df_legal = pd.DataFrame(legal_data)

def LegalRisk(row):
    if row["audit_score"] < 30 or row["contract_deviations"] > 40:
        return 2
    elif row["audit_score"] > 60 or row["contract_deviations"] > 20:
        return 1
    else:
        return 0
    
df_legal["risk"] = df_legal.apply(LegalRisk, axis=1)
df_legal.to_csv("Data/OTHER_RISKS/LEGAL_RISK.csv", index=False)

print("Data For Other Risks (strategic, financial, legal) Created Successfully")
print("Location: Data/OTHER_RISKS/")