#!/usr/bin/env python
# coding: utf-8

# ### Select Assets based on highest aggregate Sharpe Ratio

# In[7]:


import pandas as pd
import numpy as np
import cvxpy as cp

# Load the data
df_input = pd.read_csv("etf_sharpe_summary_filtered.csv")
df_input = df_input.dropna(subset=['5Y Sharpe Ratio'])

# Sort to find top 3 Sharpe ETFs
df_input = df_input.sort_values(by='5Y Sharpe Ratio', ascending=False).reset_index(drop=True)
df_input['Top3'] = 0
df_input.loc[:2, 'Top3'] = 1  # mark top 3

tickers = df_input['Ticker'].values
classes = df_input['Class'].values
sharpe = df_input['5Y Sharpe Ratio'].values
top3_flags = df_input['Top3'].values
n = len(tickers)

# Define desired ETF count range
min_etfs = 20
max_etfs = 40

# Class-level bounds on number of selected ETFs
class_bounds = {
    'Equities': (0.30, 0.40),
    'Fixed Income': (0.30, 0.40),
    'Alternatives': (0.10, 0.15),
    'Liquidity': (0.15, 0.15),
    'Blended / Multi-Asset': (0.00, 0.10)
}

# Optimization variables
w = cp.Variable(n)
z = cp.Variable(n, boolean=True)

# Constraints
constraints = [
    cp.sum(w) == 1,
    w >= (1 / max_etfs) * z,  # min 1/n if selected
    cp.sum(z) >= min_etfs,
    cp.sum(z) <= max_etfs
]

# ETF-level max weight: 8% for top 3, 5% for others
for i in range(n):
    max_wt = 0.08 if top3_flags[i] == 1 else 0.05
    constraints.append(w[i] <= max_wt * z[i])

# Class-based selection count constraints
for cls, (min_frac, max_frac) in class_bounds.items():
    idx = np.where(classes == cls)[0]
    if len(idx) > 0:
        constraints += [
            cp.sum(z[idx]) >= min_frac * min_etfs,
            cp.sum(z[idx]) <= max_frac * max_etfs
        ]

# Objective: maximize weighted Sharpe Ratio
objective = cp.Maximize(sharpe @ w)

# Solve the problem
problem = cp.Problem(objective, constraints)

try:
    problem.solve(solver=cp.GLPK_MI)
    df_input['Weight'] = w.value
    df_output = df_input[df_input['Weight'] > 0.001].sort_values(by='Weight', ascending=False)
    df_output.to_csv("final_portfolio_top3_exception.csv", index=False)
    print("✅ Optimization complete. Saved to final_portfolio_top3_exception.csv")
except Exception as e:
    print("❌ Optimization failed:", e)


# In[8]:


df_output


# In[ ]:




