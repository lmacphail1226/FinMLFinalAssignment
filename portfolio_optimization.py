import pandas as pd
import numpy as np
import cvxpy as cp

def optimize_portfolio(filepath,season):
    # Load the data
    df_input = pd.read_csv(filepath)
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
    min_etfs = 5

    # Class-level bounds on number of selected ETFs
    class_bounds = {
    'Equities': (0, 1),
    'Fixed Income': (0, 1),
    'Alternatives': (0, 1),
    'Liquidity': (0, 1),
    'Blended / Multi-Asset': (0, 1)
}

    # Optimization variables
    w = cp.Variable(n)
    z = cp.Variable(n, boolean=True)

    # Constraints
    constraints = [
        cp.sum(w) == 1,
        w <= 0.08,
        w >= (1 / n) * z,  # min 1/n if selected
        cp.sum(z) >= min_etfs
    ]

    # Class-based selection count constraints
    for cls, (min_frac, max_frac) in class_bounds.items():
        idx = np.where(classes == cls)[0]
        if len(idx) > 0:
            constraints += [
                cp.sum(z[idx]) >= min_frac * min_etfs,
                cp.sum(z[idx]) <= max_frac * min_etfs
            ]

    # Objective: maximize weighted Sharpe Ratio
    objective = cp.Maximize(sharpe @ w)

    # Solve the problem
    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.GLPK_MI)
    df_input['Weight'] = w.value
    df_output = df_input[df_input['Weight'] > 0.001].sort_values(by='Weight', ascending=False)
    
    return df_output