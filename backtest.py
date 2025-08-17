import pandas as pd 
import numpy as np 
import yfinance as yf
from pandas.tseries.offsets import DateOffset

def normalize_tickers(tickers):
    import pandas as pd
    clean = []
    for t in tickers:
        if pd.isna(t):
            continue
        t = str(t).strip().strip('"').strip("'").upper()
        if t:
            clean.append(t)
    return list(dict.fromkeys(clean))

def load_prices_robust(tickers, start=None, end=None):
    """
    Fetch each ticker individually and OUTER-join on the union of real trading dates.
    Ensures a DatetimeIndex even when some tickers return empty.
    """
    import yfinance as yf
    import pandas as pd

    tickers = normalize_tickers(tickers)
    series_list = []
    idx_union = None

    for t in tickers:
        s = yf.download(t, start=start, end=end, auto_adjust=True, progress=False).get("Close")
        if s is None:
            s = pd.Series(name=t, dtype="float64")
        elif isinstance(s, pd.DataFrame):
            s = s.squeeze("columns")
        s = s.rename(t)

        # build union of indices from non-empty series
        if not s.empty:
            idx_union = s.index if idx_union is None else idx_union.union(s.index)
        series_list.append(s)

    if idx_union is None:
        raise ValueError("No data returned for any requested tickers (check symbols/date range/network).")

    # reindex every series to the union and concat
    frames = [(s.reindex(idx_union)).to_frame() for s in series_list]
    px = pd.concat(frames, axis=1).sort_index()
    px.index = pd.DatetimeIndex(px.index)  # force DatetimeIndex
    return px

def equity_to_monthly_returns(eq: pd.Series) -> pd.Series:
    return eq.pct_change().dropna()


def alpha_beta_vs_benchmark(rets_port: pd.Series, rets_bench: pd.Series, freq=12):
    df = pd.concat([rets_port.rename("rp"), rets_bench.rename("rb")], axis=1, join="inner").dropna()
    if len(df) < 3 or df["rb"].var(ddof=0) == 0:
        return np.nan, np.nan
    beta = df["rp"].cov(df["rb"]) / df["rb"].var()
    alpha_m = df["rp"].mean() - beta * df["rb"].mean()      # monthly intercept
    alpha_a = (1.0 + alpha_m)**freq - 1.0                   # annualized alpha
    return alpha_a, beta


def metrics_vs_bench(eq: pd.Series, bench_eq: pd.Series, freq=12):
    rp = equity_to_monthly_returns(eq)
    rb = equity_to_monthly_returns(bench_eq)
    if rp.empty:
        return {"Return": np.nan, "Volatility": np.nan, "Alpha": np.nan, "Beta": np.nan, "Sharpe": np.nan}
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    ann_vol   = rp.std(ddof=0) * np.sqrt(freq)
    sharpe    = (rp.mean() * freq) / ann_vol if ann_vol > 0 else np.nan
    alpha, beta = alpha_beta_vs_benchmark(rp, rb, freq=freq)
    return {"Return": total_ret, "Volatility": ann_vol, "Alpha": alpha, "Beta": beta, "Sharpe": sharpe}


def sixty_forty_equity_capital(mpx_all: pd.DataFrame, start, end=None,
                               capital: float = 100_000.0, w_spy=0.6, w_agg=0.4,
                               use_integer_shares: bool = False, include_leftover_cash: bool = True):
    # Reuse your existing helper for single-ticker buy&hold
    eq_spy = bh_benchmark_equity_capital(mpx_all, "SPY", start=start, end=end,
                                         capital=capital * w_spy,
                                         use_integer_shares=use_integer_shares,
                                         include_leftover_cash=include_leftover_cash)
    eq_agg = bh_benchmark_equity_capital(mpx_all, "AGG", start=start, end=end,
                                         capital=capital * w_agg,
                                         use_integer_shares=use_integer_shares,
                                         include_leftover_cash=include_leftover_cash)
    if eq_spy is None or eq_agg is None:
        return None
    eq_6040 = eq_spy.add(eq_agg, fill_value=0.0)
    eq_6040.name = "60/40 (SPY/AGG)"
    return eq_6040


def run_single_5y_capital_6040(alloc_df: pd.DataFrame, start: str, end: str | None = None,
                               capital: float = 100_000.0, use_integer_shares: bool = False):
    """
    Builds your buy-and-hold portfolio equity and a 60/40 (SPY/AGG) buy-and-hold benchmark
    over the same 5-year (or clipped) window, then computes metrics:
    Return, Volatility (ann), Alpha, Beta, Sharpe — alpha/beta vs 60/40.
    """
    # ---- load (reuse your robust loader + monthly) ----
    tickers = normalize_tickers(alloc_df["ticker"].tolist())
    px = load_prices_robust(tickers + ["SPY","AGG"], start=pd.to_datetime(start) - pd.DateOffset(years=1), end=end)
    mpx = to_monthly_last(px)

    # portfolio-only monthly table and exact 5Y span
    mpx_port = mpx[tickers].dropna(how="all")
    if mpx_port.empty:
        raise ValueError("No monthly data for portfolio tickers after resampling.")

    req_start = pd.to_datetime(start)
    eff_start = max(req_start, mpx_port.index.min())
    eff_end   = eff_start + pd.DateOffset(years=5)
    if end is not None:
        eff_end = min(eff_end, pd.to_datetime(end))
    eff_end   = min(eff_end, mpx_port.index.max())

    # ---- equities ----
    port_eq  = bh_equity_curve_by_capital(mpx_port, alloc_df, start=eff_start, end=eff_end,
                                          capital=capital, use_integer_shares=use_integer_shares)
    bench_eq = sixty_forty_equity_capital(mpx, start=eff_start, end=port_eq.index.max(),
                                          capital=capital, use_integer_shares=use_integer_shares)
    if bench_eq is None:
        raise ValueError("Failed to build 60/40 benchmark (SPY/AGG data missing at start).")

    # ---- metrics table ----
    rows = []
    rows.append({"Name": "Portfolio", **metrics_vs_bench(port_eq, bench_eq)})
    rows.append({"Name": "60/40 (SPY/AGG)", **metrics_vs_bench(bench_eq, bench_eq)})  # alpha=0, beta=1 numerically

    out = pd.DataFrame(rows)
    out.insert(1, "Start", port_eq.index.min().date().isoformat())
    out.insert(2, "End",   port_eq.index.max().date().isoformat())
    return out, port_eq, bench_eq


def to_monthly_last(px):
    import pandas as pd
    if not isinstance(px.index, pd.DatetimeIndex):
        try:
            px = px.copy()
            px.index = pd.to_datetime(px.index)
        except Exception as e:
            raise TypeError(f"Index is not datetime-like; got {type(px.index).__name__}") from e
    if px.empty:
        raise ValueError("Price table is empty after download.")
    return px.resample("ME").last().ffill()


def bh_equity_curve_by_capital(
    mpx: pd.DataFrame,
    alloc_df: pd.DataFrame,     # columns: ['ticker', 'Weight'] OR ['ticker','value']
    start, end=None,
    capital: float = 100_000.0,
    use_integer_shares: bool = False,
    include_leftover_cash: bool = True,
    scale_values_to_capital: bool = True,):
	
	"""
    Allocate 'capital' at the first month-end >= start.
    - If 'value' provided: per-ticker dollars at t₀ (optionally scaled to match 'capital').
    - Else if 'Weight' provided: dollars = capital * Weight / sum(Weights_at_start).
    - Compute shares at t₀, then portfolio value = sum_i shares_i * Price_i(t) (+ leftover cash).
    """
    # restrict to end if provided
	mpx = mpx.loc[:end] if end else mpx
    # snap to first available month-end >= start
	idx = mpx.index[mpx.index >= pd.to_datetime(start)]
	if len(idx) == 0:
		raise ValueError("No monthly bars on/after the requested start.")
	s = idx[0]

    # tickers with a price at start
	ok = list(mpx.columns[~mpx.loc[s].isna()])
	if len(ok) == 0:
		raise ValueError(f"No constituents have data at {s.date()}.")

    # build dollars per ticker at t0
	alloc_df = alloc_df.copy()
	alloc_df["ticker"] = alloc_df["ticker"].astype(str)
	if "value" in alloc_df.columns:
		dollars = (alloc_df.set_index("ticker")["value"].astype(float).reindex(ok).fillna(0.0))
		total = dollars.sum()
		leftover = 0.0
		if scale_values_to_capital:
			if total <= 0:
				raise ValueError("Sum of provided 'value' is zero; cannot scale.")
			dollars = dollars * (capital / total)
		else:
			leftover = capital - total
	elif "Weight" in alloc_df.columns:
		w = (alloc_df.set_index("ticker")["Weight"].astype(float).reindex(ok).fillna(0.0))
		wsum = w.sum()
		if wsum <= 0:
			raise ValueError("Weights sum to zero after aligning to available tickers at start.")
		dollars = capital * (w / wsum)
		leftover = 0.0
	else:
		raise ValueError("alloc_df must have 'Weight' or 'value' column.")

    # compute shares at start
	p0 = mpx.loc[s, ok]
	if use_integer_shares:
		shares = np.floor(dollars / p0)
		spent = (shares * p0).sum()
		leftover += float(capital - spent)  # keep uninvested as cash
	else:
		shares = dollars / p0
        # with fractional shares, 'spent' equals dollars; leftover already set by 'value' logic

    # equity curve: shares * price + leftover cash
	win = mpx.loc[s:]
	weights_shares = shares.reindex(ok) 

	eq = win[ok].mul(weights_shares, axis=1).sum(axis=1)
	if include_leftover_cash:
		eq = eq + leftover
	eq.name = "Portfolio"
	return eq


def series_metrics_from_equity(eq: pd.Series, freq=12):
    rets = eq.pct_change().dropna()
    if len(rets) == 0:
        return dict(CAGR=np.nan, AnnVol=np.nan, Sharpe=np.nan, MaxDD=np.nan, TotalReturn=np.nan)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    yrs = len(rets)/freq
    cagr = (1+total)**(1/yrs) - 1 if yrs > 0 else np.nan
    annvol = rets.std(ddof=0)*np.sqrt(freq)
    sharpe = cagr/annvol if annvol>0 else np.nan
    dd = (eq/eq.cummax() - 1).min()
    return dict(CAGR=cagr, AnnVol=annvol, Sharpe=sharpe, MaxDD=dd, TotalReturn=total)


def bh_benchmark_equity_capital(
    mpx: pd.DataFrame, ticker: str, start, end=None,
    capital: float = 100_000.0, use_integer_shares: bool = False, include_leftover_cash: bool = True
):
    if ticker not in mpx.columns: 
        return None
    mpx = mpx.loc[:end] if end else mpx
    idx = mpx.index[mpx.index >= pd.to_datetime(start)]
    if len(idx)==0 or np.isnan(mpx.loc[idx[0], ticker]): 
        return None
    s = idx[0]
    p0 = mpx.at[s, ticker]
    if use_integer_shares:
        shares = np.floor(capital / p0)
        leftover = capital - shares * p0
    else:
        shares = capital / p0
        leftover = 0.0
    eq = mpx[ticker].loc[s:] * shares
    if include_leftover_cash:
        eq = eq + leftover
    eq.name = ticker
    return eq


def coverage_report(mpx, tickers):
    sub = mpx[normalize_tickers(tickers)]
    # drop months where *all* portfolio columns are NaN
    sub = sub.dropna(how="all")
    first = sub.apply(lambda s: s.first_valid_index())
    last  = sub.apply(lambda s: s.last_valid_index())
    cnt   = sub.count()
    out = pd.DataFrame({"first": first, "last": last, "bars": cnt})
    out["first"] = out["first"].dt.date
    out["last"]  = out["last"].dt.date
    return out.sort_index()


def run_single_5y_capital(alloc_df: pd.DataFrame, start: str, capital: float = 100_000.0,
                          use_integer_shares: bool = False, end=None, require_full_5y: bool = False):
    import pandas as pd, numpy as np
    from pandas.tseries.offsets import DateOffset

    tickers = list(alloc_df["ticker"].values)
    bench = ["SPY","AGG"]

    # load combined prices so we can compare to benchmarks too
    px = load_prices_robust(tickers + bench, start=pd.to_datetime(start) - DateOffset(years=1), end=end)
    mpx = to_monthly_last(px)

    mpx_port = mpx[tickers]

    # --- sanity / coverage checks for your portfolio universe
    missing_cols = [t for t in tickers if t not in mpx.columns]
    if missing_cols:
        raise ValueError(f"These tickers returned no data: {missing_cols}")

    mpx_port = mpx[tickers].dropna(how="all")
    if mpx_port.empty:
        raise ValueError("No monthly data at all for your portfolio tickers (after resampling). "
                         "Double-check symbols and date range.")

    req_start = pd.to_datetime(start)
    port_first, port_last = mpx_port.index.min(), mpx_port.index.max()

    if port_last < req_start:
    	raise ValueError(
        "Requested start {req} is after the last available month {last} for your portfolio tickers.\n"
        "Portfolio coverage: {first} .. {last}.\n"
        "Pick a start <= {last} (e.g., {last}) or fix the tickers (exchange suffix, delisting, typos)."
        .format(req=req_start.date(), first=port_first.date(), last=port_last.date())
    )

    # Effective window: snap start to the earliest month that actually exists for your portfolio,
    # then cap to 5 years or available data, whichever is shorter.
    eff_start = max(req_start, port_first)
    eff_end = eff_start + DateOffset(years=5)
    if eff_end > port_last:
        if require_full_5y:
            raise ValueError(f"Not enough history for a full 5 years.\n"
                             f"Portfolio coverage: {port_first.date()}..{port_last.date()}, "
                             f"requested start: {req_start.date()}")
        # clip to available data
        eff_end = port_last

    # --- build portfolio equity (buy & hold at t0 using your weights/values)
    port_eq = bh_equity_curve_by_capital(
        mpx_port, alloc_df, start=eff_start, end=eff_end,
        capital=capital, use_integer_shares=use_integer_shares
    )
    eff_end = port_eq.index.max()  # align benches to actual end (after clipping)

    # --- benchmarks with same capital logic
    out = {"Portfolio": series_metrics_from_equity(port_eq)}
    for t in bench:
        beq = bh_benchmark_equity_capital(
            mpx, t, start=eff_start, end=eff_end,
            capital=capital, use_integer_shares=use_integer_shares
        )
        out[t] = series_metrics_from_equity(beq) if beq is not None else {k: np.nan for k in out["Portfolio"].keys()}

    # tidy comparison
    rows = []
    for name, met in out.items():
        rows.append({
            "Name": name,
            "Start": port_eq.index.min().date().isoformat(),
            "End":   port_eq.index.max().date().isoformat(),
            **{k: met[k] for k in ["TotalReturn","CAGR","AnnVol","Sharpe","MaxDD"]}
        })
    cmp_df = pd.DataFrame(rows).sort_values("Name").reset_index(drop=True)

    # convenience: add outperformance vs SPY/AGG on TotalReturn
    # After building cmp_df with rows for ["Portfolio", "SPY", "AGG"] (some may be missing)
    port_tr = cmp_df.loc[cmp_df["Name"] == "Portfolio", "TotalReturn"].iloc[0]

    for b in ["SPY", "AGG"]:
        bench = cmp_df.loc[cmp_df["Name"] == b, "TotalReturn"]
        bench_tr = bench.iloc[0] if not bench.empty else np.nan
        # assign a scalar so Pandas broadcasts to the whole column cleanly
        cmp_df[f"Port - {b} (TotalRet)"] = port_tr - bench_tr

    return cmp_df, port_eq



def rolling_5y_comparison_capital(
    alloc_df: pd.DataFrame, start="2000-01-01", end=None, capital: float = 100_000.0,
    step="ME", use_integer_shares: bool = False
):
    tickers = list(alloc_df["ticker"].values)
    bench = ["SPY","AGG"]
    px = load_prices_robust(tickers + bench, start=start, end=end)
    mpx = to_monthly_last(px)

    months = 60
    idx = mpx.index
    step_m = 1 if step == "ME" else 12
    rows = []
    for i in range(0, len(idx)-months+1, step_m):
        s, e = idx[i], idx[i+months-1]
        # portfolio
        try:
            port_eq = bh_equity_curve_by_capital(mpx[tickers], alloc_df, start=s, end=e,
                                                 capital=capital, use_integer_shares=use_integer_shares)
        except ValueError:
            continue
        stats_port = series_metrics_from_equity(port_eq)
        row = {"segment_start": s.date().isoformat(), "segment_end": e.date().isoformat(),
               **{f"Port_{k}": v for k,v in stats_port.items()}}

        # benches
        for b in bench:
            beq = bh_benchmark_equity_capital(mpx, b, start=s, end=e, capital=capital,
                                              use_integer_shares=use_integer_shares)
            stats_b = series_metrics_from_equity(beq) if beq is not None else {k: np.nan for k in stats_port}
            row.update({f"{b}_{k}": v for k,v in stats_b.items()})
            row[f"Port_minus_{b}_TotalRet"] = row["Port_TotalReturn"] - row.get(f"{b}_TotalReturn", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)
