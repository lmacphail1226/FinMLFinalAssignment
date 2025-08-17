import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

# Define a function to keep only tickers where all values after year 1 are valid
def has_complete_data_after_year1(group):
    if len(group) <= 12:
        return False  # not enough data to evaluate
    after_year1 = group.iloc[12:]  # drop first 12 months
    return after_year1[['Return', 'Volatility', 'Sharpe Ratio']].notnull().all().all()


# Classify ETFs into asset classes
def classify_etf(name, category):
    text = (str(name) + " " + str(category)).lower()
    if "bond" in text or "treasury" in text:
        return "Fixed Income"
    elif (
        "real estate" in text
        or "reit" in text
        or "gold" in text
        or "commodity" in text
        or "crypto" in text
    ) and "goldman" not in text:
        return "Alternatives"
    elif (
        "short-term" in text
        or "t-bill" in text
        or "cash" in text
        or "ultra short" in text
    ):
        return "Liquidity"
    elif (
        "allocation" in text
        or "multi-asset" in text
        or "balanced" in text
        or "target" in text
    ):
        return "Blended / Multi-Asset"
    else:
        return "Equities"


def create_5yr_data(start_date, end_date, etf_df):

    date_format = "%Y-%m-%d"
    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)

    start_year = start_date.year
    end_year = end_date.year
    monthly_file_path = "etf_monthly_metrics_"+str(start_year)+"_"+str(end_year)+".csv" 
    sharpe_file_path = "etf_sharpe_summary_"+str(start_year)+"_"+str(end_year)+".csv" 

    if os.path.exists(monthly_file_path):
        os.remove(monthly_file_path)
        print(f"File '{monthly_file_path}' deleted successfully.")
    else:
        print(f"File '{monthly_file_path}' does not exist.")
        
    if os.path.exists(sharpe_file_path):
        os.remove(sharpe_file_path)
        print(f"File '{sharpe_file_path}' deleted successfully.")
    else:
        print(f"File '{sharpe_file_path}' does not exist.")



    # === CONFIGURATION ===
    BATCH_SIZE = 400
    RF_ANNUAL = 0.03
    RF_MONTHLY = RF_ANNUAL / 12
    MONTHLY_OUTPUT = monthly_file_path
    SHARPE_OUTPUT = sharpe_file_path

    # === LOAD TICKERS ===
    etf_list = etf_df
    tickers = etf_list['ticker'].dropna().unique().tolist()

    # === MAIN LOOP ===
    monthly_header_written = False
    sharpe_header_written = False

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i+BATCH_SIZE]
        print(f"\n📦 Batch {i//BATCH_SIZE + 1} — Downloading {len(batch)} ETFs...")

        try:
            data = yf.download(
                tickers=batch,
                start=start_date,
                end=end_date,
                interval="1mo",
                auto_adjust=True,
                group_by='ticker',
                progress=False,
                threads=True
            )
        except Exception as e:
            print(f"❌ Failed to download batch: {e}")
            continue

        monthly_frames = []
        sharpe_records = []

        for ticker in batch:
            try:
                df = data[ticker][['Close']].rename(columns={'Close': 'Adj Close'})
                df['Ticker'] = ticker
                df['Return'] = df['Adj Close'].pct_change(fill_method=None)
                df['Volatility'] = df['Return'].rolling(12).std() * np.sqrt(12)
                df['Sharpe Ratio'] = (df['Return'].rolling(12).mean() - RF_MONTHLY) / df['Return'].rolling(12).std()
                df = df.reset_index()

                # Append to monthly data
                monthly_frames.append(df)

                # Compute full-period Sharpe (excluding first 12 rows)
                df_clean = df.iloc[12:].copy()
                if len(df_clean) >= 12:
                    excess_ret = df_clean['Return'] - RF_MONTHLY
                    avg_ret = excess_ret.mean()
                    vol = df_clean['Return'].std()
                    sharpe = (avg_ret * 12) / (vol * np.sqrt(12))
                    sharpe_records.append({"Ticker": ticker, "5Y Sharpe Ratio": sharpe})
            except Exception as e:
                print(f"⚠️ Skipping {ticker}: {e}")

        # === SAVE MONTHLY DATA ===
        if monthly_frames:
            monthly_combined = pd.concat(monthly_frames, ignore_index=True)
            monthly_combined.to_csv(MONTHLY_OUTPUT, mode='a', header=not monthly_header_written, index=False)
            monthly_header_written = True
            print(f"✅ Saved {len(monthly_combined)} monthly rows.")

        # === SAVE 5Y SHARPE SUMMARY ===
        if sharpe_records:
            sharpe_df = pd.DataFrame(sharpe_records)
            sharpe_df.to_csv(SHARPE_OUTPUT, mode='a', header=not sharpe_header_written, index=False)
            sharpe_header_written = True
            print(f"✅ Saved {len(sharpe_df)} 5Y Sharpe scores.")

        time.sleep(1)  # avoid rate-limiting


    # Remove Data that has NAs

    monthly_df = pd.read_csv(monthly_file_path)

    monthly_df = monthly_df.sort_values(['Ticker', 'Date'])

    # Apply the filter
    filtered_df = monthly_df.groupby('Ticker').filter(has_complete_data_after_year1)

    filtered_df.to_csv("etf_monthly_metrics_filtered_"+str(start_year)+"_"+str(end_year)+".csv" , index=False)

    # Remove Same ETFs from Sharpe File

    sharpe_df = pd.read_csv(sharpe_file_path)

    filt = pd.DataFrame(filtered_df["Ticker"].unique()).rename(columns={0: 'Ticker'})

    sharpe_df_filt = sharpe_df.merge(filt,on="Ticker")
    
    # Classify Remaining ETFs into Asset Classes
    remaining_ETFs = sharpe_df_filt["Ticker"].unique()

    # Initialize classification list
    etf_classifications = []

    # Loop through tickers and classify
    for ticker in remaining_ETFs:
        print("classifying: "+ticker)
        try:
            info = yf.Ticker(ticker).info
            name = info.get('shortName', '')
            category = info.get('category', '')
            classification = classify_etf(name, category)
            etf_classifications.append({'Ticker': ticker, 'Name': name, 'Category': category, 'Class': classification})
        except Exception as e:
            etf_classifications.append({'Ticker': ticker, 'Name': '', 'Category': '', 'Class': 'Unclassified'})

    df_classified = pd.DataFrame(etf_classifications)

    df_classified.merge(sharpe_df_filt,on="Ticker").to_csv("etf_sharpe_summary_filtered_"+str(start_year)+"_"+str(end_year)+".csv",index=False)
