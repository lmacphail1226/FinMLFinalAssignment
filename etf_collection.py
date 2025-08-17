import requests
import pandas as pd
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

API_KEY = "FWjYhAphGxZUquLTQxQRF0QpSzaLtOJS"

params = {
    "type": "ETF",
    "active": "true",
    "limit": 1000,
    "sort": "ticker",
    "apiKey": API_KEY
}

# Initial URL
url = "https://api.polygon.io/v3/reference/tickers"

def create_etf_list(API_KEY, params, url):

    all_etfs = []

    while True:
        response = requests.get(url, params=params)
        data = response.json()

        # Check for errors
        if response.status_code != 200 or "results" not in data:
            print("Error:", data)
            break

        all_etfs.extend(data["results"])

        # Get next_url and manually re-attach API key
        next_url = data.get("next_url")
        if not next_url:
            break

        # Rebuild next_url with API key
        parsed_url = urlparse(next_url)
        query = parse_qs(parsed_url.query)
        query["apiKey"] = API_KEY
        new_query_string = urlencode(query, doseq=True)
        url = urlunparse(parsed_url._replace(query=new_query_string))

        # Clear params for next iteration
        params = {}

    # Save results to CSV
    etf_df = pd.DataFrame(all_etfs)
    etf_df.to_csv("polygon_etf_list.csv", index=False)