from openbb import obb
import pandas as pd
import os

def fetch_market_data(symbol="QQQ", start_date="2020-01-01", provider="yfinance"):
    """
    Fetches historical market data using OpenBB.
    """
    print(f"Fetching data for {symbol} from {start_date}...")
    
    try:
        # OpenBB v4 syntax
        df = obb.equity.price.historical(symbol, start_date=start_date, provider=provider).to_df()
        
        output_path = os.path.join("data", "raw", f"{symbol}_historical.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path)
        print(f"Data saved to {output_path}")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    fetch_market_data()
