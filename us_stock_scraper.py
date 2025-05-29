import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from io import StringIO
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_us_tickers():
    """
    Get all US stock tickers from multiple sources
    
    Returns:
    - List of US stock ticker symbols
    """
    tickers = set()
    
    try:
        # Method 1: Get tickers from NASDAQ
        nasdaq_url = "https://www.nasdaq.com/api/screener/stocks?tableonly=true&limit=0&download=true"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(nasdaq_url, headers=headers)
        if response.status_code == 200:
            df_nasdaq = pd.read_csv(StringIO(response.text))
            nasdaq_tickers = df_nasdaq['Symbol'].tolist()
            tickers.update(nasdaq_tickers)
            logger.info(f"Retrieved {len(nasdaq_tickers)} tickers from NASDAQ")
    except Exception as e:
        logger.warning(f"Failed to get NASDAQ tickers: {e}")
    
    try:
        # Method 2: Get S&P 500 tickers from Wikipedia
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_table = pd.read_html(sp500_url)[0]
        sp500_tickers = sp500_table['Symbol'].str.replace('.', '-').tolist()
        tickers.update(sp500_tickers)
        logger.info(f"Retrieved {len(sp500_tickers)} S&P 500 tickers")
    except Exception as e:
        logger.warning(f"Failed to get S&P 500 tickers: {e}")
    
    try:
        # Method 3: Get additional tickers from a comprehensive list
        # This uses a GitHub repository that maintains US stock lists
        github_url = "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/constituents.csv"
        df_github = pd.read_csv(github_url)
        github_tickers = df_github['Symbol'].tolist()
        tickers.update(github_tickers)
        logger.info(f"Retrieved {len(github_tickers)} additional tickers")
    except Exception as e:
        logger.warning(f"Failed to get GitHub tickers: {e}")
    
    # Filter out problematic tickers and clean the list
    clean_tickers = []
    for ticker in tickers:
        if ticker and isinstance(ticker, str):
            # Clean ticker symbol
            clean_ticker = ticker.strip().upper().replace('.', '-')
            # Filter out non-stock symbols and problematic tickers
            if (len(clean_ticker) <= 5 and 
                clean_ticker.isalpha() or '-' in clean_ticker and 
                not any(char in clean_ticker for char in ['/', '^', '='])):
                clean_tickers.append(clean_ticker)
    
    # Remove duplicates and sort
    clean_tickers = sorted(list(set(clean_tickers)))
    logger.info(f"Final ticker count: {len(clean_tickers)}")
    
    return clean_tickers

def fetch_stock_batch(ticker_batch, period="1mo", interval="1d", max_retries=3):
    """
    Fetch stock data for a batch of tickers with retry logic
    
    Parameters:
    - ticker_batch: list of ticker symbols
    - period: time period for data
    - interval: data interval
    - max_retries: maximum number of retry attempts
    
    Returns:
    - Dictionary with ticker as key and data as value
    """
    results = {}
    
    for ticker in ticker_batch:
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period=period, interval=interval)
                
                if not data.empty:
                    # Add ticker column for identification
                    data['Ticker'] = ticker
                    results[ticker] = data
                    logger.debug(f"Successfully fetched data for {ticker}")
                else:
                    logger.warning(f"No data available for {ticker}")
                
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                    time.sleep(1)  # Wait before retry
                else:
                    logger.error(f"Failed to fetch {ticker} after {max_retries} attempts: {e}")
    
    return results

def get_all_us_stocks_data(period="1mo", interval="1d", batch_size=50, max_workers=5):
    """
    Fetch stock data for all US stocks
    
    Parameters:
    - period: time period for data
    - interval: data interval
    - batch_size: number of stocks to process in each batch
    - max_workers: number of concurrent threads
    
    Returns:
    - DataFrame with all stock data
    """
    logger.info("Starting to fetch all US stock tickers...")
    all_tickers = get_all_us_tickers()
    
    if not all_tickers:
        logger.error("No tickers found!")
        return None
    
    logger.info(f"Fetching data for {len(all_tickers)} stocks...")
    
    # Split tickers into batches
    ticker_batches = [all_tickers[i:i + batch_size] for i in range(0, len(all_tickers), batch_size)]
    
    all_stock_data = []
    processed_count = 0
    
    # Process batches with threading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch jobs
        future_to_batch = {
            executor.submit(fetch_stock_batch, batch, period, interval): batch 
            for batch in ticker_batches
        }
        
        # Process completed batches
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch_results = future.result()
                
                # Convert batch results to list of DataFrames
                for ticker, data in batch_results.items():
                    all_stock_data.append(data)
                    processed_count += 1
                
                logger.info(f"Processed batch of {len(batch)} tickers. Total completed: {processed_count}")
                
                # Add small delay between batches to be respectful to the API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
    
    if all_stock_data:
        # Combine all data into a single DataFrame
        logger.info("Combining all stock data...")
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        # Reorder columns for better readability
        columns_order = ['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        if all(col in combined_df.columns for col in columns_order):
            remaining_cols = [col for col in combined_df.columns if col not in columns_order]
            combined_df = combined_df[columns_order + remaining_cols]
        
        logger.info(f"Successfully retrieved data for {len(combined_df)} stock entries")
        return combined_df
    else:
        logger.error("No stock data was successfully retrieved")
        return None

def get_stock_info_batch(tickers, batch_size=20):
    """
    Get detailed information for multiple stocks in batches
    
    Parameters:
    - tickers: list of ticker symbols
    - batch_size: number of stocks to process at once
    
    Returns:
    - DataFrame with stock information
    """
    all_info = []
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        logger.info(f"Processing info batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}")
        
        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Extract key information
                stock_info = {
                    'Ticker': ticker,
                    'Company_Name': info.get('shortName', 'N/A'),
                    'Sector': info.get('sector', 'N/A'),
                    'Industry': info.get('industry', 'N/A'),
                    'Current_Price': info.get('currentPrice', np.nan),
                    'Market_Cap': info.get('marketCap', np.nan),
                    'PE_Ratio': info.get('trailingPE', np.nan),
                    'PB_Ratio': info.get('priceToBook', np.nan),
                    'Dividend_Yield': info.get('dividendYield', np.nan),
                    '52_Week_High': info.get('fiftyTwoWeekHigh', np.nan),
                    '52_Week_Low': info.get('fiftyTwoWeekLow', np.nan),
                    'Beta': info.get('beta', np.nan),
                    'Volume': info.get('volume', np.nan),
                    'Average_Volume': info.get('averageVolume', np.nan)
                }
                
                all_info.append(stock_info)
                
            except Exception as e:
                logger.warning(f"Failed to get info for {ticker}: {e}")
        
        # Rate limiting
        time.sleep(1)
    
    if all_info:
        return pd.DataFrame(all_info)
    else:
        return None

def save_to_multiple_formats(data, base_filename):
    """
    Save data to multiple file formats
    
    Parameters:
    - data: DataFrame to save
    - base_filename: base filename without extension
    """
    try:
        # Save as CSV
        csv_filename = f"{base_filename}.csv"
        data.to_csv(csv_filename, index=False)
        logger.info(f"Data saved to {csv_filename}")
        
        # Save as Excel
        excel_filename = f"{base_filename}.xlsx"
        data.to_excel(excel_filename, index=False)
        logger.info(f"Data saved to {excel_filename}")
        
        # Save as Parquet (more efficient for large datasets)
        parquet_filename = f"{base_filename}.parquet"
        data.to_parquet(parquet_filename, index=False)
        logger.info(f"Data saved to {parquet_filename}")
        
    except Exception as e:
        logger.error(f"Error saving data: {e}")

def main():
    """
    Main function to scrape all US stocks data
    """
    start_time = datetime.now()
    logger.info("Starting US stocks data scraping process...")
    
    # Get stock price data for all US stocks
    logger.info("Fetching stock price data...")
    stock_data = get_all_us_stocks_data(period="1mo", interval="1d", batch_size=30, max_workers=3)
    
    if stock_data is not None and not stock_data.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_to_multiple_formats(stock_data, f"all_us_stocks_data_{timestamp}")
        
        # Get detailed stock information for unique tickers
        unique_tickers = stock_data['Ticker'].unique().tolist()[:500]  # Limit to first 500 for demo
        logger.info(f"Fetching detailed info for {len(unique_tickers)} stocks...")
        
        stock_info = get_stock_info_batch(unique_tickers, batch_size=15)
        if stock_info is not None and not stock_info.empty:
            save_to_multiple_formats(stock_info, f"all_us_stocks_info_{timestamp}")
        
        # Generate summary statistics
        logger.info("Generating summary statistics...")
        summary_stats = {
            'Total_Stocks_Processed': len(stock_data['Ticker'].unique()),
            'Total_Data_Points': len(stock_data),
            'Date_Range': f"{stock_data.index.min()} to {stock_data.index.max()}",
            'Average_Volume': stock_data['Volume'].mean(),
            'Total_Market_Activity': stock_data['Volume'].sum(),
            'Processing_Time': str(datetime.now() - start_time)
        }
        
        summary_df = pd.DataFrame([summary_stats])
        save_to_multiple_formats(summary_df, f"scraping_summary_{timestamp}")
        
        logger.info("Data scraping completed successfully!")
        logger.info(f"Total processing time: {datetime.now() - start_time}")
        
    else:
        logger.error("Failed to retrieve stock data")

if __name__ == "__main__":
    main()