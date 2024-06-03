import yfinance as yf
import pandas as pd
import os

# Define the list of ticker symbols
#ticker_symbols = ["^GSPC", "^NDX", "QQQ", "MSFT", "AAPL", "NVDA", "AMZN", "TSLA", "COST", "AVGO", "LULU"]
ticker_symbols = ["NQ=F", "QQQ", "MES=F" ]
# MES=F stands for Micro E-mini S&P 500 Index Futu. It is a contract offered by the CME Group 
#   that is one-tenth the size of the E-mini futures on the S&P 500 index. 
#   The contract is designed to help manage exposure to the 500 large-cap stocks 
#   that are tracked by the S&P 500 Index
# NQ=F Nasdaq 100

# Data interval
t_interval="1m"
#t_interval="1h"
#t_interval="1d"

# Ensure the 'stockdata' folder exists
output_folder = "stockdata"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to download and save historical data for a list of ticker symbols
def download_and_save_stock_data(ticker_symbols, output_folder):
    for ticker_symbol in ticker_symbols:
        # Fetch the historical data from the first day it started trading
        stock_data = yf.Ticker(ticker_symbol)
        #stock_hist = stock_data.history(period="max", auto_adjust=False)
        stock_hist = stock_data.history(period="max", interval=t_interval, auto_adjust=False)

        # Ensure that the data includes 'Adj Close'
        if 'Adj Close' not in stock_hist.columns:
            stock_hist['Adj Close'] = stock_data.history(period="max", interval=t_interval, auto_adjust=True)['Close']

        # Drop the 'Dividends' and 'Stock Splits' columns
        if 'Dividends' in stock_hist.columns:
            stock_hist = stock_hist.drop(columns=['Dividends'])
        if 'Stock Splits' in stock_hist.columns:
            stock_hist = stock_hist.drop(columns=['Stock Splits'])    
                   
        
        if len(stock_hist) > 1000:
            
            #print(stock_hist.head())    
            
            # Assuming you have already fetched the historical data and stored it in stock_hist_1min
            first_date = stock_hist.index[0].strftime("%Y-%m-%d")  # Convert to string in "YYYY-MM-DD" format
            last_date = stock_hist.index[-1].strftime("%Y-%m-%d")

            # Concatenate the strings
            result_string = f"{first_date}_{last_date}_" + t_interval
            #print(result_string)  # Replace this with your desired action (e.g., saving to a file)

            # Rename 'Date' column to 'Datetime'
            stock_hist.index.name = 'Datetime'
            stock_hist.reset_index(inplace=True)
        
            # Define the output filename
            output_file = os.path.join(output_folder, f"{ticker_symbol}_max_"+result_string+".csv")

            # Save the data to a CSV file
            stock_hist.to_csv(output_file, index=False)

            print(f"Saved historical data for {ticker_symbol} to {output_file}")

# Call the function with the list of ticker symbols and output folder
download_and_save_stock_data(ticker_symbols, output_folder)

