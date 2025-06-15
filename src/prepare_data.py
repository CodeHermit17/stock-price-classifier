from yfinance import download as yfdownload
import pandas as pd
import os
import numpy as np

in_path='/mnt/c/Users/kashy/Documents/Code/Projects/stock-price-classifier/data/raw'
out_path='/mnt/c/Users/kashy/Documents/Code/Projects/stock-price-classifier/data/processed'

ticker= ['HDFCBANK.NS', 'RELIANCE.NS', 'ITC.NS', 'WIPRO.NS', 'VEDL.NS']

def fetch_and_savetoraw(ticker,interval):
    df=yfdownload(
        ticker,
        start='2025-03-01',
        end='2025-06-14',
        interval=interval)
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: 'Date'})
    df.columns.name = None
    file_path=os.path.join(in_path,f"{ticker}.csv")
    df.to_csv(file_path)

def process_data(ticker):
    df=pd.read_csv(os.path.join(in_path,f"{ticker}.csv"))
    df[['SMA10','SMA20','Momentum','Daily_Return','SMA_diff','Range','Target']]=None
    df['SMA10']=df['Close'].rolling(window=10).sum() / 10
    df['SMA20']=df['Close'].rolling(window=20).sum() / 20
    df['Momentum']=df['Close'].diff()
    df['Daily_Return']=df['Close'].pct_change()
    df['SMA_diff']= df['Close']-df['SMA20']
    df['Range']=df['High']-df['Low']
    df['Target'] = np.where(df['Close'] - df['Close'].shift(-1) > 0, 1, 0)
    df=df.drop(columns=['Open','High','Low'])
    df.to_csv(os.path.join(out_path,f"{ticker}.csv"))


for ticks in ticker:
    fetch_and_savetoraw(ticks,'1d')
    process_data(ticks)    