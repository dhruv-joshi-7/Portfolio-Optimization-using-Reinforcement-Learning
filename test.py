from nsepy import get_history
import bs4 as bs
import pickle
import requests
import pandas as pd
import datetime as dt
import os
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')
pd.core.common.is_list_like = pd.api.types.is_list_like

#FUNCTION TO SAVE TICKERS

def save_SP():
    tickers = []
    resp = requests.get('https://en.wikipedia.org/wiki/NIFTY_50')
    soup = bs.BeautifulSoup(resp.text,'lxml')
    table = soup.find('table',{'id':'constituents'})
    
    for row in table.find_all('tr')[1:]:
        ticker = row.find_all('td')[1].text.replace('\n','')
        if "." in ticker:
            ticker = ticker.replace('.NS','')
            print('ticker replaced to', ticker) 
        tickers.append(ticker)
        
        with open("NIFTYticker.pickle","wb") as f:
            pickle.dump(tickers,f)
    return tickers    
 
tickers = save_SP()
print(tickers)
